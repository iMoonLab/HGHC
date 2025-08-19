import pickle
import numpy as np
import os
from scipy.sparse import csr_matrix
from pathlib import Path
from typing import Dict, Optional, Set, List, Tuple
from collections import OrderedDict
import json, hashlib
import torch
import torch.nn.functional as F

try:
    import scipy.sparse as sp
except Exception:
    sp = None


def ensure_torch_coo(H, device=None, dtype=torch.float32):
    """Convert input to a coalesced torch sparse COO tensor. Supports SciPy sparse, torch sparse (COO/CSR), torch dense, and numpy."""
    if isinstance(H, torch.Tensor):
        if H.layout == torch.sparse_coo:
            return H.coalesce().to(device)
        if H.layout == torch.sparse_csr:
            return H.to_sparse_coo().coalesce().to(device)
        return H.to(device=device, dtype=dtype).to_sparse_coo().coalesce()

    if sp is not None and sp.issparse(H):
        coo = H.tocoo()
        idx = np.vstack([coo.row, coo.col])
        indices = torch.as_tensor(idx, dtype=torch.long, device=device)
        if coo.data is None:
            values = torch.ones(coo.nnz, dtype=dtype, device=device)
        else:
            values = torch.as_tensor(coo.data, dtype=dtype, device=device)
        return torch.sparse_coo_tensor(indices, values, size=coo.shape, device=device).coalesce()

    dense = torch.as_tensor(H, dtype=dtype, device=device)
    return dense.to_sparse_coo().coalesce()


def pick_tail_classes(y_train: np.ndarray, threshold: float) -> List[int]:
    """Tail classes: count < threshold * head count (computed on train). If none, return all classes."""
    cls, cnt = np.unique(y_train, return_counts=True)
    head = int(cnt.max())
    tails = [int(c) for c, k in zip(cls, cnt) if k < head * threshold]
    if not tails:
        tails = list(map(int, cls))
    return tails


def load_and_align(feat_pkl: Path, lab_pkl: Path, idx2node: List[Tuple[str, str]]):
    """Align features/labels to idx2node order by ASIN intersection. Return X, y, kept_asins, asin2row."""
    with feat_pkl.open("rb") as f:
        F = pickle.load(f)
    with lab_pkl.open("rb") as f:
        L = pickle.load(f)

    X_all = F["features"].astype(np.float32, copy=False)
    asins_all = [str(a) for a in F["asins"]]
    y_all = L["labels"].astype(np.int64, copy=False)
    assert len(asins_all) == len(X_all) == len(y_all), "features/labels/asins 数量不一致"

    asin2row = {a: i for i, a in enumerate(asins_all)}
    rows = []
    kept_nodes = []
    for (_, a) in idx2node:
        i = asin2row.get(str(a))
        if i is not None:
            rows.append(i)
            kept_nodes.append(a)
    if not rows:
        raise RuntimeError("对齐后没有样本，请检查 asins 与特征/标签是否对应。")

    rows = np.asarray(rows, dtype=np.int64)
    X = X_all[rows]
    y = y_all[rows]
    kept_asins = [asins_all[i] for i in rows.tolist()]
    return X, y, kept_asins, asin2row


@torch.no_grad()
def _build_pos_sets(H_sub: torch.Tensor, n_nodes: int, device):
    """Build positive index pairs and per-node positive-edge sets from sparse H_sub."""
    Hc = ensure_torch_coo(H_sub, device=device)
    rows, cols = Hc.indices()
    P = rows.numel()
    pos_sets = [set() for _ in range(n_nodes)]
    r = rows.cpu().tolist()
    c = cols.cpu().tolist()
    for vi, ej in zip(r, c):
        pos_sets[vi].add(ej)
    return rows, cols, pos_sets, P


def bce_with_neg_sampling_sparse(
    Zn: torch.Tensor,
    Ze: torch.Tensor,
    S: torch.Tensor,
    H_sub: torch.Tensor,
    neg_per_pos: int = 5,
    max_try: int = 20,
) -> torch.Tensor:
    """Compute BCE with positive entries and negative sampling without forming dense scores."""
    device = Zn.device
    n_b, m_b = Zn.size(0), Ze.size(0)
    ZnS = Zn @ S
    rows, cols, pos_sets, P = _build_pos_sets(H_sub, n_nodes=n_b, device=device)
    rows = rows.to(device)
    cols = cols.to(device)
    pos_logits = (ZnS[rows] * Ze[cols]).sum(dim=1)
    neg_nodes = torch.repeat_interleave(rows, neg_per_pos)
    neg_edges = torch.empty(P * neg_per_pos, dtype=torch.long)

    import random
    neg_edges_cpu = []
    for vi in rows.cpu().tolist():
        ps = pos_sets[vi]
        for _ in range(neg_per_pos):
            e_try = None
            for _ in range(max_try):
                cand = random.randrange(m_b)
                if cand not in ps:
                    e_try = cand
                    break
            if e_try is None:
                e_try = random.randrange(m_b)
            neg_edges_cpu.append(e_try)
    neg_edges = torch.tensor(neg_edges_cpu, dtype=torch.long, device=device)

    neg_logits = (ZnS[neg_nodes] * Ze[neg_edges]).sum(dim=1)
    logits = torch.cat([pos_logits, neg_logits], dim=0)
    labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
    return F.binary_cross_entropy_with_logits(logits, labels)


def _build_csr_from_user_items(user2items_sorted: Dict[str, List[int]], n_items: int):
    """Build CSR for user→items and inverted CSR for items→users."""
    user_ids = np.array(list(user2items_sorted.keys()), dtype=object)
    lists = list(user2items_sorted.values())

    offsets_u = np.zeros(len(lists) + 1, dtype=np.int64)
    nnz = 0
    for i, li in enumerate(lists, start=1):
        nnz += len(li)
        offsets_u[i] = nnz
    indices_items = np.empty(nnz, dtype=np.int64)
    pos = 0
    for li in lists:
        L = len(li)
        if L:
            indices_items[pos:pos + L] = np.asarray(li, dtype=np.int64)
            pos += L

    users_all = np.repeat(np.arange(len(lists), dtype=np.int64), np.diff(offsets_u))
    items_all = indices_items
    deg_items = np.bincount(items_all, minlength=n_items).astype(np.int64)
    offsets_i = np.zeros(n_items + 1, dtype=np.int64)
    offsets_i[1:] = np.cumsum(deg_items)
    order = np.argsort(items_all, kind="stable")
    indices_users = users_all[order]

    return user_ids, offsets_u, indices_items, offsets_i, indices_users


def load_hypergraph(
    kept_asin2row: Dict[str, int],
    fallback_user_json: Optional[Path] = None,
    show_progress: bool = True,
    use_cache: bool = True,
    force_rebuild: bool = False,
):
    """Build hypergraph from merged_data JSON. Supports disk cache keyed by kept_asin2row mapping."""
    try:
        from tqdm.auto import tqdm as _tqdm
    except Exception:
        _tqdm = None

    def _wrap_tqdm(it, **kw):
        if show_progress and (_tqdm is not None):
            return _tqdm(it, **kw)
        return it

    if fallback_user_json is None:
        raise RuntimeError("必须提供 merged_data 的 JSON 路径（fallback_user_json）。")
    json_path = Path(fallback_user_json)
    if not json_path.exists():
        raise FileNotFoundError(f"未找到 JSON 文件：{json_path}")

    def _hash_mapping(m: Dict[str, int]) -> str:
        blob = json.dumps(sorted(m.items()), ensure_ascii=False).encode("utf-8")
        return hashlib.sha1(blob).hexdigest()

    cache_hash = _hash_mapping(kept_asin2row)
    base = json_path.with_suffix("")
    cache_npz  = base.with_name(base.name + f".csr-{cache_hash[:8]}.npz")
    cache_users= base.with_name(base.name + f".csr-{cache_hash[:8]}.users.json")
    cache_meta = base.with_name(base.name + f".csr-{cache_hash[:8]}.meta.json")

    if use_cache and (not force_rebuild) and cache_npz.exists() and cache_users.exists() and cache_meta.exists():
        try:
            with cache_meta.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            n_items_now = (max(kept_asin2row.values()) + 1) if kept_asin2row else 0
            if meta.get("hash") == cache_hash and meta.get("n_items") == n_items_now and meta.get("version") == "v1":
                arrs = np.load(cache_npz)
                with cache_users.open("r", encoding="utf-8") as f:
                    user_ids = np.array(json.load(f), dtype=str)
                offsets_u     = arrs["offsets_u"]
                indices_items = arrs["indices_items"]
                offsets_i     = arrs["offsets_i"]
                indices_users = arrs["indices_users"]
                return user_ids, offsets_u, indices_items, offsets_i, indices_users
        except Exception:
            pass

    with json_path.open("r", encoding="utf-8") as f:
        merged_data = json.load(f)
    if not isinstance(merged_data, list):
        raise ValueError(f"{json_path} 顶层必须是 list")

    user2rows = OrderedDict()
    for rec in _wrap_tqdm(merged_data, desc="映射 items→row", unit="user", total=len(merged_data)):
        uid = str(rec.get("user") or rec.get("user_id") or rec.get("uid") or "")
        if not uid:
            continue
        items = rec.get("items") or []
        rows = user2rows.get(uid)
        if rows is None:
            rows = set()
            user2rows[uid] = rows
        for itx in items:
            px = itx.get("prefix")
            a  = itx.get("asin")
            if not a:
                continue
            rid = kept_asin2row.get(f"{px}::{a}") if px is not None else None
            if rid is None:
                rid = kept_asin2row.get(str(a))
            if rid is not None:
                rows.add(int(rid))

    user2items_sorted = OrderedDict((uid, sorted(rows)) for uid, rows in user2rows.items())
    n_items = (max(kept_asin2row.values()) + 1) if kept_asin2row else 0

    user_ids, offsets_u, indices_items, offsets_i, indices_users = _build_csr_from_user_items(
        user2items_sorted, n_items
    )

    if use_cache:
        try:
            np.savez_compressed(
                cache_npz,
                offsets_u=np.asarray(offsets_u, dtype=np.int64),
                indices_items=np.asarray(indices_items, dtype=np.int64),
                offsets_i=np.asarray(offsets_i, dtype=np.int64),
                indices_users=np.asarray(indices_users, dtype=np.int64),
            )
            with cache_users.open("w", encoding="utf-8") as f:
                json.dump(list(map(str, user_ids)), f, ensure_ascii=False)
            meta = {
                "version": "v1",
                "hash": cache_hash,
                "n_items": int(n_items),
                "num_users": int(len(user_ids)),
                "source_json": json_path.name,
            }
            with cache_meta.open("w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False)
        except Exception as e:
            print(f"[WARN] 写缓存失败：{e}")

    return user_ids, offsets_u, indices_items, offsets_i, indices_users


def subgraph_from_vertices(V_sel: np.ndarray,
                           offsets_i, indices_users,
                           offsets_u, indices_items):
    """Build induced sub-hypergraph on selected items; return maps, local edge list, CSR H_sub, and local members per edge."""
    V_sel = np.asarray(V_sel, dtype=np.int64)
    V_set = set(V_sel.tolist())

    E_set: Set[int] = set()
    for v in V_sel:
        s, e = int(offsets_i[v]), int(offsets_i[v + 1])
        E_set.update(indices_users[s:e].tolist())

    if not E_set:
        raise RuntimeError("所采样顶点没有任何关联超边。")

    E_list = np.array(sorted(E_set), dtype=np.int64)
    E_map = {u: j for j, u in enumerate(E_list)}
    V_map = {v: i for i, v in enumerate(V_sel)}

    indptr = [0]
    indices = []
    edge_members_local: List[List[int]] = [[] for _ in range(len(E_list))]

    for v in V_sel:
        js_local = []
        s, e = int(offsets_i[v]), int(offsets_i[v + 1])
        for u in indices_users[s:e].tolist():
            if u not in E_map:
                continue
            j = E_map[u]
            js_local.append(j)
        js_local = sorted(set(js_local))
        indices.extend(js_local)
        indptr.append(len(indices))

    for j, u_global in enumerate(E_list):
        s, e = int(offsets_u[u_global]), int(offsets_u[u_global + 1])
        items_global = indices_items[s:e]
        for v in items_global:
            if v in V_set:
                edge_members_local[j].append(V_map[v])

    indptr = np.asarray(indptr, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    data = np.ones_like(indices, dtype=np.float32)
    H_sub = csr_matrix((data, indices, indptr),
                       shape=(len(V_sel), len(E_list)),
                       dtype=np.float32)

    return V_map, E_list, H_sub, edge_members_local


def load_idx2node_from_asins_pkl(p: Path) -> List[Tuple[str, str]]:
    """Parse various pickle schemas into a sorted list of (prefix, asin) ordered by idx."""
    with p.open("rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict):
        if "node_id_to_prefix" in obj and "node_id_to_asin" in obj:
            px = list(map(str, obj["node_id_to_prefix"]))
            aa = list(map(str, obj["node_id_to_asin"]))
            assert len(px) == len(aa)
            return list(zip(px, aa))

        if "pair_to_id" in obj:
            items = []
            for key, idx in obj["pair_to_id"].items():
                if isinstance(key, (tuple, list)) and len(key) == 2:
                    items.append((int(idx), (str(key[0]), str(key[1]))))
            if items:
                items.sort(key=lambda x: x[0])
                return [k for _, k in items]

        if "key_to_id_str" in obj:
            items = []
            for k, idx in obj["key_to_id_str"].items():
                if not isinstance(k, str):
                    continue
                for sep in ("::", "||", ",", "|", "\t"):
                    if sep in k:
                        px, a = k.split(sep, 1)
                        items.append((int(idx), (px, a)))
                        break
            if items:
                items.sort(key=lambda x: x[0])
                return [k for _, k in items]

        for k in ("idx2node", "node_key_list", "nodes"):
            if k in obj and isinstance(obj[k], list) and (not obj[k] or isinstance(obj[k][0], (tuple, list))):
                return [(str(a), str(b)) for (a, b) in obj[k]]

        if obj and all(isinstance(v, int) for v in obj.values()):
            items = []
            for key, idx in obj.items():
                if isinstance(key, (tuple, list)) and len(key) == 2:
                    items.append((int(idx), (str(key[0]), str(key[1]))))
                elif isinstance(key, str):
                    for sep in ("::", "||", ",", "|", "\t"):
                        if sep in key:
                            px, a = key.split(sep, 1)
                            items.append((int(idx), (px, a)))
                            break
            if items:
                items.sort(key=lambda x: x[0])
                return [k for _, k in items]

    raise ValueError(f"无法解析 {p} 为 idx -> (prefix, asin) 列表")


def stratified_split_ratio(y: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split into train/val/test with 9:1:1 ratio per class."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    tr, va, te = [], [], []
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        ntr = int(round(n * 9 / 11.0))
        nva = int(round(n * 1 / 11.0))
        if ntr + nva > n:
            nva = max(0, n - ntr)
        tr.append(idx[:ntr])
        va.append(idx[ntr:ntr + nva])
        te.append(idx[ntr + nva:])
    return np.concatenate(tr), np.concatenate(va), np.concatenate(te)


def compute_full_edge_features(H: sp.csr_matrix, X: np.ndarray) -> np.ndarray:
    """Compute edge features as mean of incident node features."""
    assert sp.isspmatrix_csr(H), "H must be CSR"
    deg_e = np.asarray(H.sum(axis=0)).ravel().astype(np.float32)
    deg_e = np.maximum(deg_e, 1e-8)
    Xe = (H.T @ X) / deg_e[:, None]
    return Xe.astype(np.float32, copy=False)


def per_class_batch_sample(pool: Dict[int, np.ndarray], k_per_class: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly sample k_per_class indices per class (with replacement if needed)."""
    picks = []
    for c, ids in pool.items():
        if ids.size == 0:
            continue
        if ids.size >= k_per_class:
            sel = rng.choice(ids, size=k_per_class, replace=False)
        else:
            sel = rng.choice(ids, size=k_per_class, replace=True)
        picks.append(sel)
    return np.concatenate(picks) if picks else np.empty(0, dtype=np.int64)


def build_clean_subgraph(
    V_sel_init: np.ndarray,
    y: np.ndarray,
    per_class_k: int,
    rng: np.random.Generator,
    offsets_i, indices_users, offsets_u, indices_items,
    max_retries: int = 5
):
    """Resample within-class until no local edge has degree==1; final fallback removes such edges."""
    V_sel = V_sel_init.copy()
    tried = 0
    while True:
        V_map, E_list, H_sub, edge_members_local = subgraph_from_vertices(
            V_sel, offsets_i, indices_users, offsets_u, indices_items
        )
        bad_edges = []
        for j, mem in enumerate(edge_members_local):
            if len(mem) <= 1:
                bad_edges.append(j)
        if not bad_edges:
            return V_map, E_list, H_sub, edge_members_local, V_sel
        if tried >= max_retries:
            keep_cols = np.array([j for j in range(len(E_list)) if j not in bad_edges], dtype=np.int64)
            if keep_cols.size == len(E_list):
                return V_map, E_list, H_sub, edge_members_local, V_sel
            E_list_new = E_list[keep_cols]
            H_sub = H_sub[:, keep_cols].tocsr()
            edge_members_local = [edge_members_local[j] for j in keep_cols.tolist()]
            return V_map, E_list_new, H_sub, edge_members_local, V_sel
        classes = {}
        for j in bad_edges:
            mem = edge_members_local[j]
            if not mem:
                continue
            v_local = mem[0]
            inv_map = {lv: gv for gv, lv in V_map.items()}
            v_global = inv_map[v_local]
            c = int(y[v_global])
            classes.setdefault(c, []).append(v_global)
        V_sel_set: Set[int] = set(V_sel.tolist())
        for c, globals_in_bad in classes.items():
            need = len(globals_in_bad)
            cand_all = np.where(y == c)[0]
            cand = np.array([g for g in cand_all if g not in V_sel_set], dtype=np.int64)
            if cand.size == 0:
                cand = cand_all
            rep = rng.choice(cand, size=need, replace=(cand.size < need))
            for old_g, new_g in zip(globals_in_bad, rep):
                if old_g in V_sel_set:
                    V_sel_set.remove(old_g)
                V_sel_set.add(int(new_g))
        V_sel = np.fromiter(V_sel_set, dtype=np.int64)
        if V_sel.size > per_class_k * len(classes.keys()):
            V_sel = rng.choice(V_sel, size=per_class_k * len(classes.keys()), replace=False)
        tried += 1



@torch.no_grad()
def compute_ze_full_memmap(
    H_items_users: sp.csr_matrix,
    X_items: np.ndarray,
    model,
    device: torch.device,
    out_path: Path,
    chunk_edges: int = 200_000,
    dtype: str = "float16",
):
    """Compute Ze for all edges in chunks and persist to memmap with meta."""
    H_items_users = H_items_users.tocsr()
    H_users_items = H_items_users.T.tocsr()
    M_edges = H_users_items.shape[0]
    probe = torch.zeros((1, X_items.shape[1]), dtype=torch.float32, device=device)
    d_embed = int(model.encoder_edge(probe).shape[1])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        os.remove(out_path)
    Ze_mm = np.memmap(out_path, mode="w+", dtype=dtype, shape=(M_edges, d_embed))
    print(f"[Ze] start: |E|={M_edges:,}  -> {out_path}  d_embed={d_embed}")
    for beg in range(0, M_edges, chunk_edges):
        end = min(beg + chunk_edges, M_edges)
        H_blk = H_users_items[beg:end, :]
        deg = np.asarray(H_blk.sum(axis=1)).ravel().astype(np.float32)
        deg = np.maximum(deg, 1e-8)
        Xe_blk = (H_blk @ X_items) / deg[:, None]
        Xe_t = torch.from_numpy(Xe_blk).to(device)
        Ze_blk = model.encoder_edge(Xe_t).cpu().numpy().astype(np.float32)
        Ze_mm[beg:end, :] = Ze_blk.astype(np.float16 if dtype == "float16" else np.float32)
        Ze_mm.flush()
        print(f"[Ze]    block {beg:,}~{end:,} done")
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump({"rows": M_edges, "d_embed": d_embed, "dtype": dtype}, f)
    print(f"[Ze] all done. meta -> {meta_path}")
    return Ze_mm, d_embed


def build_topM_class_edges(
    offsets_i: np.ndarray, indices_users: np.ndarray,
    y: np.ndarray, train_idx: np.ndarray, classes: List[int],
    topM: int = 2000
) -> Dict[int, np.ndarray]:
    """For each class, count edge occurrences in train set and return top-M edge ids."""
    cls2top = {}
    train_mask = np.zeros(len(y), dtype=bool)
    train_mask[train_idx] = True
    for c in classes:
        items_c = np.where((y == c) & train_mask)[0]
        if items_c.size == 0:
            cls2top[c] = np.empty(0, dtype=np.int64)
            continue
        starts = offsets_i[items_c]
        ends = offsets_i[items_c + 1]
        total = int((ends - starts).sum())
        if total == 0:
            cls2top[c] = np.empty(0, dtype=np.int64)
            continue
        edge_ids = np.empty(total, dtype=np.int64)
        pos = 0
        for s, e in zip(starts, ends):
            k = e - s
            if k:
                edge_ids[pos:pos+k] = indices_users[s:e]
                pos += k
        edge_ids = edge_ids[:pos]
        max_eid = int(edge_ids.max()) if edge_ids.size else -1
        if max_eid < 0:
            cls2top[c] = np.empty(0, dtype=np.int64)
            continue
        cnt = np.bincount(edge_ids, minlength=max_eid+1)
        top_js = np.argpartition(cnt, -min(topM, cnt.size))[-min(topM, cnt.size):]
        top_js = top_js[np.argsort(cnt[top_js])[::-1]]
        cls2top[c] = np.sort(top_js.astype(np.int64))
    return cls2top


def edges_of(v: int, offsets_i: np.ndarray, indices_users: np.ndarray) -> np.ndarray:
    """Return edge ids of node v."""
    s, e = int(offsets_i[v]), int(offsets_i[v+1])
    return indices_users[s:e]

def union_limited(a: np.ndarray, b: np.ndarray, cap: int) -> np.ndarray:
    """Union then truncate to cap (ascending)."""
    if a.size == 0 and b.size == 0:
        return a
    u = np.union1d(a, b)
    if u.size > cap:
        u = u[:cap]
    return u

def class_deg_target(
    c: int, y: np.ndarray, V_tr: np.ndarray,
    offsets_i: np.ndarray, max_cap: int
) -> float:
    """Average degree of train nodes in class c, clipped to [1, max_cap]."""
    ids = V_tr[y[V_tr] == c]
    if ids.size == 0:
        return 1.0
    deg = (offsets_i[ids+1] - offsets_i[ids]).astype(np.int64)
    mu = float(deg.mean()) if deg.size else 1.0
    mu = max(1.0, min(mu, float(max_cap)))
    return mu

