import torch.optim as optim
import time
from config import cfg
from utils import *


def train_hsmote_on_tails(
    model: HSMOTE,
    X: np.ndarray,
    y: np.ndarray,
    V_tr: np.ndarray,
    tails: List[int],
    Xe_full: np.ndarray,
    offsets_i, indices_users, offsets_u, indices_items,
    device: torch.device
):
    model.to(device).train()
    opt = optim.Adam(model.parameters(), lr=cfg.lr_hsmote, weight_decay=cfg.wd_hsmote)
    rng = np.random.default_rng(cfg.seed)
    pool: Dict[int, np.ndarray] = {c: V_tr[y[V_tr] == c] for c in tails}
    batches = int(getattr(cfg, "hsmote_batches", 1000))
    per_class_k = int(getattr(cfg, "per_class_train", 50))
    neg_k = int(getattr(cfg, "neg_per_pos", 5))
    log_every = int(getattr(cfg, "log_every", 1))
    print(f"\n[HSMOTE] start  | batches={batches}  per_class_k={per_class_k}  tails={len(pool)}  neg_per_pos={neg_k}")
    print("     batch     CE_rec    MSE_rec    CE_tot   |U|    |E|     n_nodes  elapsed    ETA     it/s")
    print("-"*90)
    t0 = time.time()
    for b in range(1, batches + 1):
        V_sel0 = per_class_batch_sample(pool, per_class_k, rng)
        if V_sel0.size == 0:
            raise RuntimeError("Tail classes are empty in training set.")
        V_map, E_list, H_sub, edge_members_local, V_sel = build_clean_subgraph(
            V_sel0, y, per_class_k, rng, offsets_i, indices_users, offsets_u, indices_items, max_retries=5
        )
        X_n_sub = X[V_sel].astype(np.float32, copy=False)
        X_e_sub = Xe_full[E_list].astype(np.float32, copy=False)
        Xn = torch.from_numpy(X_n_sub).to(device)
        Xe = torch.from_numpy(X_e_sub).to(device)
        Zn = model.encoder_node(Xn)
        Ze = model.encoder_edge(Xe)
        from load_data import ensure_torch_coo
        H_torch = ensure_torch_coo(H_sub, device=device)
        if hasattr(model, "bce_rec"):
            rec = model.bce_rec(Zn, Ze, H_torch, neg_per_pos=cfg.neg_per_pos)
        else:
            rec = _bce_rec_fallback(Zn, Ze, model.S, H_torch, cfg.neg_per_pos)
        X_hat = model.decoder_node(Zn)
        mse = nn.functional.mse_loss(X_hat, Xn)
        loss = rec + cfg.beta_mse * mse
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (b % log_every) == 0 or b == 1 or b == batches:
            elapsed = time.time() - t0
            itps = b / max(elapsed, 1e-9)
            eta = (batches - b) / max(itps, 1e-9)
            n_nodes = int(X_n_sub.shape[0])
            n_edges = int(len(E_list))
            union_edges = n_edges
            print(f"{b:5d}/{batches:<5d}  "
                  f"{float(rec):8.4f}  {float(mse):8.4f}  {float(loss):8.4f}  "
                  f"{union_edges:5d}  {n_edges:5d}  {n_nodes:8d}  "
                  f"{elapsed/60:6.2f}m  {eta/60:6.2f}m  {itps:6.2f}",
                  flush=True)
    print("-"*90)
    print(f"[HSMOTE] done in { (time.time()-t0)/60:.2f} min")
    torch.save(model, "cache/model.pth")
    return model


def _H_from_offsets(n_items: int, offsets_i, indices_users, n_users: int) -> csr_matrix:
    """
    直接用（物品→用户）的 CSR 构造 H：
      H ∈ {0,1}^{|𝓥|×|𝓔|}，CSR 格式：
        - indptr = offsets_i（行指针）
        - indices = indices_users（列号）
        - data = 1
    """
    data = np.ones_like(indices_users, dtype=np.float32)
    return csr_matrix((data, indices_users, offsets_i),
                      shape=(n_items, n_users), dtype=np.float32)


def _bce_rec_fallback(Zn, Ze, S, H_sub_torch, neg_per_pos: int):
    """Backward-compatible BCE reconstruction with negative sampling."""
    from load_data import bce_with_neg_sampling_sparse as bce_neg
    return bce_neg(Zn, Ze, S, H_sub_torch, neg_per_pos=neg_per_pos)


def _ensure_float32(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float32, copy=False)


@torch.no_grad()
def synthesize_vertices_bernoulli(
    model,
    X: np.ndarray, y: np.ndarray,
    H_items_users: sp.csr_matrix,
    V_tr: np.ndarray,
    tails: List[int],
    device: torch.device,
    target_tail_ratio: float = 0.5,
    ze_memmap_path: Optional[Path] = None,
    Ze_full_array: Optional[np.ndarray] = None,
    topM_class_edges: Optional[Dict[int, np.ndarray]] = None,
    topM: int = 2000,
    cand_cap_per_syn: int = 4000,
    max_edges_per_syn: int = 128,
    ze_score_chunk: int = 20_000,
    block_new_nodes: int = 2000,
):
    """Synthesize tail-class nodes via Bernoulli sampling over reduced edge candidates."""
    H_items_users = H_items_users.tocsr()
    N, M = H_items_users.shape
    offsets_i = H_items_users.indptr.astype(np.int64, copy=False)
    indices_users = H_items_users.indices.astype(np.int64, copy=False)
    y_tr = y[V_tr]
    _, cnt = np.unique(y_tr, return_counts=True)
    head = int(cnt.max()) if cnt.size else 0
    target = int(round(head * float(target_tail_ratio)))
    print(f"[SYN] target per tail class in train = {target}  (head={head}, ratio={target_tail_ratio:.0%})")
    if ze_memmap_path is not None:
        Ze_full = open_ze_memmap(ze_memmap_path, M_edges=M, fallback_dtype="float16")
        print(f"[Ze] opened memmap: shape={Ze_full.shape}, dtype={Ze_full.dtype}")
    elif Ze_full_array is not None:
        Ze_full = Ze_full_array
    else:
        raise ValueError("Either ze_memmap_path or Ze_full_array must be provided")
    if topM_class_edges is None:
        print("[SYN] building Top-M class edges ...")
        topM_class_edges = build_topM_class_edges(
            offsets_i, indices_users, y, V_tr, tails, topM=topM
        )
        print("[SYN] Top-M ready.")
    rng = np.random.default_rng(cfg.seed + 17)
    blocks_H = []
    blocks_X = []
    blocks_y = []
    cur_block_members: List[List[int]] = []
    cur_block_X: List[np.ndarray] = []
    cur_block_y: List[int] = []
    total_new = 0
    deg_target_cache = {}
    def _flush_block():
        nonlocal cur_block_members, cur_block_X, cur_block_y, blocks_H, blocks_X, blocks_y
        if not cur_block_y:
            return
        n_new = len(cur_block_y)
        nnz_est = sum(len(u) for u in cur_block_members)
        indptr = np.zeros(n_new + 1, dtype=np.int64)
        indices = np.empty(nnz_est, dtype=np.int64)
        pos = 0
        for i, us in enumerate(cur_block_members):
            k = len(us)
            if k:
                indices[pos:pos+k] = np.asarray(us, dtype=np.int64)
                pos += k
            indptr[i+1] = pos
        data = np.ones(pos, dtype=np.float32)
        H_rows = sp.csr_matrix((data, indices[:pos], indptr), shape=(n_new, M), dtype=np.float32)
        blocks_H.append(H_rows)
        blocks_X.append(np.asarray(cur_block_X, dtype=np.float32))
        blocks_y.append(np.asarray(cur_block_y, dtype=np.int64))
        print(f"[SYN][flush] new_rows={n_new}  nnz={pos:,}  blocks={len(blocks_H)}")
        cur_block_members = []
        cur_block_X = []
        cur_block_y = []
    for c in tails:
        idx_c = V_tr[y[V_tr] == c]
        cur = idx_c.size
        need = max(0, target - cur)
        print(f"[SYN][class {c}] train={cur}, need={need}")
        if need <= 0 or cur == 0:
            continue
        mu = deg_target_cache.get(c)
        if mu is None:
            mu = class_deg_target(c, y, V_tr, offsets_i, max_cap=max_edges_per_syn)
            deg_target_cache[c] = mu
        hot_edges = topM_class_edges.get(c, np.empty(0, dtype=np.int64))
        Xc = torch.from_numpy(_ensure_float32(X[idx_c])).to(device)
        Zc = F.normalize(model.encoder_node(Xc), dim=1, eps=1e-8)
        S = model.S
        made = 0
        for t in range(need):
            p = int(rng.integers(0, idx_c.size))
            q = int(rng.integers(0, max(idx_c.size-1, 1)))
            if idx_c.size > 1 and q >= p:
                q += 1
            z_p = Zc[p]
            z_q = Zc[q]
            delta = float(rng.random())
            z_syn = (1.0 - delta) * z_p + delta * z_q
            e_p = edges_of(int(idx_c[p]), offsets_i, indices_users)
            e_q = edges_of(int(idx_c[q]), offsets_i, indices_users)
            cand_small = np.union1d(e_p, e_q)
            cand_e = union_limited(cand_small, hot_edges, cap=cand_cap_per_syn)
            if cand_e.size == 0:
                cand_e = np.arange(M, dtype=np.int64)
                if cand_e.size > cand_cap_per_syn:
                    cand_e = cand_e[:cand_cap_per_syn]
            sum_p = 0.0
            best_prob = -1.0
            best_edge = -1
            for beg in range(0, cand_e.size, ze_score_chunk):
                sl = slice(beg, min(beg + ze_score_chunk, cand_e.size))
                Ze_np = Ze_full[cand_e[sl]]
                Ze_t = torch.from_numpy(Ze_np.astype(np.float32, copy=False)).to(device)
                probs = torch.sigmoid(z_syn @ S @ Ze_t.T).detach().cpu().numpy().astype(np.float32)
                sum_p += float(probs.sum())
                j_local = int(np.argmax(probs))
                if float(probs[j_local]) > best_prob:
                    best_prob = float(probs[j_local])
                    best_edge = int(cand_e[beg + j_local])
            scale = 1.0 if sum_p <= 1e-9 else min(1.0, float(mu) / float(sum_p))
            chosen: List[int] = []
            if scale > 0.0:
                remained = max_edges_per_syn
                for beg in range(0, cand_e.size, ze_score_chunk):
                    if remained <= 0:
                        break
                    sl = slice(beg, min(beg + ze_score_chunk, cand_e.size))
                    Ze_np = Ze_full[cand_e[sl]]
                    Ze_t = torch.from_numpy(Ze_np.astype(np.float32, copy=False)).to(device)
                    probs = torch.sigmoid(z_syn @ S @ Ze_t.T).detach().cpu().numpy().astype(np.float32)
                    probs *= scale
                    draws = (rng.random(probs.shape).astype(np.float32) < probs)
                    if not np.any(draws):
                        continue
                    idx_loc = np.where(draws)[0]
                    if idx_loc.size > remained:
                        part = np.argsort(probs[idx_loc])[-remained:]
                        idx_loc = idx_loc[part]
                    chosen.extend(cand_e[beg + idx_loc].tolist())
                    remained -= len(idx_loc)
            if not chosen:
                chosen = [best_edge] if best_edge >= 0 else []
            x_syn = model.decoder_node(z_syn[None, :]).squeeze(0).detach().cpu().numpy().astype(np.float32)
            cur_block_members.append(chosen)
            cur_block_X.append(x_syn)
            cur_block_y.append(int(c))
            made += 1
            total_new += 1
            need_print = ( (t+1) % max(1, need//5) == 0 ) or ((t+1) == need)
            if need_print:
                avg_deg = int(sum(map(len,cur_block_members))/max(1,len(cur_block_members)))
                print(f"  [SYN][class {c}] {t+1}/{need}  chosen_avg≈{avg_deg}  total_new={total_new}")
            if len(cur_block_y) >= block_new_nodes:
                _flush_block()
        print(f"[SYN][class {c}] made={made}")
    _flush_block()
    if total_new == 0:
        print("[SYN] nothing synthesized; return originals.")
        return X, y, H_items_users, V_tr
    print(f"[SYN] assembling augmented graph ...  blocks={len(blocks_H)}  total_new={total_new}")
    H_new_rows = sp.vstack(blocks_H, format="csr")
    X_new = np.vstack(blocks_X).astype(np.float32, copy=False)
    y_new = np.concatenate(blocks_y).astype(np.int64, copy=False)
    H_aug = sp.vstack([H_items_users, H_new_rows], format="csr")
    H_aug.sum_duplicates()
    if H_aug.nnz:
        H_aug.data[:] = 1.0
    H_aug.sort_indices()
    X_aug = np.vstack([X, X_new]).astype(np.float32, copy=False)
    y_aug = np.concatenate([y, y_new]).astype(np.int64, copy=False)
    N_old = X.shape[0]
    V_tr_aug = np.concatenate([V_tr, np.arange(N_old, N_old + total_new, dtype=np.int64)])
    print(f"[SYN] done: +{total_new} nodes  X={X_aug.shape}  y={y_aug.shape}  H(nnz)={H_aug.nnz:,}")
    return X_aug, y_aug, H_aug, V_tr_aug


def finetune_with_P_fixed(
    model: HSMOTE,
    P: sp.csr_matrix,
    X: np.ndarray,
    y: np.ndarray,
    V_tr_aug: np.ndarray,
    V_val: np.ndarray,
    device: torch.device
) -> LinearHead:
    for p in model.parameters():
        p.requires_grad = False
    for p in model.encoder_node.parameters():
        p.requires_grad = True
    model.encoder_node.to(device).train()
    C = int(np.max(y)) + 1
    with torch.no_grad():
        d_embed = int(model.encoder_node(torch.from_numpy(X[:1]).to(device)).shape[1])
    head = LinearHead(d_embed=d_embed, C=C).to(device)
    opt = optim.Adam([
        {"params": model.encoder_node.parameters(), "lr": cfg.lr_enc, "weight_decay": cfg.wd_enc},
        {"params": head.parameters(),               "lr": cfg.lr_head, "weight_decay": cfg.wd_head},
    ])
    ce = nn.CrossEntropyLoss()
    S = V_tr_aug.shape[0]
    bs = int(getattr(cfg, "bs_seeds", 4096))
    E = int(getattr(cfg, "ft_epochs", 10))
    patience = int(getattr(cfg, "precls_patience", 3))
    no_improve, best_val = 0, float("inf")
    best_state = None
    rng = np.random.default_rng(cfg.seed+123)
    for ep in range(1, E+1):
        model.encoder_node.train(); head.train()
        perm = rng.permutation(S)
        total_loss, nb = 0.0, 0
        for beg in range(0, S, bs):
            sel_rows = V_tr_aug[perm[beg: beg+bs]]
            P_sub = P[sel_rows, :]
            U_cols = np.unique(P_sub.indices)
            loc = np.searchsorted(U_cols, P_sub.indices)
            rows_rep = np.repeat(np.arange(P_sub.shape[0]), np.diff(P_sub.indptr))
            ind = torch.tensor(np.vstack([rows_rep, loc]), dtype=torch.long, device=device)
            val = torch.tensor(P_sub.data, dtype=torch.float32, device=device)
            P_batch = torch.sparse_coo_tensor(ind, val, size=(P_sub.shape[0], U_cols.size)).coalesce()
            X_sub = torch.from_numpy(X[U_cols]).to(device)
            Z_sub = model.encoder_node(X_sub)
            Z_merge = torch.sparse.mm(P_batch, Z_sub)
            logits = head(Z_merge)
            yb = torch.from_numpy(y[sel_rows]).to(device, dtype=torch.long)
            loss = ce(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss); nb += 1
        model.encoder_node.eval(); head.eval()
        with torch.no_grad():
            Z_all = []
            enc_bs = int(getattr(cfg, "encode_bs", 4096))
            for beg in range(0, X.shape[0], enc_bs):
                Z_all.append(model.encoder_node(torch.from_numpy(X[beg: beg+enc_bs]).to(device)).cpu().numpy())
            Z_all = np.vstack(Z_all)
            P_val = P[V_val, :]
            Z_merge_val = P_val.dot(Z_all)
            logits_val = head(torch.from_numpy(Z_merge_val).to(device)).cpu().numpy()
            val_ce = cross_entropy_numpy(logits_val, y[V_val])
            pred_va = logits_val.argmax(1)
            acc_va  = accuracy(y[V_val], pred_va)
            f1_va   = macro_f1(y[V_val], pred_va, C)
            pro_va  = softmax_logits(logits_val)
            auc_va  = auc_ovr_macro(y[V_val], pro_va, C)
        print(f"[FT][{ep}/{E}] trainCE={total_loss/max(1,nb):.4f} | "
              f"valCE={val_ce:.4f} acc={acc_va:.3f} f1={f1_va:.3f} auc={auc_va:.3f}")
        if val_ce + 1e-6 < best_val:
            best_val = val_ce
            no_improve = 0
            best_state = {
                "encoder_node": {k: v.detach().cpu() for k,v in model.encoder_node.state_dict().items()},
                "head":         {k: v.detach().cpu() for k,v in head.state_dict().items()},
                "meta": {"d_embed": d_embed, "C": C}
            }
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[FT] early stop at ep={ep} (best CE={best_val:.4f})")
                break
    if best_state is not None:
        model.encoder_node.load_state_dict(best_state["encoder_node"], strict=False)
        head.load_state_dict(best_state["head"], strict=False)
    return head


def main():
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(f"[INFO] device={device}")
    d = Path(cfg.data_dir)
    p_asins = d / cfg.asins_pkl
    p_feat  = d / cfg.feat_pkl
    p_lab   = d / cfg.lab_pkl
    p_json  = d / cfg.user_json
    idx2node = load_idx2node_from_asins_pkl(p_asins)

    X, y, kept_asins, asin2row = load_and_align(p_feat, p_lab, idx2node)
    N, d_in = X.shape; C = int(np.max(y)) + 1
    print(f"[INFO] |V|={N:,} d={d_in} C={C}")

    kept_map = {a: i for i,a in enumerate(kept_asins)}
    user_ids, offsets_u, indices_items, offsets_i, indices_users = load_hypergraph(kept_map, fallback_user_json=p_json)
    H = _H_from_offsets(N, offsets_i, indices_users, len(user_ids)).tocsr()
    print(f"[INFO] |E|={len(user_ids):,}  nnz={H.nnz:,}")
    Xe_full = compute_full_edge_features(H, X).astype(np.float32, copy=False)
    print(f"[EDGE] Xe_full ready: {Xe_full.shape}")
    V_tr, V_va, V_te = stratified_split_ratio(y, seed=cfg.seed)

    tails = pick_tail_classes(y[V_tr], threshold=cfg.tail_threshold)
    print(f"[SPLIT] train={len(V_tr):,} val={len(V_va):,} test={len(V_te):,}")
    print(f"[TAIL] count={len(tails)}  ids={tails}")

    model = HSMOTE(d_in=cfg.d_in, d_hid=cfg.d_hid, d_embed=cfg.d_embed)
    model = train_hsmote_on_tails(
        model, X, y, V_tr, tails, Xe_full,
        offsets_i, indices_users, offsets_u, indices_items,
        device
    )

    ze_path = Path("cache/Ze_full_fp16.mmap")
    Ze_mm, d_embed = compute_ze_full_memmap(
        H_items_users=H, X_items=X, model=model, device=device,
        out_path=ze_path, chunk_edges=200_000, dtype="float16"
    )
    X_aug, y_aug, H_aug, V_tr_aug = synthesize_vertices_bernoulli(
        model=model,
        X=X, y=y, H_items_users=H,
        V_tr=V_tr, tails=tails,
        device=device,
        target_tail_ratio=0.5,
        ze_memmap_path=ze_path,
        topM=2000,
        cand_cap_per_syn=4000,
        max_edges_per_syn=500,
        ze_score_chunk=20_000,
        block_new_nodes=2000
    )

    print("[PPR] building clique expansion A (binary)...")
    offsets_i = H_aug.indptr.astype(np.int64, copy=False)
    indices_users = H_aug.indices.astype(np.int64, copy=False)
    H_col = H_aug.T.tocsr()
    offsets_u = H_col.indptr.astype(np.int64, copy=False)
    indices_items = H_col.indices.astype(np.int64, copy=False)
    P_hat = build_P_items_topk_bipartite(
        offsets_i, indices_users, offsets_u, indices_items,
        alpha=cfg.ppr_alpha, eps=cfg.ppr_eps, topk=cfg.ppr_topk,
        chunk_seeds=getattr(cfg, "ppr_chunk", 50_000),
        use_int32=True,
    )
    print(f"[PPR] P_hat ready: shape={P_hat.shape}, nnz={P_hat.nnz:,}, row≤{cfg.ppr_topk}")
    head = finetune_with_P_fixed(
        model, P_hat, X_aug, y_aug, V_tr_aug, V_va, device
    )

    with torch.no_grad():
        Z_all = encode_all_in_batches(model.encoder_node, X_aug, device, bs=cfg.encode_bs)
        Z_merge_full = P_hat.dot(Z_all)
        Z_after = diffuse_3steps(H_aug, Z_merge_full, alpha=cfg.infer_alpha, steps=cfg.infer_steps)
        logits_te = head(torch.from_numpy(Z_after[V_te]).to(device)).cpu().numpy()
    pred = logits_te.argmax(1)
    acc = accuracy(y[V_te], pred)
    f1  = macro_f1(y[V_te], pred, int(np.max(y))+1)
    prob= softmax_logits(logits_te)
    auc = auc_ovr_macro(y[V_te], prob, int(np.max(y))+1)
    print(f"[TEST]  Acc={acc:.4f}  Macro-F1={f1:.4f}  AUC-OVR={auc:.4f}")


if __name__ == "__main__":
    main()
