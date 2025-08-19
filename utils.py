import random
from load_data import *
from models import *
from scipy.sparse import save_npz, load_npz

import ppr as PPR


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(arg: str = "auto"):
    if arg == "cpu":
        return torch.device("cpu")
    if arg == "cuda":
        return torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    return torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean()) if y_true.size else 0.0


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, C: int) -> float:
    f1s = []
    for c in range(C):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0


def softmax_logits(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def auc_ovr_macro(y_true: np.ndarray, prob: np.ndarray, C: int) -> float:
    """Lightweight AUC (one-vs-rest, rank-based approximation) without sklearn."""
    def auc01(pos, neg):
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        s = np.concatenate([pos, neg])
        r = s.argsort().argsort().astype(np.float64) + 1
        rp = r[:len(pos)]
        return (rp.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

    aucs = []
    for c in range(C):
        pos = (y_true == c)
        aucs.append(auc01(prob[pos, c], prob[~pos, c]))
    return float(np.mean(aucs)) if aucs else 0.5


def per_class_report(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     C: int,
                     class_names=None,
                     sort_by: str = "support",
                     descending: bool = True) -> dict:
    """
    Compute per-class precision/recall/F1/support and print a table.
    Returns a dict with arrays for further analysis/plotting.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    assert y_true.shape[0] == y_pred.shape[0]
    assert np.all((y_true >= 0) & (y_true < C)), "y_true 超出类别范围"
    assert np.all((y_pred >= 0) & (y_pred < C)), "y_pred 超出类别范围"

    support = np.bincount(y_true, minlength=C).astype(np.int64)

    tp = np.zeros(C, dtype=np.int64)
    fp = np.zeros(C, dtype=np.int64)
    fn = np.zeros(C, dtype=np.int64)
    for c in range(C):
        mask_t = (y_true == c)
        mask_p = (y_pred == c)
        tp[c] = np.sum(mask_t & mask_p)
        fp[c] = np.sum(~mask_t & mask_p)
        fn[c] = np.sum(mask_t & ~mask_p)

    eps = 1e-12
    precision = tp / np.maximum(tp + fp, 1)
    recall    = tp / np.maximum(tp + fn, 1)
    f1 = np.where((precision + recall) > eps,
                  2 * precision * recall / (precision + recall + eps), 0.0)

    macro_p = precision.mean() if C > 0 else 0.0
    macro_r = recall.mean() if C > 0 else 0.0
    macro_f = f1.mean() if C > 0 else 0.0
    weighted_f = (f1 * (support / np.maximum(support.sum(), 1))).sum() if support.sum() > 0 else 0.0
    acc = float((y_true == y_pred).mean())

    key_map = {
        "support": support,
        "f1": f1,
        "recall": recall,
        "precision": precision,
        "id": np.arange(C),
    }
    key = key_map.get(sort_by, support)
    order = np.argsort(key)
    if descending:
        order = order[::-1]

    name_col = []
    if class_names is None:
        name_col = [f"{i:>3d}" for i in range(C)]
    else:
        assert len(class_names) == C
        name_col = [str(n) for n in class_names]

    header = f"{'cls':>4}  {'name':<16}  {'supp':>7}  {'prec':>6}  {'rec':>6}  {'f1':>6}"
    print(header)
    print("-" * len(header))
    for i in order:
        print(f"{i:>4}  {name_col[i]:<16}  {support[i]:>7d}  {precision[i]:>6.3f}  {recall[i]:>6.3f}  {f1[i]:>6.3f}")

    print("-" * len(header))
    print(f"{'ALL':>4}  {'(macro/weighted)':<16}  {int(support.sum()):>7d}  "
          f"{macro_p:>6.3f}  {macro_r:>6.3f}  {macro_f:>6.3f}  |  "
          f"Acc={acc:.3f}  W-F1={weighted_f:.3f}")

    return {
        "support": support,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f,
        "weighted_f1": weighted_f,
        "accuracy": acc,
        "order": order,
    }


def cross_entropy_numpy(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Simple CE in numpy: softmax logits then take -log p[y]."""
    x = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(x)
    p = ex / ex.sum(axis=1, keepdims=True)
    idx = (np.arange(y_true.size), y_true)
    eps = 1e-12
    return float(-np.log(p[idx] + eps).mean())


def diffuse_3steps(H: csr_matrix, Z: np.ndarray, alpha=0.1, steps=3) -> np.ndarray:
    """
    Inference-time 3-step diffusion:
      Z^{l+1} = (1-α) D_v^{-1/2} H D_e^{-1} H^T D_v^{-1/2} Z^{l} + α Z
    """
    H = H.tocsr()
    Dv = np.asarray(H.sum(axis=1)).ravel().astype(np.float32)
    De = np.asarray(H.sum(axis=0)).ravel().astype(np.float32)
    inv_sqrt_Dv = 1.0 / np.maximum(np.sqrt(Dv), 1.0)
    inv_De = 1.0 / np.maximum(De, 1.0)

    Zm = Z.copy()
    for _ in range(steps):
        A1 = (Zm * inv_sqrt_Dv[:, None])
        A2 = H.T @ A1
        A2 = A2 * inv_De[:, None]
        A3 = H @ A2
        A3 = A3 * inv_sqrt_Dv[:, None]
        Zm = (1 - alpha) * A3 + alpha * Z
    return Zm


def encode_all_in_batches(encoder_node: nn.Module,
                          X: np.ndarray,
                          device: torch.device,
                          bs: int = 4096) -> np.ndarray:
    encoder_node.eval()
    out = []
    with torch.no_grad():
        for i in (range(0, X.shape[0], bs)):
            xb = torch.from_numpy(X[i:i + bs]).to(device)
            zb = encoder_node(xb).cpu().numpy().astype(np.float32, copy=False)
            out.append(zb)
    return np.vstack(out)


def open_ze_memmap(ze_path: Path, M_edges: int, fallback_dtype: str = "float16"):
    """
    Safely open Ze memmap:
      1) Prefer reading .meta.json for d_embed/dtype
      2) Otherwise infer d_embed from file size.
    Returns np.memmap with shape (M_edges, d_embed).
    """
    import os, json
    ze_path = Path(ze_path)
    meta_path = ze_path.with_suffix(ze_path.suffix + ".meta.json")

    dtype = fallback_dtype
    d_embed = None

    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if int(meta.get("rows", -1)) not in (-1, M_edges):
                print(f"[Ze][WARN] meta rows={meta.get('rows')} != M_edges={M_edges}, 仍按 M_edges 打开")
            d_embed = int(meta.get("d_embed"))
            dtype = str(meta.get("dtype", fallback_dtype))
        except Exception as e:
            print(f"[Ze][WARN] 读取 meta 失败：{e}，改为按文件大小推断")

    if d_embed is None:
        file_size = os.path.getsize(ze_path)
        itemsize = np.dtype(dtype).itemsize
        if file_size % (M_edges * itemsize) != 0:
            raise ValueError(
                f"[Ze] 文件大小 {file_size} 与 (rows={M_edges}, dtype={dtype}) 不整除，无法推断 d_embed。"
            )
        d_embed = file_size // (M_edges * itemsize)

    Ze_full = np.memmap(ze_path, mode="r", dtype=dtype, shape=(M_edges, d_embed))
    return Ze_full


def load_augmented_dataset(out_dir: str, tag: str):
    out = Path(out_dir)
    X  = np.load(out / f"X_{tag}.npy", mmap_mode=None)
    y  = np.load(out / f"y_{tag}.npy", mmap_mode=None)
    H  = load_npz(out / f"H_{tag}.npz")
    Vt = np.load(out / f"train_idx_{tag}.npy", mmap_mode=None)
    return X, y, H, Vt


def build_bipartite_adj_from_offsets(
    N: int, M: int,
    offsets_i: np.ndarray, indices_users: np.ndarray,
    offsets_u: np.ndarray, indices_items: np.ndarray,
    use_int32: bool = True,
) -> sp.csr_matrix:
    """
    Build bipartite adjacency B ((N+M)×(N+M)).
    Rows 0..N-1 are items; rows N..N+M-1 are users.
    Use exactly O(nnz(H)) undirected edges (item <-> user).
    """
    itype = np.int32 if use_int32 else np.int64
    N = int(N); M = int(M)
    nnz = int(offsets_i[-1])

    indptr = np.zeros(N + M + 1, dtype=itype)
    indptr[1:N+1] = offsets_i[1:].astype(itype, copy=False)
    deg_u = (offsets_u[1:] - offsets_u[:-1]).astype(itype, copy=False)
    indptr[N+1:] = (indptr[N] + np.cumsum(deg_u, dtype=itype))

    indices = np.empty(2 * nnz, dtype=itype)
    data    = np.ones(2 * nnz, dtype=np.float32)

    pos = 0
    for i in range(N):
        s, e = int(offsets_i[i]), int(offsets_i[i+1])
        k = e - s
        if k:
            indices[pos:pos+k] = indices_users[s:e] + N
            pos += k
    for u in range(M):
        s, e = int(offsets_u[u]), int(offsets_u[u+1])
        k = e - s
        if k:
            indices[pos:pos+k] = indices_items[s:e]
            pos += k

    B = sp.csr_matrix((data, indices, indptr), shape=(N+M, N+M), dtype=np.float32)
    B.sort_indices()
    return B


def build_P_items_topk_bipartite(
    offsets_i: np.ndarray, indices_users: np.ndarray,
    offsets_u: np.ndarray, indices_items: np.ndarray,
    alpha: float, eps: float, topk: int,
    chunk_seeds: int = 50_000,
    use_int32: bool = True,
) -> sp.csr_matrix:
    """
    Run top-k PPR on the bipartite graph for all item rows.
    Keep only item→item columns, then re-normalize each row.
    Returns P (N×N) with ≤ topk nonzeros per row; empty rows get a self-loop.
    """
    N = int(len(offsets_i) - 1)
    M = int(len(offsets_u) - 1)

    B = build_bipartite_adj_from_offsets(
        N, M, offsets_i, indices_users, offsets_u, indices_items, use_int32=use_int32
    )

    rows = []
    for beg in range(0, N, chunk_seeds):
        end = min(beg + chunk_seeds, N)
        idx = np.arange(beg, end, dtype=np.int64)

        P_blk = PPR.topk_ppr_matrix(B, alpha=alpha, eps=eps, idx=idx, topk=topk).tocsr()

        P_blk = P_blk[:, :N].tocsr()
        P_blk.sort_indices()
        indptr, data = P_blk.indptr, P_blk.data
        for i in range(P_blk.shape[0]):
            s, e = indptr[i], indptr[i+1]
            if e > s:
                ssum = float(np.sum(data[s:e]))
                if ssum > 0:
                    data[s:e] /= ssum
            else:
                P_blk[i, beg + i] = 1.0
        P_blk.eliminate_zeros()
        rows.append(P_blk)

        print(f"[PPR][{beg:,}~{end:,}] done: nnz={P_blk.nnz:,}")

    P = sp.vstack(rows, format="csr")
    P.sort_indices()
    return P
