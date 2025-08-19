> This repository contains the code and data for the paper “Hypergraph-Based High-Order Correlation Analysis for Large-Scale Long-Tailed Data.”

## Highlights

- **HSMOTE**: Lightweight dual encoder–decoder (`models.HSMOTE`) trained with BCE reconstruction and negative sampling on node–hyperedge incidence.
- **Tail oversampling via synthesis**: Interpolation in embedding space + **Bernoulli edge membership sampling** over class-aware candidate sets (`synthesize_vertices_bernoulli`), with **Top-M per-class popular edges**, candidate capping, and class-specific degree targets.
- **Global structure aggregation**: Build **row-normalized Top-k PPR** on the item–user bipartite graph, keep item→item mass, and **fine-tune `encoder_node + LinearHead` with fixed P**.
- **Inference-time diffusion**: 3-step lightweight diffusion (`utils.diffuse_3steps`) to smooth representations before classification.

## Repository Structure

```
repo/
├── config.py                 # Hyperparameters & paths (edit to your data locations)
├── main.py                   # End-to-end: load → hypergraph → HSMOTE → synth → PPR → finetune → eval
├── models.py                 # HSMOTE and LinearHead
├── load_data.py              # Loading/alignment, subgraph builder, BCE w/ negative sampling, helpers
├── utils.py                  # Training utils, metrics, PPR builders, diffusion, memmap, etc.
├── ppr.py                    # PPR
├── data/                     # raw features/labels/user JSON
├── cache/                    # runtime caches
└── README.md
```

## Data & Formats

`config.py` points to your files (update paths as needed):

```python
# Example fields — adapt to your setup
data_dir   = "Datasets/subset"
asins_pkl  = "asins_**.pkl"       
feat_pkl   = "features_**.pkl"    
lab_pkl    = "labels_**.pkl"      
user_json  = "user_products_1000.json"
```

**Expected formats**

- `features_**.pkl`: `{'features': float32 [N, d], 'asins': List[str]}`

- `labels_**.pkl`:   `{'labels': int64 [N]}`

- `asins_**.pkl`: several schemas are supported; `load_idx2node_from_asins_pkl` auto-parses and aligns

- `user_products_1000.json`: user–item interactions, e.g.:

  ```json
  [
    {"user": "u1", "items": [{"prefix":"P","asin":"A1"}, {"prefix":"P","asin":"A2"}]},
    {"user": "u2", "items": [{"asin":"B3"}]}
  ]
  ```

**Hypergraph construction & cache**

- Datasets download: https://drive.google.com/file/d/1v8nXKoIrd7bmfGZyW6N3_WT0dZUASDrw/view?usp=sharing

- `load_hypergraph` builds CSR for item→user (`offsets_i`, `indices_users`) and user→item (`offsets_u`, `indices_items`).
- A stable cache key (from `kept_asin2row`) is used to write `.csr-<hash>.npz`, `.users.json`, and `.meta.json` for fast reloads.

## Quick Start (End-to-End)

Run:

```bash
python main.py
```

Pipeline in `main.py`:

1. **Load & align**: `load_and_align` aligns features/labels to `idx2node` → `X, y`.
2. **Build hypergraph**: `load_hypergraph` → CSR `H`; compute full edge features `Xe_full`.
3. **Split**: `stratified_split_ratio` per class.
4. **Train HSMOTE**: `train_hsmote_on_tails` on tail classes via batch subgraphs (BCE + MSE).
5. **Synthesize**: `compute_ze_full_memmap` encodes edge embeddings `Ze` (memmap). `synthesize_vertices_bernoulli` creates new tail nodes + merges dataset.
6. **Build PPR**: `build_P_items_topk_bipartite` on the bipartite graph, keep item→item rows, re-normalize → `P_hat`.
7. **Fine-tune & evaluate**: `finetune_with_P_fixed` (fixed `P_hat`) on train/val; inference uses 3-step diffusion.

Console logs include `[HSMOTE]`, `[PPR]`, `[FT]` (fine-tuning), and final `[TEST]` metrics.

## Configuration (Key Hyperparameters)

Adjust in `config.py` :

- **Model**: `d_in`, `d_hid`, `d_embed`
- **HSMOTE pretraining**: `lr_hsmote`, `wd_hsmote`, `beta_mse`, `neg_per_pos`, `hsmote_batches`, `per_class_train`, `log_every`
- **Tail threshold**: `tail_threshold` (relative to the largest class)
- **Synthesis**: `target_tail_ratio`, `topM`, `cand_cap_per_syn`, `max_edges_per_syn`, `ze_score_chunk`, `block_new_nodes`
- **PPR**: `ppr_alpha`, `ppr_eps`, `ppr_topk`, `ppr_chunk`
- **Fine-tuning**: `lr_enc`, `wd_enc`, `lr_head`, `wd_head`, `ft_epochs`, `precls_patience`, `bs_seeds`
- **Inference**: `infer_alpha`, `infer_steps`, `encode_bs`

------

## Contact

- Maintainer: **Xiangmin Han / Tsinghua University**
- Email: **simon.xmhan@gmail.com_**
