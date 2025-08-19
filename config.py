from types import SimpleNamespace

cfg = SimpleNamespace(
    # IO
    data_dir="Datasets/Amazon-LT-1000",
    asins_pkl="asins_1000.pkl",
    feat_pkl="features_1000.pkl",
    lab_pkl="labels_1000.pkl",
    user_json="user_products_1000.json",
    hsmote_model="cache/model.pth",

    # device & seed
    device="cuda:0",
    seed=42,

    # model dims
    d_in=512,     # input feature dim
    d_hid=32,
    d_embed=32,

    # split & tails
    tail_threshold=0.5,

    # HSMOTE training (batches)
    hsmote_batches=200,
    per_class_train=50,
    neg_per_pos=5,
    beta_mse=0.5,
    lr_hsmote=1e-3,
    wd_hsmote=1e-5,

    # synthesis
    target_tail_ratio=0.5,
    syn_log_every=5000,

    # PPR (your ppr.py)
    ppr_alpha=0.1,
    ppr_eps=1e-4,
    ppr_topk=64,

    # fine-tune (with fixed P)
    ft_epochs=20,
    bs_seeds=4096,
    lr_enc=1e-3,  wd_enc=1e-5,
    lr_head=1e-2, wd_head=1e-5,
    precls_patience=3,

    # encoding & inference
    encode_bs=4096,
    infer_alpha=0.1,
    infer_steps=3,
)
