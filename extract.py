"""Offline extraction for the Attention Visualization web tool.

Runs ViT-B/16 on one image, captures per-layer per-head Q / K / V and the
per-head attention matrices, projects them to 3D, and writes the static data
the browser viewer consumes:

  site/data/<name>/meta.json    coords + metadata (nested lists)
  site/data/<name>/attn.bin     uint8 attention, shape [L, H, N, N], w = byte/255
  site/data/<name>/image.jpg    de-normalized 224x224 input (for patch crops)

Attention view: per-(layer,head) PCA (64d -> 3d) over that layer's q,k,v, so
"near == high attention" stays readable while the layer slider switches frames.
"""
import os, json, argparse
import numpy as np
import torch
import timm
from PIL import Image
from timm.data import resolve_data_config, create_transform

MODEL = "vit_base_patch16_224.augreg2_in21k_ft_in1k"


def pca_fit(M, k=3):
    mu = M.mean(0)
    Mc = M - mu
    # SVD-based PCA; sign-fix so the largest-magnitude loading is positive
    U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    comps = Vt[:k].copy()
    for i in range(k):
        if comps[i][np.argmax(np.abs(comps[i]))] < 0:
            comps[i] *= -1
    return mu, comps


def pca_tr(M, mu, comps):
    return (M - mu) @ comps.T


def spearman_near_attn(posQ, posK, A):
    """mean over q of Spearman(-dist(q,:), attn(q,:)). Pure-numpy rank corr."""
    def rank(x):
        order = x.argsort()
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(len(x))
        return r
    rs = []
    for qi in range(posQ.shape[0]):
        d = np.linalg.norm(posK - posQ[qi], axis=1)
        a, b = rank(-d), rank(A[qi])
        a -= a.mean(); b -= b.mean()
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        if denom > 0:
            rs.append(float((a * b).sum() / denom))
    return float(np.mean(rs)) if rs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--name", required=True, help="dataset slug, e.g. dog")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "docs", "data"))
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = timm.create_model(MODEL, pretrained=True).to(dev).eval()
    cfg = resolve_data_config({}, model=m)
    tf = create_transform(**cfg)
    img = Image.open(args.image).convert("RGB")
    x = tf(img).unsqueeze(0).to(dev)

    NB, NH, HD, D = len(m.blocks), m.blocks[0].attn.num_heads, 64, 768
    N = 197  # 1 cls + 196 patches
    grid = 14

    store = {}

    def mk_hooks(i):
        blk = m.blocks[i]
        blk.attn.qkv.register_forward_hook(
            lambda mod, inp, out: store.setdefault(i, {}).__setitem__("qkv", out.detach()))

    for i in range(NB):
        mk_hooks(i)
    with torch.no_grad():
        m.forward_features(x)

    def split_qkv(qkv):  # (1,N,2304) -> q,k,v each (NH,N,HD)
        t = qkv[0].reshape(N, 3, NH, HD).permute(1, 2, 0, 3)
        return t[0], t[1], t[2]

    # ---- gather per-layer tensors ----
    per = {}
    attn = np.zeros((NB, NH, N, N), dtype=np.float32)
    for i in range(NB):
        s = store[i]
        qh, kh, vh = split_qkv(s["qkv"])                  # (NH,N,HD)
        per[i] = dict(qh=qh.cpu().numpy(), kh=kh.cpu().numpy(), vh=vh.cpu().numpy())
        blk = m.blocks[i]
        for h in range(NH):
            q = blk.attn.q_norm(qh[h]); k = blk.attn.k_norm(kh[h])
            a = torch.softmax((q * (HD ** -0.5)) @ k.T, dim=-1)
            attn[i, h] = a.cpu().numpy()

    # ---- attention-view PCA: per (layer, head) over that layer's q,k,v ----
    # attention is a within-layer operation, so a per-layer frame keeps
    # "near == high attention" readable; cross-layer motion is the trajectory
    # view's job. q,k,v share one frame per (layer,head) so Q-near-K is meaningful.
    attn_view = {"q": [[None] * NB for _ in range(NH)],
                 "k": [[None] * NB for _ in range(NH)],
                 "v": [[None] * NB for _ in range(NH)]}
    sp_report = []
    for h in range(NH):
        for i in range(NB):
            stack = np.concatenate([per[i][t][h] for t in ("qh", "kh", "vh")], 0)
            mu, comps = pca_fit(stack, 3)
            for tname, key in (("q", "qh"), ("k", "kh"), ("v", "vh")):
                attn_view[tname][h][i] = np.round(pca_tr(per[i][key][h], mu, comps), 4).tolist()
            if h == 0 and i in (3, 6, 9):
                pQ = np.array(attn_view["q"][h][i]); pK = np.array(attn_view["k"][h][i])
                sp_report.append((i, h, spearman_near_attn(pQ, pK, attn[i, h])))

    # ---- write outputs ----
    outdir = os.path.join(args.out, args.name)
    os.makedirs(outdir, exist_ok=True)
    (attn * 255).round().clip(0, 255).astype(np.uint8).tofile(os.path.join(outdir, "attn.bin"))

    # de-normalized image for client-side patch crops
    mean = torch.tensor(cfg["mean"]).view(3, 1, 1)
    std = torch.tensor(cfg["std"]).view(3, 1, 1)
    disp = (x[0].cpu() * std + mean).clamp(0, 1)
    Image.fromarray((disp.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(
        os.path.join(outdir, "image.jpg"), quality=92)

    meta = {
        "model": MODEL, "image": os.path.basename(args.image),
        "n_layers": NB, "n_heads": NH, "n_tokens": N, "patch_grid": grid, "has_cls": True,
        "attn": {"file": "attn.bin", "dtype": "uint8", "scale": 255,
                 "shape": [NB, NH, N, N]},
        "attn_view": attn_view,
    }
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f)

    mb = os.path.getsize(os.path.join(outdir, "meta.json")) / 1e6
    ab = os.path.getsize(os.path.join(outdir, "attn.bin")) / 1e6
    print(f"wrote {outdir}")
    print(f"  meta.json {mb:.1f} MB   attn.bin {ab:.1f} MB")
    print("  near==attention Spearman (per-head 64->3d, per-layer fit):")
    for L, h, r in sp_report:
        print(f"    L{L} h{h}: {r:+.3f}")


if __name__ == "__main__":
    main()
