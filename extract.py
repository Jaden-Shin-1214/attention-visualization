"""Offline extraction for the Attention Visualization web tool — DINOv3-L.

Runs vit_large_patch16_dinov3 on one image, captures per-layer per-head
post-RoPE Q / K / V and the attention matrices (RoPE applied, so correct), plus
each block's output for the per-layer PCA-RGB "Layer Visualization". Writes:

  docs/data/<name>/meta.json     coords + metadata (nested lists)
  docs/data/<name>/attn.bin      uint8 attention, shape [L, H, N, N], w = byte/255
  docs/data/<name>/image.jpg     de-normalized 256x256 input
  docs/data/<name>/layers/L##.png  per-layer PCA-RGB of MLP/block-output patches

Attention view: per-(layer,head) PCA (64d -> 3d) over that layer's post-RoPE q,k,v.
DINOv3-L: 24 layers, 16 heads, 261 tokens (5 prefix [CLS+4 reg] + 256 patches), grid 16.
"""
import os, json, argparse
import numpy as np
import torch, timm
from PIL import Image
from timm.layers import apply_rot_embed_cat
from timm.data import resolve_data_config, create_transform

MODEL = "vit_large_patch16_dinov3"


def pca_fit(M, k=3):
    mu = M.mean(0); U, S, Vt = np.linalg.svd(M - mu, full_matrices=False)
    comps = Vt[:k].copy()
    for i in range(k):
        if comps[i][np.argmax(np.abs(comps[i]))] < 0:
            comps[i] *= -1
    return mu, comps


def pca_tr(M, mu, comps):
    return (M - mu) @ comps.T


def pca_rgb(F):
    """patch features (P,C) -> per-image PCA top-3 -> robust-normalized RGB (P,3)."""
    mu, comps = pca_fit(F.astype(np.float64), 3)
    proj = pca_tr(F.astype(np.float64), mu, comps)
    lo = np.percentile(proj, 2, axis=0); hi = np.percentile(proj, 98, axis=0)
    rgb = np.clip((proj - lo) / (hi - lo + 1e-9), 0, 1)
    return (rgb * 255).round().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "docs", "data"))
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = timm.create_model(MODEL, pretrained=True).to(dev).eval()
    NB, NH, HD = len(m.blocks), m.blocks[0].attn.num_heads, 64
    NPT = m.blocks[0].attn.num_prefix_tokens
    cfg = resolve_data_config({}, model=m); tf = create_transform(**cfg)
    img = Image.open(args.image).convert("RGB")
    x = tf(img).unsqueeze(0).to(dev)

    cap = {}; blkout = {}

    def wrap(i):
        a = m.blocks[i].attn; a.fused_attn = False
        def fwd(t, rope=None, attn_mask=None, is_causal=False):
            B, N, C = t.shape
            qkv = a.qkv(t).reshape(B, N, 3, a.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0); q, k = a.q_norm(q), a.k_norm(k)
            if rope is not None:
                npt = a.num_prefix_tokens; half = getattr(a, "rotate_half", False)
                q = torch.cat([q[:, :, :npt], apply_rot_embed_cat(q[:, :, npt:], rope, half=half)], 2).type_as(v)
                k = torch.cat([k[:, :, :npt], apply_rot_embed_cat(k[:, :, npt:], rope, half=half)], 2).type_as(v)
            attn = ((q * a.scale) @ k.transpose(-2, -1)).softmax(-1)
            cap[i] = (q[0].cpu().numpy(), k[0].cpu().numpy(), v[0].cpu().numpy(), attn[0].cpu().numpy())
            out = (attn @ v).transpose(1, 2).reshape(B, N, C)
            return a.proj_drop(a.proj(a.norm(out)))
        a.forward = fwd
        m.blocks[i].register_forward_hook(
            lambda mod, inp, out, i=i: blkout.__setitem__(i, out.detach()[0].cpu().numpy()))

    for i in range(NB):
        wrap(i)
    with torch.no_grad():
        m.forward_features(x)

    N = cap[0][3].shape[-1]               # 261
    grid = int(round((N - NPT) ** 0.5))   # 16

    # ---- attention matrices -> uint8 ----
    attn = np.stack([cap[i][3] for i in range(NB)], 0).astype(np.float32)  # (NB,NH,N,N)

    # ---- attention-view PCA: per (layer, head) over post-rope q,k,v ----
    attn_view = {t: [[None] * NB for _ in range(NH)] for t in ("q", "k", "v")}
    for h in range(NH):
        for i in range(NB):
            q, k, v = cap[i][0][h], cap[i][1][h], cap[i][2][h]            # (N,64) each
            mu, comps = pca_fit(np.concatenate([q, k, v], 0).astype(np.float64), 3)
            for name, arr in (("q", q), ("k", k), ("v", v)):
                attn_view[name][h][i] = np.round(pca_tr(arr.astype(np.float64), mu, comps), 3).tolist()

    # ---- write ----
    outdir = os.path.join(args.out, args.name); os.makedirs(outdir, exist_ok=True)
    layerdir = os.path.join(outdir, "layers"); os.makedirs(layerdir, exist_ok=True)
    (attn * 255).round().clip(0, 255).astype(np.uint8).tofile(os.path.join(outdir, "attn.bin"))

    # per-layer PCA-RGB of block-output patch tokens
    for i in range(NB):
        patches = blkout[i][NPT:]                       # (256,1024)
        rgb = pca_rgb(patches).reshape(grid, grid, 3)
        Image.fromarray(rgb, "RGB").save(os.path.join(layerdir, f"L{i:02d}.png"))

    mean = torch.tensor(cfg["mean"]).view(3, 1, 1); std = torch.tensor(cfg["std"]).view(3, 1, 1)
    disp = (x[0].cpu() * std + mean).clamp(0, 1)
    Image.fromarray((disp.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(
        os.path.join(outdir, "image.jpg"), quality=92)

    meta = {
        "model": MODEL, "image": os.path.basename(args.image),
        "n_layers": NB, "n_heads": NH, "n_tokens": N, "patch_grid": grid,
        "n_prefix": NPT, "has_cls": True,
        "attn": {"file": "attn.bin", "dtype": "uint8", "scale": 255, "shape": [NB, NH, N, N]},
        "attn_view": attn_view,
    }
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f)
    mb = os.path.getsize(os.path.join(outdir, "meta.json")) / 1e6
    ab = os.path.getsize(os.path.join(outdir, "attn.bin")) / 1e6
    print(f"wrote {outdir}: meta {mb:.1f}MB  attn {ab:.1f}MB  layers {NB} pngs  (N={N}, grid={grid})")


if __name__ == "__main__":
    main()
