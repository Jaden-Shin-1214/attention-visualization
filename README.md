# Attention Visualization

An interactive playground for watching **how a Vision Transformer's Q · K · V spaces take shape** —
layer by layer, head by head — and how a query's attention flows to its keys and on to their values.

A sibling of [Feature Visualization](https://jaden-shin-1214.github.io/feature-visualization/).

**Live:** https://jaden-shin-1214.github.io/attention-visualization/

## What you can do

- Pick a sample image and explore its tokens.
- Move the **Layer / Head / Query** sliders (or click a Q node).
- Watch a query light up its keys (line weight = softmax attention) and each key link to its value,
  in a per-(layer, head) **64-d PCA** space where spatial proximity tracks attention.
- Read the full 197×197 attention map, a per-query patch heatmap, and the query's patch marked
  on the input image.

## Model

`vit_base_patch16_224.augreg2_in21k_ft_in1k` (ViT-B/16, 12 layers · 12 heads · 197 tokens = 1 CLS + 196 patches),
loaded via [timm](https://github.com/huggingface/pytorch-image-models).

## Layout

```
extract.py        offline: image -> ViT -> per-(layer,head) PCA + attention -> docs/data/<name>/
docs/             the static site (served by GitHub Pages)
  index.html  app.js  style.css  vendor/three.js
  serve.py        local dev server with caching disabled
  data/<name>/    meta.json (coords) + attn.bin (uint8 attention) + image.jpg
  data/index.json sample manifest
```

## Add an image

```bash
conda run -n featviz python extract.py --image path/to/img.jpg --name myimg
# then add {"name":"myimg","label":"My image"} to docs/data/index.json
```

## Run locally

```bash
cd docs && python3 serve.py 8021
# open http://localhost:8021
```

Built with [three.js](https://threejs.org). Sample images from the
[ImageNet sample set](https://github.com/EliSchwartz/imagenet-sample-images).
