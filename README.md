---
title: Kmeans Clustering
emoji: 🎨
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: "6.6.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

# K-Means Color Extractor

Extract the dominant colors from any image using the K-Means clustering algorithm. Upload a photo, pick how many colors you want, and get back a segmented image, a color palette, and the raw RGB values.

**Live demos:**
- [Hugging Face Space](https://huggingface.co/spaces/prashant-gulati/kmeans_clustering)
- [Google Colab](https://colab.research.google.com/drive/1fyhpMLgGKGxU5WwfvgLgJ0edsWlrUn7q#scrollTo=0a3c9adc)

---

## What it does

1. Reshapes the image into a list of pixels (R, G, B)
2. Runs K-Means to group those pixels into *k* clusters
3. Replaces every pixel with its cluster's centroid color
4. Renders a segmented image, a color swatch palette, and the hex/RGB values

## Demo

| Input | Segmented (k=4) | Palette |
|-------|----------------|---------|
| Upload any image | Pixels replaced by dominant colors | Color swatches + RGB values |

## Quickstart

```bash
# 1. Clone
git clone https://github.com/prashant-gulati/kmeans_clustering
cd kmeans_clustering

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python app.py
```

Then open the local Gradio URL printed in your terminal.

## Usage

1. Upload an image (JPEG, PNG, or RGBA — alpha channel is handled automatically)
2. Choose **k** — the number of dominant colors to extract (2 – 6)
3. Click **Submit**
4. View the segmented image, the palette, and the RGB values

## How K-Means works here

K-Means treats each pixel as a point in 3D color space (R, G, B). It iteratively assigns pixels to the nearest centroid and recomputes centroids until convergence. The final centroids are the dominant colors.

- `n_init=10` — runs 10 times with different seeds, keeps the best result
- `random_state=42` — reproducible output

## Stack

| Library | Role |
|---------|------|
| [Gradio](https://gradio.app) | Interactive web UI |
| [scikit-learn](https://scikit-learn.org) | K-Means implementation |
| [NumPy](https://numpy.org) | Pixel array manipulation |
| [Pillow](https://python-pillow.org) | Image loading |

## Project structure

```
kmeans_clustering/
├── app.py            # Gradio app + K-Means logic
├── requirements.txt
├── artist.jpeg       # Example images
├── cat.jpg
├── quote.jpeg
├── skull.jpg
└── woman.jpeg
```