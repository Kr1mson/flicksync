# FlickSync
## Reverse Video Search using TimeSformer & FAISS

FlickSync is a **video similarity search system** that uses the [TimeSformer](https://arxiv.org/abs/2102.05095) transformer model (pretrained on Kinetics-400) to generate deep temporal video embeddings, and [FAISS](https://github.com/facebookresearch/faiss) for fast nearest-neighbor retrieval. Upload any video and FlickSync returns the most visually similar clips from the UCF-101 dataset.

---

## ✨ Features

- **Deep Video Embeddings** — TimeSformer CLS-token pooling captures rich temporal representations across frames.
- **Fast Similarity Search** — FAISS L2-indexed, L2-normalized embeddings enable sub-millisecond retrieval at scale.
- **Modern Web UI** — A two-page frontend (`index.html` + `results.html`) with drag-and-drop upload, GIF previews, similarity bars, and per-result inspection.
- **REST API** — Flask backend exposes `/api/search`, `/api/health`, and `/api/thumbnail` endpoints.
- **GIF Previews** — Animated previews generated server-side for both the query video and all retrieved results.
- **Evaluation Notebook** — `evaluation.ipynb` measures Recall@1/3/5 and renders a t-SNE cluster visualization of the embedding space.

---

## 📂 Project Structure

```
flicksync/
├── app.py                      # Flask API (search, health, thumbnail)
├── Dockerfile                  # Docker container definition
├── docker-compose.yml          # Container orchestration
├── pyproject.tomlW              # Project metadata & dependencies
├── uv.lock                     # Locked dependency versions
├── .dockerignore
├── .gitignore
├── .python-version
├── frontend/
│   ├── index.html              # Upload page
│   └── results.html            # Results page
├── notebooks/
│   └── evaluation.ipynb        # Recall@k evaluation & t-SNE visualization
├── tests/
│   └── smoke_test.py           # Basic sanity checks
└── utility/
    ├── __init__.py
    ├── embedder.py             # Embedding generation utilities
    └── video_processing.py     # Frame loading & TimeSformer extraction
```

---

## 📦 Dataset

The system indexes the [UCF-101](https://www.crcv.ucf.edu/research/data-sets/ucf101/) action recognition dataset — 13,320 videos across 101 action categories.

---

## 🛠️ Getting Started

**Requirements**

- Python 3.8+
- PyTorch (CPU or CUDA)
- [uv](https://github.com/astral-sh/uv) for dependency management
- UCF-101 dataset ([download here](https://www.crcv.ucf.edu/research/data-sets/ucf101/))

**Installation**

```bash
git clone https://github.com/your-org/flicksync.git
cd flicksync
uv sync
```

**Dataset setup**

Download UCF-101 and place it in the project root so the structure matches:
```
flicksync/
└── UCF101/
    ├── train/
    ├── test/
    └── val/
```

**Build the FAISS index** *(one-time setup)*

Run the embedder to generate embeddings for all training videos and save the FAISS index:

```bash
uv run python utility/embedder.py
```

This will create `embeddings/faiss_ucf101.index` and `embeddings/embedding_map.pkl`. Depending on your hardware this takes a while — a GPU is strongly recommended.

**Run the API**

```bash
uv run python app.py
```

The Flask server starts on `http://localhost:5000`.

**Open the frontend**

Open `frontend/index.html` directly in your browser. For local development the API calls are hardcoded to `http://localhost:5000`.

---

## 🔌 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/search` | POST | Upload a video file (multipart) or send a JSON `{"path": "...", "k": 5}` to retrieve the top-k similar videos. |
| `/api/health` | GET | Returns index status, vector count, and compute device. |
| `/api/thumbnail` | GET | Streams an animated GIF preview for any indexed video path (`?path=...`). |

**Search response fields:**

```json
{
  "query_path": "my_video.avi",
  "query_gif_b64": "<base64 animated GIF>",
  "k": 5,
  "device": "cuda",
  "embedding_time_ms": 843.2,
  "search_time_ms": 0.18,
  "results": [
    {
      "rank": 1,
      "id": 412,
      "distance": 0.21,
      "similarity_score": 94.75,
      "path": "UCF101/train/BalanceBeam/v_BalanceBeam_g01_c01.avi"
    }
  ]
}
```

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Model | `facebook/timesformer-base-finetuned-k400` |
| Input frames | 8 uniformly sampled frames per video |
| Frame size | 224 × 224 |
| Embedding dim | 768 (CLS token) |
| FAISS index | `IndexFlatIP` with L2 normalization (cosine similarity) |

Embeddings are L2-normalized before indexing and before search, so FAISS inner-product scores correspond directly to cosine similarity.

---

## 📊 Evaluation Results

Evaluated on a held-out subset of UCF-101 test videos:

| Metric | Score |
|---|---|
| Recall@1 | 0.9797 |
| Recall@3 | 0.9737 |
| Recall@5 | 0.9564 |

**Recall@k** measures how often the correct action class appears within the top-k retrieved results. A score of 0.98 at k=1 means the system returns the correct class as the top result ~98% of the time.

The `evaluation.ipynb` notebook reproduces these numbers and renders a t-SNE plot showing how well TimeSformer embeddings separate the 101 action classes in 2D space.

---

## 🗺️ How It Works

1. **Frame sampling** — 8 frames are uniformly sampled from each video.
2. **Embedding** — Frames are passed through TimeSformer; the CLS token output (768-dim) is taken as the video representation.
3. **Normalization** — Embeddings are L2-normalized so that FAISS inner-product search equals cosine similarity.
4. **Indexing** — All training-set embeddings are stored in a FAISS `IndexFlatIP` index alongside a `{index_id → file_path}` map.
5. **Query** — A query video goes through the same pipeline; FAISS returns the top-k nearest neighbors in under 1 ms.
6. **Preview** — The backend generates animated GIF previews on the fly from the raw video files.