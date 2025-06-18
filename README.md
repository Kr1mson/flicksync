# FlickSync

## Reverse Video Search using Timesformer & FAISS

This project implements a **video similarity search system** using the [Timesformer](https://arxiv.org/abs/2102.05095) transformer model (pretrained on Kinetics-400) to generate video embeddings, and [FAISS](https://github.com/facebookresearch/faiss) for efficient nearest neighbor search. Users can upload a video, and the app will return visually similar videos from the UCF101 dataset using precomputed embedding indexes.

---

## ✨ Features

- **Video Embedding:** Uses Timesformer to extract powerful video representations.
- **Similarity Search:** Efficiently retrieves similar videos using FAISS vector search.
- **Interactive Frontend:** Built with Streamlit for easy video upload and result visualization.
- **GIF Previews:** Generates GIF previews for both uploaded and retrieved videos.

---

## 🗂️ Project Structure

```
team-name/
├── README.md
├── Project_Requirements_doc.md
├── src/
|   ├── embedder.ipynb # Notebook for embedding generation & FAISS indexing
|   └── frontend.py # Streamlit frontend app
├── docs/
│   ├── architecture_diagram.png
│   └── tech_stack.md
├── demo_folder/
|   ├── demo/
|   │   ├── demo_embeddings/ # demo embeddings generated on first 20 classes of UCF101
|   |   ├── demo_data_subsets/
|   │   └── demo.py
│   └── screenshots/
└── team_info.doc
```

---

## 📂 Dataset

- The system uses the [UCF101](https://www.crcv.ucf.edu/research/data-sets/ucf101/) action recognition dataset, which contains 13,320 videos across 101 action categories.

- The embedder.ipynb notebook (inside src/) supports generating Timesformer embeddings for all 101 classes, enabling full-scale similarity search.

- For a quick test and faster demo experience, a precomputed FAISS index is included in the demo_folder/embeddings/ directory.

- This allows the app to run immediately without requiring full dataset processing.

---

## 🛠️ Getting Started

**Requirements:**

- Python 3.8+
- Jupyter Notebook
- PyTorch
- `transformers`, `datasets`, `pandas`, `scikit-learn`, and other standard ML/NLP libraries

**Setup:**

1. Clone the repository.
2. Install dependencies:

```bash
pip install requirements.txt
```

4. Open `embedder.ipynb` to generate embeddings for the videos.
5. Use `frontend.py` to search for similar videos and compare the different pooling strategies.

---

## 🧠 Models

- Timesformer (default, Hugging Face)

- Easily extensible to other video transformer models

- Leverages CLS pooling to generate contextually informed embeddings

---

## 🚀 Demo Setup

- Install all libraries using requirements.txt

- Run demo.py in demo_src

- Download the UCF101 dataset and utilise any of the classes for demo testing

---

## 📊 Results

- Retrieves and displays the top-k most similar videos to a given query using transformer-based embeddings and FAISS.

- Visual previews (GIFs) make it easy to assess retrieval quality.

- Achieves high retrieval accuracy, with combined recall@1: 0.9797, recall@3: 0.9737, and recall@5: 0.9564, indicating that the correct class is almost always among the top results.

- Recall@k measures how often the correct item appears within the top-k retrieved results. A higher recall@k indicates better retrieval performance, meaning the system is more likely to present relevant results to the user quickly.

- The system is efficient and scalable, capable of handling large video datasets and real-time search scenarios by indexing normalized embeddings with FAISS.
