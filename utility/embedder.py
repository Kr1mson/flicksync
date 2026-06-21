import torch
import os
from video_processing import load_video_frames_opencv, extract_embedding
from transformers import TimesformerModel, TimesformerConfig
import faiss
import pickle
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


model = TimesformerModel.from_pretrained(
    "facebook/timesformer-base-finetuned-k400",
    trust_remote_code=True,
    use_safetensors=True,
    ignore_mismatched_sizes=True)
model.eval().to(device)
print(f"Model loaded on {device}")


video_folder = "UCF101/train"
video_paths = []
for root, _, files in os.walk(video_folder):
    for f in files:
        if f.endswith((".mp4", ".avi")):
            video_paths.append(os.path.join(root, f))
print(f"Total videos found: {len(video_paths)}")


dimension = 768
index = faiss.IndexFlatL2(dimension)
embedding_map = {}
for path in tqdm(video_paths, desc="Processing videos"):
    try:
        frames = load_video_frames_opencv(path)
        emb = extract_embedding(frames, model, device)
        faiss.normalize_L2(emb)
        index.add(emb)
        embedding_map[len(embedding_map)] = path
    except Exception as e:
        print(f"Failed to process {path}: {e}")
print(f"Total videos processed: {len(embedding_map)}")


faiss.write_index(index, "embeddings/faiss_ucf101.index")
with open("embeddings/embedding_map.pkl", "wb") as f:
    pickle.dump(embedding_map, f)
print("Saved successfully.")