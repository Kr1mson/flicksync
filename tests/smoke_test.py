import time
import pickle
import torch
import faiss
from transformers import TimesformerModel
from utility.video_processing import load_video_frames_opencv, extract_embedding
device = "cuda" if torch.cuda.is_available() else "cpu"

model = TimesformerModel.from_pretrained(
    "facebook/timesformer-base-finetuned-k400",
    trust_remote_code=True,
    use_safetensors=True,
    ignore_mismatched_sizes=True
).eval().to(device)

try:
    faiss_index = faiss.read_index("embeddings/faiss_ucf101.index")
    with open("embeddings/embedding_map.pkl", "rb") as f:
        id_map = pickle.load(f)
except FileNotFoundError as e:
    raise SystemExit(f"Missing required file: {e}") from e

path = "UCF101/test/BalanceBeam/v_BalanceBeam_g05_c04.avi"
k=5
start_emb_time = time.perf_counter()

frames = load_video_frames_opencv(path)

# Timing the embedding extraction
t0 = time.perf_counter()
frames = load_video_frames_opencv(path)
t1 = time.perf_counter()
embs = extract_embedding(frames, model, device).astype("float32")
t2 = time.perf_counter()
faiss.normalize_L2(embs)
D, I = faiss_index.search(embs, k)
t3 = time.perf_counter()

print(f"\n{'─'*40}")
print(f"  Frame loading   : {(t1-t0)*1000:7.2f} ms")
print(f"  Embedding       : {(t2-t1)*1000:7.2f} ms")
print(f"  Normalize+Search: {(t3-t2)*1000:7.2f} ms")
print(f"  Total           : {(t3-t0)*1000:7.2f} ms")
print(f"{'─'*40}\n")

# Display results
print(f"{'─'*40}")
print(f"  Query: {path}")
print(f"{'─'*40}")
print(f"  {'Rank':<6} {'Score':>7}  {'Label':<22} {'Path'}")
print(f"  {'─'*4:<6} {'─'*5:>7}  {'─'*22:<22} {'─'*30}")
for rank, (dist, idx) in enumerate(zip(D[0].tolist(), I[0].tolist()), start=1):
    video_path = id_map.get(idx, "N/A")
    parts = video_path.replace("\\", "/").split("/")
    label = parts[-2] if len(parts) >= 2 else "?"
    filename = parts[-1]
    score = max(0, 1 - dist / 4) * 100
    print(f"  #{rank:<5} {score:>6.1f}%  {label:<22} {filename}")
print(f"{'─'*40}\n")
