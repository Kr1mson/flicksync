import os
import time
import pickle
import tempfile
import imageio
import torch
import io
import cv2
import numpy as np
import faiss
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from transformers import TimesformerModel
from utility.video_processing import load_video_frames_opencv, extract_embedding

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def make_gif_b64(video_path, n=12):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    indices = np.linspace(0, total - 1, n, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (320, 180))
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return None
    buf = io.BytesIO()
    duration = max(0.08, (total / fps) / n)
    imageio.mimsave(buf, frames, format="GIF", duration=duration, loop=0, plugin="pillow")
    buf.seek(0)
    import base64
    return base64.b64encode(buf.read()).decode("utf-8")

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/api/search", methods=["POST"])
def search():
    temp_path = None
    display_query_path = None

    try:
        path = None
        k = 5

        if "file" in request.files:
            uploaded_file = request.files["file"]

            if not uploaded_file or not uploaded_file.filename:
                return jsonify({"error": "No file selected"}), 400

            filename = secure_filename(uploaded_file.filename)
            if not allowed_file(filename):
                return jsonify({"error": "Unsupported file type"}), 400

            k = int(request.form.get("k", 5))
            k = max(1, min(k, 100))

            suffix = os.path.splitext(filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                uploaded_file.save(tmp.name)
                temp_path = tmp.name

            path = temp_path
            display_query_path = filename

        else:
            data = request.get_json(silent=True) or {}
            path = (data.get("path") or "").strip()
            k = int(data.get("k", 5))
            k = max(1, min(k, 100))
            display_query_path = path
            if not path:
                return jsonify({"error": "Path or uploaded file is required"}), 400
            if not os.path.exists(path):
                return jsonify({"error": f"File not found: {path}"}), 400

        start_emb_time = time.perf_counter()
        frames = load_video_frames_opencv(path)
        embs = extract_embedding(frames, model, device).astype("float32")
        faiss.normalize_L2(embs)
        emb_ms = (time.perf_counter() - start_emb_time) * 1000

        start_search_time = time.perf_counter()
        D, I = faiss_index.search(embs, k)
        search_ms = (time.perf_counter() - start_search_time) * 1000

        results = []
        for rank, (dist, idx) in enumerate(zip(D[0].tolist(), I[0].tolist()), start=1):
            results.append({
                "rank": rank,
                "id": idx,
                "distance": float(dist),
                "similarity_score": round(float(max(0, 1 - dist / 4) * 100), 2),
                "path": id_map.get(idx)
            })

        query_gif_b64 = make_gif_b64(path)
        print(f"[DEBUG] query_gif_b64 length: {len(query_gif_b64) if query_gif_b64 else 'None'}")
        return jsonify({
            "query_path": display_query_path,
            "query_gif_b64": query_gif_b64,
            "k": k,
            "device": str(device),
            "embedding_time_ms": round(emb_ms, 2),
            "search_time_ms": round(search_ms, 2),
            "results": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "device": str(device),
        "index_size": faiss_index.ntotal
    })

@app.route("/api/thumbnail")
def serve_thumbnail():
    path = request.args.get("path", "").strip()
    if not path or not os.path.exists(path):
        return jsonify({"error": "Not found"}), 404
    gif_b64 = make_gif_b64(path)
    if not gif_b64:
        return jsonify({"error": "No frames extracted"}), 500
    import base64
    response = app.response_class(base64.b64decode(gif_b64), mimetype="image/gif")
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)