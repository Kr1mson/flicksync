import os
import sys
import asyncio
import io
import streamlit as st
import faiss
import pickle
import numpy as np
import cv2
import torch
from transformers import TimesformerModel, TimesformerConfig
import imageio
import time
from video_processing import load_video_frames_opencv, extract_embedding

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
@st.cache_resource
def load_timesformer():
    
    config = TimesformerConfig.from_pretrained("facebook/timesformer-base-finetuned-k400")
    model = TimesformerModel.from_pretrained(
        "facebook/timesformer-base-finetuned-k400",
        trust_remote_code=True,
        use_safetensors=True).to(device)
    
    return model

def get_video_gif(video_path, n_frames=10, output_size=(224,224)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idxs = list(map(int, np.linspace(0, frame_count - 1, n_frames)))
    
    frames = []
    idx = 0
    retrieved = 0
    while cap.isOpened() and retrieved < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in frame_idxs:
            frame = cv2.resize(frame, output_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            retrieved += 1
        idx += 1
    cap.release()

    if not frames:
        return None
    
    duration = (1.0 / fps) * (frame_count / n_frames) if fps > 0 and frame_count > 0 else 0.1
    
    gif_bytes = io.BytesIO()
    imageio.mimsave(gif_bytes, frames, format='GIF', duration=duration, loop=0)
    gif_bytes.seek(0)
    return gif_bytes

def main():
    st.set_page_config(page_title="Video Similarity Search", layout="wide")

    
    
    faiss_index = faiss.read_index(f"embeddings/faiss_ucf101.index")
    with open(f"embeddings/embedding_map.pkl", "rb") as f:
        id_map= pickle.load(f)

    model = load_timesformer()

    st.title("Video Similarity Search")
    st.write("Upload a video to find similar content")

    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        file_ext = uploaded_file.name.split(".")[-1]
        temp_path = f"temp_video.{file_ext}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Find Similar Videos"):
            with st.spinner("Processing video..."):
                frames = load_video_frames_opencv(temp_path)
                gif_bytes = get_video_gif(temp_path)

                if len(frames)>=8:
                    st.subheader("Uploaded Video Preview")
                    if gif_bytes is not None:
                        st.image(gif_bytes, width=300)
                    else:
                        st.write("Preview not available")

                    start_emb_time = time.time()
                    embs = extract_embedding(frames, model, device)
                    end_emb_time = time.time()
                    embedding_duration = end_emb_time - start_emb_time
                    st.success(f"Embedding extraction time: {embedding_duration:.2f} seconds")

                    k = 5
                    
                    
                    st.write(f"Results for CLS embeddings:")
                    start_search_time = time.perf_counter()
                    D, I = faiss_index.search(embs, k)
                    end_search_time = time.perf_counter()
                    search_duration = end_search_time - start_search_time
                    st.info(f"Search time for CLS embeddings: {search_duration*1000:.2f} ms")

                    cols = st.columns(k)
                    for col, idx in zip(cols, I[0]):
                        video_path = id_map.get(idx, None)
                        with col:
                            if video_path:
                                gif_bytes = get_video_gif(video_path)
                                if gif_bytes is not None:
                                    col.image(gif_bytes, use_container_width=True)
                                    st.write(f"video_path: {video_path}")
                                else:
                                    col.write("Preview not available")
                            else:
                                col.write("Video not found")
                    
                else:
                    st.error("Not enough frames extracted from the video.")

            if os.path.exists(temp_path):
                os.remove(temp_path)



if __name__ == "__main__":
    main()
