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
from transformers import TimesformerModel, AutoImageProcessor
import imageio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

@st.cache_resource
def load_timesformer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400",trust_remote_code=True, 
    use_safetensors=True).to(device).eval()
    processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400") 
    return model, processor, device

def main():
    st.set_page_config(page_title="Video Similarity Search", layout="wide")

    faiss_indices = {}
    id_maps = {}
    for emb_type in ["mean", "max", "cls"]:
        faiss_indices[emb_type] = faiss.read_index(f"video_embeddings_{emb_type}.index")
        with open(f"id_map_{emb_type}.pkl", "rb") as f:
            id_maps[emb_type] = pickle.load(f)

    model, processor, device = load_timesformer()

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
                frames = sample_frames(temp_path)
                gif_bytes = get_video_gif(temp_path)

                if frames and len(frames) >= 8:
                    st.subheader("Uploaded Video Preview")
                    if gif_bytes is not None:
                        st.image(gif_bytes, width=300)
                    else:
                        st.write("Preview not available")

                    embs = get_video_embeddings_all(frames, model, processor, device)

                    for key in embs:
                        embs[key] = embs[key] / np.linalg.norm(embs[key])

                    k = 5
                    tabs = st.tabs(["Mean Pooling", "Max Pooling", "CLS Token"])
                    for tab, emb_type in zip(tabs, ["mean", "max", "cls"]):
                        with tab:
                            st.write(f"Results for {emb_type} embeddings:")
                            D, I = faiss_indices[emb_type].search(np.expand_dims(embs[emb_type], axis=0), k)
                            cols = st.columns(k)
                            for col, idx in zip(cols, I[0]):
                                video_path =id_maps[emb_type].get(idx, None)
                                with col:
                                    if video_path:
                                        demo_path = video_path.replace("Dataset/", "")
                                        gif_bytes = get_video_gif(demo_path)
                                        if gif_bytes is not None:
                                            col.image(gif_bytes, use_container_width=True)
                                        else:
                                            col.write(f"{demo_path} not found")
                                    else:
                                        col.write("Video not found")
                else:
                    st.error("Not enough frames extracted from the video.")

            if os.path.exists(temp_path):
                os.remove(temp_path)

def sample_frames(video_path, n_frames=16, output_size=(224,224)):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error opening video file.")
        return frames

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        st.error("Video has no frames.")
        return frames

    frame_idxs = list(map(int, np.linspace(0, frame_count - 1, n_frames)))
    idx = 0
    while cap.isOpened() and len(frames) < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in frame_idxs:
            frame = cv2.resize(frame, output_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        idx += 1
    cap.release()
    return frames

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

def get_video_embeddings_all(frames, model, processor, device):
    frames_tensor = torch.tensor(np.array(frames)).permute(0,3,1,2).unsqueeze(0).to(device).float() / 255.0

    pixel_mean = torch.tensor(processor.image_mean).view(1,3,1,1).to(device)
    pixel_std = torch.tensor(processor.image_std).view(1,3,1,1).to(device)
    frames_tensor = (frames_tensor - pixel_mean) / pixel_std

    with torch.no_grad():
        outputs = model(frames_tensor)

    features = outputs.last_hidden_state  

    mean_pool = features.mean(dim=1).squeeze(0).cpu().numpy()
    max_pool = features.max(dim=1).values.squeeze(0).cpu().numpy()
    cls_token = features[:, 0, :].squeeze(0).cpu().numpy()

    return {
        "mean": mean_pool,
        "max": max_pool,
        "cls": cls_token
    }

if __name__ == "__main__":
    main()
