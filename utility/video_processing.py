import cv2
import torch
import numpy as np
from torchvision import transforms
device = "cuda" if torch.cuda.is_available() else "cpu"
def load_video_frames_opencv(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < num_frames:
        raise ValueError(f"Video too short: {total_frames} < {num_frames}")
    frame_indices = set()
    attempts = 0
    while len(frame_indices) < num_frames and attempts < 3:
        offsets = np.linspace(0, total_frames-1, num_frames-len(frame_indices))
        frame_indices.update(int(round(o)) for o in offsets)
        attempts += 1
    frames = []
    for idx in sorted(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    if len(frames) < num_frames:
        last_frame = frames[-1]
        frames += [last_frame] * (num_frames - len(frames))
    video_tensor = torch.tensor(np.array(frames), dtype=torch.float32).permute(0, 3, 1, 2)
    video_tensor /= 255.0
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    return transform(video_tensor)  # [T, C, H, W]
def extract_embedding(video_tensor, model, device):
    inputs = video_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(inputs)
        cls_embedding = outputs.last_hidden_state[:, 0]
    return cls_embedding.cpu().numpy()