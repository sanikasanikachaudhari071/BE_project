import os
import cv2
import glob
import kagglehub
import numpy as np

# We'll import local preprocess components
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.preprocess import detect_and_crop_faces, align_face

def extract_faces_from_video(video_path, output_dir, max_frames=5, frame_interval=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Skipping bad video: {video_path}")
        return

    frame_idx = 0
    extracted_count = 0

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if frame_rgb.shape[0] < 50 or frame_rgb.shape[1] < 50:
            frame_idx += 1
            continue

        try:
            faces = detect_and_crop_faces(frame_rgb)
        except Exception as e:
            print("Skipping frame due to MTCNN crash:", e)
            frame_idx += 1
            continue

        for i, f in enumerate(faces):
            aligned = align_face(f["face"], f["landmarks"], f["box"])
            
            # Save raw aligned face as JPEG
            aligned_bgr = cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)
            save_path = os.path.join(output_dir, f"{video_name}_f{frame_idx}_p{i}.jpg")
            cv2.imwrite(save_path, aligned_bgr)
            extracted_count += 1

        frame_idx += 1
        if extracted_count >= max_frames:
            break

    cap.release()

if __name__ == "__main__":
    print("Downloading dataset using kagglehub...")
    # NOTE: Run this in Colab!
    dataset_path = kagglehub.dataset_download("reubensuju/celeb-df-v2")
    print(f"Dataset downloaded to: {dataset_path}")

    # Output paths
    BASE_OUT = "/content/extracted_faces"
    FAKE_OUT = os.path.join(BASE_OUT, "fake")
    REAL_OUT = os.path.join(BASE_OUT, "real")

    os.makedirs(FAKE_OUT, exist_ok=True)
    os.makedirs(REAL_OUT, exist_ok=True)

    print("Scanning entire dataset directory for video files...")
    all_videos = glob.glob(os.path.join(dataset_path, "**", "*.mp4"), recursive=True)
    all_videos.extend(glob.glob(os.path.join(dataset_path, "**", "*.avi"), recursive=True))

    print(f"Total video files found in dataset: {len(all_videos)}")

    for v in all_videos:
        normalized_path = v.replace("\\", "/").lower()
        if "youtube-real" in normalized_path or "celeb-real" in normalized_path:
            # It's a REAL video
            extract_faces_from_video(v, REAL_OUT, max_frames=5)
        elif "celeb-synthesis" in normalized_path:
            # It's a FAKE video
            extract_faces_from_video(v, FAKE_OUT, max_frames=5)
        else:
            # Fallback for unrecognized folders, but usually safe to ignore or just print
            pass
            
    print("Face extraction complete!")
