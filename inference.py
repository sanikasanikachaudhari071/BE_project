import os
import torch
import yt_dlp
import numpy as np
from densenet.densenet import SpatialDenseNet, get_spatial_vectors
from frequencycnn.frequency import FrequencyCNN, extract_frequency_features
from Transformer.transfromermodel import FusionTransformer
from preprocessing.preprocess import run_preprocessing_media

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_models(freq_model_path="freq_model.pth", fusion_model_path="deepfake_model.pth"):
    # 1. Spatial Model (Pretrained DenseNet)
    spatial_model = SpatialDenseNet().to(device)
    spatial_model.eval()

    # 2. Frequency Model
    freq_model = FrequencyCNN().to(device)
    if os.path.exists(freq_model_path):
        freq_model.load_state_dict(torch.load(freq_model_path, map_location=device))
    else:
        print(f"Warning: {freq_model_path} not found. Frequency features will be random!")
    freq_model.eval()

    # 3. Fusion Model
    fusion_model = FusionTransformer().to(device)
    if os.path.exists(fusion_model_path):
        fusion_model.load_state_dict(torch.load(fusion_model_path, map_location=device))
    else:
        print(f"Warning: {fusion_model_path} not found. Predictions will be random!")
    fusion_model.eval()

    return spatial_model, freq_model, fusion_model

def predict_media(media_path, spatial_model, freq_model, fusion_model):
    spatial_np, freq_np, meta = run_preprocessing_media(media_path)
    if spatial_np is None:
        return None, "No faces detected in the media."

    with torch.no_grad():
        # Get embeddings exactly as they were cached during training (without extra normalization)
        t_sp = torch.tensor(spatial_np, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        sp_vecs = spatial_model(t_sp).cpu()
        
        fr_vecs = extract_frequency_features(freq_np, freq_model, device)

        # Fusion model expects batch dimension
        sp_vecs = sp_vecs.unsqueeze(0).to(device)
        fr_vecs = fr_vecs.unsqueeze(0).to(device)

        out = fusion_model(sp_vecs, fr_vecs).squeeze()
        if out.dim() == 0:
            out = out.unsqueeze(0)
            
        prob = torch.sigmoid(out).item()
        
    return prob, meta

def download_video_from_url(url, output_path="downloaded_video.mp4"):
    ydl_opts = {
        'outtmpl': output_path,
        'format': 'best[height<=480][ext=mp4]/best[ext=mp4]',
        'quiet': True,
        'noplaylist': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

