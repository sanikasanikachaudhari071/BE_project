import os
import glob
import torch
import numpy as np
import cv2
import sys
from collections import defaultdict
import albumentations as A

# Adjust path to import from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.preprocess import spatial_preprocess, frequency_preprocess
from densenet.densenet import SpatialDenseNet
from frequencycnn.frequency import get_frequency_vectors

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_face_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def gather_video_faces(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.jpg"))
    video_dict = defaultdict(list)
    for f in files:
        basename = os.path.basename(f)
        vid_name = basename.split("_f")[0]
        video_dict[vid_name].append(f)
    return video_dict

if __name__ == "__main__":
    BASE_IN = "/content/extracted_faces"
    FAKE_IN = os.path.join(BASE_IN, "fake")
    REAL_IN = os.path.join(BASE_IN, "real")

    BASE_OUT = "/content/embeddings"
    FAKE_OUT = os.path.join(BASE_OUT, "fake")
    REAL_OUT = os.path.join(BASE_OUT, "real")

    os.makedirs(FAKE_OUT, exist_ok=True)
    os.makedirs(REAL_OUT, exist_ok=True)

    fake_vids = gather_video_faces(FAKE_IN)
    real_vids = gather_video_faces(REAL_IN)

    print(f"Found {len(fake_vids)} FAKE sequences, {len(real_vids)} REAL sequences.")

    # ========================================================
    # ROBUST SOCIAL MEDIA AUGMENTATION PIPELINE
    # Simulates Instagram/TikTok compression to improve generalization
    # ========================================================
    aug_pipeline = A.Compose([
        A.ImageCompression(quality_lower=60, quality_upper=90, p=0.8),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.HorizontalFlip(p=0.5),
    ])

    print("Collecting subset of frequency data to train FrequencyCNN...")
    freq_train_data = []
    freq_train_labels = []

    # Limit to 1500 sequences per class for FrequencyCNN training
    min_len = min(len(fake_vids), len(real_vids), 1500)
    sampled_fake = list(fake_vids.keys())[:min_len]
    sampled_real = list(real_vids.keys())[:min_len]

    # Collect BOTH Base and Augmented versions so the model learns compression artifacts
    for vid in sampled_fake:
        for fpath in fake_vids[vid]:
            img = load_face_image(fpath)
            if img is not None:
                freq_train_data.append(frequency_preprocess(img))
                freq_train_labels.append(1)
                
                aug_img = aug_pipeline(image=img)["image"]
                freq_train_data.append(frequency_preprocess(aug_img))
                freq_train_labels.append(1)

    for vid in sampled_real:
        for fpath in real_vids[vid]:
            img = load_face_image(fpath)
            if img is not None:
                freq_train_data.append(frequency_preprocess(img))
                freq_train_labels.append(0)
                
                aug_img = aug_pipeline(image=img)["image"]
                freq_train_data.append(frequency_preprocess(aug_img))
                freq_train_labels.append(0)

    np_f_train = np.stack(freq_train_data)
    y_train_arr = np.array(freq_train_labels)
    
    print(f"Training FrequencyCNN on {len(np_f_train)} face samples (including augmentations)...")
    _, freq_model = get_frequency_vectors(np_f_train, y_train_arr, device, epochs=20)
    
    # Save the trained FrequencyCNN model for inference
    torch.save(freq_model.state_dict(), "freq_model.pth")
    print("Saved freq_model.pth successfully!")

    # Free up memory
    del np_f_train
    del freq_train_data
    del freq_train_labels

    # ==========================
    # CACHE ALL EMBEDDINGS
    # ==========================
    spatial_model = SpatialDenseNet().to(device)
    spatial_model.eval()

    def process_and_cache_videos(vid_dict, out_dir, label):
        print(f"Processing and caching {len(vid_dict)} videos into {out_dir} ...")
        for vid, paths in vid_dict.items():
            sp_list = []
            fr_list = []
            aug_sp_list = []
            aug_fr_list = []
            
            for p in paths:
                img = load_face_image(p)
                if img is None: continue
                
                sp_pre = spatial_preprocess(img) 
                fr_pre = frequency_preprocess(img) 
                sp_list.append(sp_pre)
                fr_list.append(fr_pre)

                # Generate augmented frame
                augmented = aug_pipeline(image=img)
                aug_img = augmented["image"]
                aug_sp_list.append(spatial_preprocess(aug_img))
                aug_fr_list.append(frequency_preprocess(aug_img))
            
            if len(sp_list) == 0: continue
            
            np_sp = np.stack(sp_list)
            np_fr = np.stack(fr_list)
            np_sp_aug = np.stack(aug_sp_list)
            np_fr_aug = np.stack(aug_fr_list)

            # Convert to tensors
            t_sp = torch.tensor(np_sp, dtype=torch.float32).permute(0,3,1,2).to(device)
            t_sp_aug = torch.tensor(np_sp_aug, dtype=torch.float32).permute(0,3,1,2).to(device)
            
            # Apply Densenet
            with torch.no_grad():
                sp_vec = spatial_model(t_sp).cpu()
                sp_vec_aug = spatial_model(t_sp_aug).cpu()

            # Apply trained Freq model
            from frequencycnn.frequency import extract_frequency_features
            fr_vec = extract_frequency_features(np_fr, freq_model, device)
            fr_vec_aug = extract_frequency_features(np_fr_aug, freq_model, device)

            # Save Base
            save_path = os.path.join(out_dir, f"{vid}.pt")
            torch.save({
                "spatial": sp_vec,
                "freq": fr_vec,
                "label": label
            }, save_path)

            # Save Augmented (Properly trained this time!)
            save_path_aug = os.path.join(out_dir, f"{vid}_aug.pt")
            torch.save({
                "spatial": sp_vec_aug,
                "freq": fr_vec_aug,
                "label": label
            }, save_path_aug)

    process_and_cache_videos(fake_vids, FAKE_OUT, 1)
    process_and_cache_videos(real_vids, REAL_OUT, 0)

    print("All embeddings cached completely!")
