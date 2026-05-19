import os
import glob
import torch
import numpy as np
import cv2
import sys
from collections import defaultdict
import albumentations as A

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
    # returns dict: video_name -> list of paths
    files = glob.glob(os.path.join(folder_path, "*.jpg"))
    video_dict = defaultdict(list)
    for f in files:
        basename = os.path.basename(f)
        # Assuming format {video_name}_f{frame_idx}_p{face_idx}.jpg
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

    # 1. We need to train FrequencyCNN first.
    # To do that, we sample frames to build a training dataset for FrequencyCNN.
    print("Collecting subset of frequency data to train FrequencyCNN...")
    freq_train_data = []
    freq_train_labels = []

    # Sample equal amount of real and fake videos to strictly prevent majority-class guessing
    min_len = min(len(fake_vids), len(real_vids), 1500)
    sampled_fake = list(fake_vids.keys())[:min_len]
    sampled_real = list(real_vids.keys())[:min_len]

    for vid in sampled_fake:
        for fpath in fake_vids[vid]:
            img = load_face_image(fpath)
            if img is not None:
                freq_train_data.append(frequency_preprocess(img))
                freq_train_labels.append(1)

    for vid in sampled_real:
        for fpath in real_vids[vid]:
            img = load_face_image(fpath)
            if img is not None:
                freq_train_data.append(frequency_preprocess(img))
                freq_train_labels.append(0)

    np_f_train = np.stack(freq_train_data)
    y_train_arr = np.array(freq_train_labels)
    
    print(f"Training FrequencyCNN on {len(np_f_train)} face samples...")
    # This trains FrequencyCNN and returns the model
    _, freq_model = get_frequency_vectors(np_f_train, y_train_arr, device, epochs=20)
    
    # Save the trained FrequencyCNN model for inference
    torch.save(freq_model.state_dict(), "freq_model.pth")
    print("Saved freq_model.pth successfully!")

    # Free memory
    del np_f_train
    del freq_train_data
    del freq_train_labels

    spatial_model = SpatialDenseNet().to(device)
    spatial_model.eval()

    aug_pipeline = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
    ])

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
                
                # Standard features
                sp_pre = spatial_preprocess(img) # (224, 224, 3)
                fr_pre = frequency_preprocess(img) # (224, 224, 1)
                sp_list.append(sp_pre)
                fr_list.append(fr_pre)

                # Augmented features
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
            dummy_y = np.zeros(len(np_fr))
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

            # (Removed the buggy augmentation saving that poisoned the dataset)


    process_and_cache_videos(fake_vids, FAKE_OUT, 1)
    process_and_cache_videos(real_vids, REAL_OUT, 0)

    print("All embeddings cached completely!")
