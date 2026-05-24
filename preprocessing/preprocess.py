import cv2
import numpy as np
import os


# ==========================
# LOAD IMAGE
# ==========================
def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Image not found at {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# ==========================
# SAFE CROP
# ==========================
def safe_crop(image, x, y, w, h):
    h_img, w_img, _ = image.shape

    x = max(0, x)
    y = max(0, y)
    w = min(w, w_img - x)
    h = min(h, h_img - y)

    if w <= 0 or h <= 0:
        return None

    return image[y:y+h, x:x+w]


# ==========================
# FACE DETECTOR
# ==========================
cascade_path = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)
def detect_and_crop_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    results = []

    for (x, y, w, h) in faces:
        face = safe_crop(image, x, y, w, h)
        if face is None:
            continue

        # fake landmarks (for compatibility)
        landmarks = {
            "left_eye": (x + w//3, y + h//3),
            "right_eye": (x + 2*w//3, y + h//3)
        }

        results.append({
            "face": face,
            "box": (x, y, w, h),
            "landmarks": landmarks
        })

    return results

# ==========================
# ALIGN FACE
# ==========================
def align_face(face, landmarks, box):
    x, y, _, _ = box

    left_eye = (
        landmarks["left_eye"][0] - x,
        landmarks["left_eye"][1] - y
    )
    right_eye = (
        landmarks["right_eye"][0] - x,
        landmarks["right_eye"][1] - y
    )

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    center = (face.shape[1] // 2, face.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    aligned = cv2.warpAffine(face, M, (face.shape[1], face.shape[0]))
    return aligned


# ==========================
# RESIZE + NORMALIZE
# ==========================
def resize_and_normalize(face, size=224):
    face = cv2.resize(face, (size, size))
    face = face.astype(np.float32) / 255.0
    return face


# ==========================
# SPATIAL
# ==========================
def spatial_preprocess(face):
    return resize_and_normalize(face)


# ==========================
# FREQUENCY (DCT)
# ==========================
def frequency_preprocess(face, size=224):
    face = resize_and_normalize(face, size)

    gray = cv2.cvtColor((face * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.float32)

    dct = cv2.dct(gray)

    # remove low-frequency (focus on artifacts)
    dct[:20, :20] = 0

    dct = np.log(np.abs(dct) + 1e-8)

    dct = (dct - dct.min()) / (dct.max() - dct.min() + 1e-8)

    return dct[..., np.newaxis].astype(np.float32)


# ==========================
# IMAGE PREPROCESS
# ==========================
def run_preprocessing_multi(image_path):
    image = load_image(image_path)

    faces = detect_and_crop_faces(image)

    # Fallback: If no face is detected, assume the image itself is an already-cropped face!
    if len(faces) == 0:
        faces = [{
            "face": image,
            "box": (0, 0, image.shape[1], image.shape[0]),
            "landmarks": {"left_eye": (0, 0), "right_eye": (0, 0)} # Dummy landmarks
        }]

    spatial_outputs = []
    freq_outputs = []
    meta = []

    for f in faces:
        face = f["face"]

        aligned = align_face(face, f["landmarks"], f["box"])

        spatial_outputs.append(spatial_preprocess(aligned))
        freq_outputs.append(frequency_preprocess(aligned))

        meta.append({
            "box": f["box"],
            "landmarks": f["landmarks"]
        })

    if len(spatial_outputs) == 0:
        return None, None, None

    return (
        np.stack(spatial_outputs),
        np.stack(freq_outputs),
        meta
    )


# ==========================
# VIDEO PREPROCESS
# ==========================
def run_preprocessing_video(video_path, frame_interval=10, max_frames=10):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Skipping bad video: {video_path}")
        return None, None, None

    spatial_all = []
    freq_all = []
    meta_all = []

    frame_idx = 0
    processed_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ✅ skip tiny frames
        if frame_rgb.shape[0] < 50 or frame_rgb.shape[1] < 50:
            frame_idx += 1
            continue

        # ✅ FULL MTCNN PROTECTION
        try:
            faces = detect_and_crop_faces(frame_rgb)
        except Exception as e:
            print("Skipping frame due to MTCNN crash:", e)
            frame_idx += 1
            continue

        # ✅ skip if no faces
        if len(faces) == 0:
            frame_idx += 1
            continue

        for f in faces:
            aligned = align_face(f["face"], f["landmarks"], f["box"])

            spatial_all.append(spatial_preprocess(aligned))
            freq_all.append(frequency_preprocess(aligned))

            meta_all.append({
                "frame": frame_idx,
                "box": f["box"],
                "landmarks": f["landmarks"]
            })
            processed_count += 1

        frame_idx += 1

        if max_frames and processed_count >= max_frames:
            break

    cap.release()

    if len(spatial_all) == 0:
        return None, None, None

    return (
        np.stack(spatial_all),
        np.stack(freq_all),
        meta_all
    )


# ==========================
# MAIN MEDIA FUNCTION
# ==========================
def run_preprocessing_media(media_path, frame_interval=5, max_frames=20):

    ext = os.path.splitext(media_path)[1].lower()

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}

    if ext in image_exts:
        return run_preprocessing_multi(media_path)

    elif ext in video_exts:
        return run_preprocessing_video(
            media_path,
            frame_interval=frame_interval,
            max_frames=max_frames
        )

    else:
        raise ValueError(f"Unsupported file type: {ext}")