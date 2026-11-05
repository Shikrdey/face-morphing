import os
import time
import urllib.request
import urllib.error
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Direct upload.wikimedia.org links avoid thumbnail server rate-limiting
PAIRS = [
    (1, "https://upload.wikimedia.org/wikipedia/commons/d/d5/Official_portrait_of_Kemi_Badenoch_MP_crop_2%2C_2024.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/c/c5/Mickey_Rooney_still.jpg"),
    (2, "https://upload.wikimedia.org/wikipedia/commons/8/81/Martin_Buber_portrait.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/c/c8/Kim_Jeong_Suk_2.jpg"),
    (3, "https://upload.wikimedia.org/wikipedia/commons/c/c0/Selena_in_1988.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/7/76/PMO_IMG_5262_%2840547346173%29_%28cropped%29.jpg"),
    (4, "https://upload.wikimedia.org/wikipedia/commons/5/59/Vera_Wang_2009_portrait_Tribeca.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/f/f7/Max_Born.jpg"),
    (5, "https://upload.wikimedia.org/wikipedia/commons/2/27/Ashley_Judd_ioc_cropped.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/e/ee/Maia_Sandu%2C_September_2025.jpg"),
    (6, "https://upload.wikimedia.org/wikipedia/commons/4/4f/Melanie_Safka_1975_crop.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/c/c6/Portrait_of_Mojtaba_Khamenei.jpg"),
    (7, "https://upload.wikimedia.org/wikipedia/commons/3/37/LanaWembley030725-16_%28cropped_2%29.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/7/7b/Yukawa.jpg"),
    (8, "https://upload.wikimedia.org/wikipedia/commons/5/54/%C3%88ve_Curie%2C_1904-2007_%28cropped%29.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/5/56/JuliannaMargulies-byPhilipRomano.jpg"),
    (9, "https://upload.wikimedia.org/wikipedia/commons/d/d0/Photograph_of_Frantz_Fanon_from_Black_Skin_White_Masks_%281967%29_dust_jacket.webp",
        "https://upload.wikimedia.org/wikipedia/commons/1/13/Amy_Goodman%2C_2010_%28cropped%29.jpg"),
    (10, "https://upload.wikimedia.org/wikipedia/commons/e/ec/Bebe_Rexha_at_the_Grammy_Museum%2C_September_2026_02_%28cropped%29.jpg",
         "https://upload.wikimedia.org/wikipedia/commons/6/65/%D0%92%D0%BB%D0%B0%D0%B4%D0%B8%D0%BC%D0%B8%D1%80_%D0%9F%D1%83%D1%82%D0%B8%D0%BD_%2808-03-2024%29_%28cropped%29_%28higher_res%29_2.jpg"),
]

os.makedirs("/tmp/pairs_cache", exist_ok=True)
base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
detector = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
)

# Wikimedia policy-compliant User-Agent header prevents HTTP 429
HEADERS = {
    "User-Agent": "CourseMorphRanker/1.0 (https://github.com/student-project; academic-assignment@example.org) Python-urllib/3.12"
}

def load_img_and_pts(url, path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        for attempt in range(5):
            try:
                print(f"Downloading {os.path.basename(path)}...")
                req = urllib.request.Request(url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=20) as r, open(path, "wb") as f:
                    f.write(r.read())
                time.sleep(1.5)  # Polite pause between requests
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = 4 * (attempt + 1)
                    print(f"  Rate limited (429). Waiting {wait}s before retry...")
                    time.sleep(wait)
                else:
                    raise

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    # Resize large raw images down before MediaPipe detection for speed
    max_dim = max(img.shape[:2])
    if max_dim > 800:
        scale_down = 800.0 / max_dim
        img = cv2.resize(img, (0, 0), fx=scale_down, fy=scale_down, interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    h, w = img.shape[:2]
    pts = np.array([[lm.x * w, lm.y * h] for lm in res.face_landmarks[0]], dtype=np.float32)
    return img, pts

SIZE = 256
scores = []

for pair_id, url1, url2 in PAIRS:
    img1_raw, pts1_raw = load_img_and_pts(url1, f"/tmp/pairs_cache/p{pair_id}_1.img")
    img2_raw, pts2_raw = load_img_and_pts(url2, f"/tmp/pairs_cache/p{pair_id}_2.img")

    cx, cy = np.mean(pts1_raw, axis=0)
    scale = SIZE / (np.max(np.ptp(pts1_raw, axis=0)) * 1.8)
    M1 = np.array([[scale, 0, SIZE/2 - scale*cx], [0, scale, SIZE*0.53 - scale*cy]], dtype=np.float32)
    img1 = cv2.warpAffine(img1_raw, M1, (SIZE, SIZE), borderMode=cv2.BORDER_REPLICATE)
    pts1 = pts1_raw * scale + M1[:, 2]

    M2, _ = cv2.estimateAffinePartial2D(pts2_raw, pts1, method=cv2.LMEDS)
    img2 = cv2.warpAffine(img2_raw, M2, (SIZE, SIZE), borderMode=cv2.BORDER_REPLICATE)
    pts2 = cv2.transform(pts2_raw.reshape(-1, 1, 2), M2).reshape(-1, 2)

    # 1. Landmark warp distance (pose and facial structure difference)
    geom_err = float(np.mean(np.linalg.norm(pts1 - pts2, axis=1)))
    # 2. Appearance/obstacle difference (beards, glasses, headgear, B&W vs color)
    color_err = float(np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32)))) / 10.0

    total_difficulty = geom_err + color_err
    scores.append((total_difficulty, geom_err, color_err, pair_id))

scores.sort()
print("\n=== RANKING FROM EASIEST (TOP) TO HARDEST (BOTTOM) ===")
for rank, (total, geom, col, pid) in enumerate(scores, 1):
    print(f"{rank:2d}. Pair {pid:2d}  (Difficulty score: {total:.2f} | Pose/Shape: {geom:.2f}, Obstacle/Tone: {col:.2f})")