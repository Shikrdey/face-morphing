import os
import shutil
import subprocess
import urllib.request
import cv2
import imageio_ffmpeg
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from scipy.spatial import Delaunay

# Clean up any old frames folder in OneDrive so OneDrive stops syncing it
shutil.rmtree("frames", ignore_errors=True)

# Use fast Linux /tmp storage for intermediate work
TEMP_DIR = "/tmp/morph_work"
shutil.rmtree(TEMP_DIR, ignore_errors=True)
os.makedirs(TEMP_DIR, exist_ok=True)

FILES = {
    "start.jpg": (
        "https://upload.wikimedia.org/wikipedia/commons/1/1a/Francoise_hardy_february_1966.jpg"
    ),
    "end.jpg": (
        "https://upload.wikimedia.org/wikipedia/commons/b/b8/Radovan_Karad%C5%BEi%C4%87_trial_judgement_24_March_2016.jpg"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
}

for filename, url in FILES.items():
  if not os.path.exists(filename):
    print(f"Downloading {filename}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(filename, "wb") as out:
      out.write(resp.read())

SIZE = 512
FPS = 24
NUM_FRAMES = 72
TARGET_MAX_BYTES = 98000

base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)


def detect_landmarks(img_bgr):
  img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
  mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
  result = detector.detect(mp_img)
  if not result.face_landmarks:
    raise RuntimeError("No face detected.")
  h, w = img_bgr.shape[:2]
  return np.array(
      [[lm.x * w, lm.y * h] for lm in result.face_landmarks[0]],
      dtype=np.float32,
  )


print("Aligning faces at 512x512...")
img1_raw = cv2.imread("start.jpg", cv2.IMREAD_COLOR)
img2_raw = cv2.imread("end.jpg", cv2.IMREAD_COLOR)

pts1_raw = detect_landmarks(img1_raw)
cx, cy = np.mean(pts1_raw, axis=0)
scale = SIZE / (np.max(np.ptp(pts1_raw, axis=0)) * 1.80)

M1 = np.array(
    [[scale, 0, (SIZE / 2.0) - scale * cx], [0, scale, (SIZE * 0.53) - scale * cy]],
    dtype=np.float32,
)

img1 = cv2.warpAffine(
    img1_raw,
    M1,
    (SIZE, SIZE),
    flags=cv2.INTER_LANCZOS4,
    borderMode=cv2.BORDER_REPLICATE,
)
pts1 = detect_landmarks(img1)

pts2_raw = detect_landmarks(img2_raw)
M2, _ = cv2.estimateAffinePartial2D(pts2_raw, pts1, method=cv2.LMEDS)
img2 = cv2.warpAffine(
    img2_raw,
    M2,
    (SIZE, SIZE),
    flags=cv2.INTER_LANCZOS4,
    borderMode=cv2.BORDER_REPLICATE,
)
pts2 = detect_landmarks(img2)

M_BOUND = SIZE - 2
pts1 = np.clip(pts1, 0, M_BOUND)
pts2 = np.clip(pts2, 0, M_BOUND)

grid_steps = np.linspace(0, M_BOUND, 5, dtype=np.float32)
border = []
for v in grid_steps:
  border.extend([[v, 0], [v, M_BOUND], [0, v], [M_BOUND, v]])
border = np.unique(np.array(border, dtype=np.float32), axis=0)

pts1 = np.vstack([pts1, border])
pts2 = np.vstack([pts2, border])
tri = Delaunay((pts1 + pts2) * 0.5)


def morph_triangle(im1, im2, im_out, t1, t2, t, alpha):
  r1 = cv2.boundingRect(np.float32([t1]))
  r2 = cv2.boundingRect(np.float32([t2]))
  r = cv2.boundingRect(np.float32([t]))
  if min(r1[2], r1[3], r2[2], r2[3], r[2], r[3]) <= 0:
    return

  t1_rect = [((t1[i][0] - r1[0]), (t1[i][1] - r1[1])) for i in range(3)]
  t2_rect = [((t2[i][0] - r2[0]), (t2[i][1] - r2[1])) for i in range(3)]
  t_rect = [((t[i][0] - r[0]), (t[i][1] - r[1])) for i in range(3)]

  mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
  cv2.fillConvexPoly(mask, np.int32(t_rect), (1.0, 1.0, 1.0), 16, 0)

  im1_rect = im1[r1[1] : r1[1] + r1[3], r1[0] : r1[0] + r1[2]]
  im2_rect = im2[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]]

  size = (r[2], r[3])
  mat1 = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t_rect))
  mat2 = cv2.getAffineTransform(np.float32(t2_rect), np.float32(t_rect))

  w1 = cv2.warpAffine(
      im1_rect,
      mat1,
      size,
      None,
      flags=cv2.INTER_LINEAR,
      borderMode=cv2.BORDER_REFLECT_101,
  )
  w2 = cv2.warpAffine(
      im2_rect,
      mat2,
      size,
      None,
      flags=cv2.INTER_LINEAR,
      borderMode=cv2.BORDER_REFLECT_101,
  )

  blended = (1.0 - alpha) * w1 + alpha * w2
  roi = im_out[r[1] : r[1] + r[3], r[0] : r[0] + r[2]]
  roi[:] = roi * (1.0 - mask) + blended * mask


img1_f = np.float32(img1)
img2_f = np.float32(img2)

print("Rendering 72 frames in fast Linux /tmp storage...")
for idx in range(NUM_FRAMES):
  t_norm = np.clip((idx - 8) / float(NUM_FRAMES - 17), 0.0, 1.0)
  alpha = float(0.5 * (1.0 - np.cos(np.pi * t_norm)))
  pts_mid = (1.0 - alpha) * pts1 + alpha * pts2
  frame = (1.0 - alpha) * img1_f + alpha * img2_f

  for indices in tri.simplices:
    morph_triangle(
        img1_f, img2_f, frame, pts1[indices], pts2[indices], pts_mid[indices], alpha
    )

  cv2.imwrite(
      f"{TEMP_DIR}/frame_{idx:03d}.png", np.uint8(np.clip(frame, 0, 255))
  )
  if (idx + 1) % 18 == 0:
    print(f"  Rendered {idx + 1}/{NUM_FRAMES} frames...")

# Fast binary search for the highest quality CRF under 98 KB (takes only ~4 tries)
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
temp_mp4 = f"{TEMP_DIR}/test.mp4"
low, high = 18, 32
best_size = 0
best_crf = None

print("Finding maximum quality under 98 KB...")
while low <= high:
  crf = (low + high) // 2
  subprocess.run(
      [
          ffmpeg_exe,
          "-y",
          "-framerate",
          str(FPS),
          "-i",
          f"{TEMP_DIR}/frame_%03d.png",
          "-c:v",
          "libx264",
          "-crf",
          str(crf),
          "-preset",
          "slow",
          "-pix_fmt",
          "yuv420p",
          "-movflags",
          "+faststart",
          "-an",
          temp_mp4,
      ],
      check=True,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
  )

  size_bytes = os.path.getsize(temp_mp4)
  print(f"  Tested CRF {crf}: {size_bytes / 1000:.1f} KB ({size_bytes} bytes)")

  if size_bytes <= TARGET_MAX_BYTES:
    shutil.copy(temp_mp4, "morph.mp4")
    best_size = size_bytes
    best_crf = crf
    high = crf - 1  # Try even higher quality (lower CRF)
  else:
    low = crf + 1  # File too large, increase compression slightly

print(
    f"\nDone! Saved morph.mp4 ({best_size} bytes / {best_size / 1000:.1f} KB at"
    f" CRF {best_crf})."
)