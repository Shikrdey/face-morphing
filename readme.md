# Face Morphing & Pair Difficulty Ranker

This repository contains two Python scripts designed to evaluate candidate face pairs and generate a smooth, high-quality face morphing video (`morph.mp4`) using **MediaPipe Face Landmarker**, **Delaunay Triangulation (`scipy`)**, and **OpenCV**.

It also includes an automated binary-search **FFmpeg** encoder that maximizes video quality while strictly keeping the final `.mp4` file size under **98 KB**.

## Project Structure

1. **`make_morph.py`**: Downloads the start and end face images, aligns them to a $512 \times 512$ canvas using 478 facial landmarks, computes a Delaunay triangulation mesh, renders 72 intermediate frames using cosine easing, and encodes the final `morph.mp4` under 98 KB.

2. **`rank_pairs.py`**: Evaluates a list of 10 candidate face pairs and ranks them from **Easiest** to **Hardest** to morph based on two metrics:
   * **Pose & Shape Difference** (Landmark alignment residual error)
   * **Obstacle & Tone Difference** (Lighting, color vs. B&W, glasses, facial hair, or occlusions)

3. **`requirements.txt`**: Contains the exact pinned Python package versions needed to run the scripts smoothly.

## Prerequisites & Setup (Virtual Environment)

Make sure you have **Python 3.10+** installed. To avoid package conflicts, set up a virtual environment (`venv`) and install the dependencies from `requirements.txt`:

### 1. Create and Activate the Virtual Environment

**On Linux / macOS / WSL:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install Required Dependencies

Once the `venv` is activated, install all required packages using `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** You do **not** need to install FFmpeg separately on your system. The `imageio-ffmpeg` package inside `requirements.txt` automatically provides a working FFmpeg binary.

## How to Use Your Own Images (Where to Make Changes)

If you have been assigned different images or candidate pairs, follow the instructions below to update the scripts:

### 1. Updating Images in `make_morph.py`

#### Option A: Using Image URLs

Open `make_morph.py` and locate the `FILES` dictionary (around **Lines 19–29**). Replace the URLs for `"start.jpg"` and `"end.jpg"` with your own image URLs:

```python
FILES = {
    "start.jpg": (
        "YOUR_START_IMAGE_URL_HERE"
    ),
    "end.jpg": (
        "YOUR_END_IMAGE_URL_HERE"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
}
```

> **Important:** The script only downloads `start.jpg` and `end.jpg` if they do not already exist in your project folder. **If you change the URLs after running the script once, make sure to delete the old `start.jpg` and `end.jpg` files first**, otherwise the script will keep using the old images!

#### Option B: Using Local Image Files

If you already have the images saved on your computer:

1. Rename your first image to `start.jpg` and your second image to `end.jpg`.
2. Place both files in the same directory as `make_morph.py`. The script will automatically detect them and skip downloading.

### 2. Updating Candidate Pairs in `rank_pairs.py`

Open `rank_pairs.py` and locate the `PAIRS` list (around **Lines 13–34**). Replace the URLs inside each tuple `(pair_id, url1, url2)` with your assigned image URLs:

```python
PAIRS = [
    (1, "PAIR_1_FIRST_IMAGE_URL", "PAIR_1_SECOND_IMAGE_URL"),
    (2, "PAIR_2_FIRST_IMAGE_URL", "PAIR_2_SECOND_IMAGE_URL"),
    # Add or replace the rest of your pairs here...
]
```

> **Tips for `rank_pairs.py`:**
>
> * **Use Direct Wikimedia Links:** If your images are hosted on Wikimedia Commons, use direct `https://upload.wikimedia.org/wikipedia/commons/...` links rather than `/thumb/` URLs to avoid `HTTP 429 Too Many Requests` rate-limiting.
> * **Clearing Cache:** `rank_pairs.py` caches downloaded images inside `/tmp/pairs_cache/`. If you change the URLs in `PAIRS`, delete the cache folder (`rm -rf /tmp/pairs_cache`) before running the script again so it fetches the new images.

### 3. Customizing Video Parameters (Optional)

If your assignment requires a different resolution, frame count, or file size limit, you can adjust these constants in `make_morph.py` (**Lines 37–40**):

```python
SIZE = 512                # Output video resolution (512x512)
FPS = 24                  # Frame rate (24 frames per second)
NUM_FRAMES = 72           # Total frames (72 frames @ 24 FPS = 3.0 seconds)
TARGET_MAX_BYTES = 98000  # Maximum allowed MP4 file size in bytes (~98 KB)
```

### 4. Note for Windows Users

Both scripts use `/tmp/morph_work` and `/tmp/pairs_cache` for fast temporary file storage (ideal for Linux, macOS, WSL, and Google Colab). While Python on Windows usually maps `/tmp` to `C:\tmp`, if you encounter any file path issues on Windows:

* In `make_morph.py` (**Line 15**), change `TEMP_DIR = "/tmp/morph_work"` to `TEMP_DIR = "temp_morph_work"`.
* In `rank_pairs.py` (**Lines 36, 70, 71**), replace `/tmp/pairs_cache` with `pairs_cache`.

## How to Run

Make sure your virtual environment (`venv`) is activated before running the scripts:

### Step 1: Rank Your Candidate Pairs

Run the ranking script to see which pair is the easiest to morph:

```bash
python rank_pairs.py
```

**Sample Output:**

```text
=== RANKING FROM EASIEST (TOP) TO HARDEST (BOTTOM) ===
 1. Pair  8  (Difficulty score: 7.42 | Pose/Shape: 3.81, Obstacle/Tone: 3.61)
 2. Pair  5  (Difficulty score: 8.15 | Pose/Shape: 4.12, Obstacle/Tone: 4.03)
 ...
```

### Step 2: Generate the Morph Video

Run the morph generator script:

```bash
python make_morph.py
```

This will align the faces, render all 72 frames, find the highest quality CRF value that fits under 98 KB, and save **`morph.mp4`** in your current working directory.