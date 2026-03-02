import cv2
import numpy as np
import os
import webbrowser
# Pure-numpy replacements for scipy — no extra dependencies needed

def uniform_filter1d(arr, size):
    """1D moving average (equivalent to scipy.ndimage.uniform_filter1d)."""
    arr = np.asarray(arr, dtype=float)
    half = size // 2
    out = np.empty_like(arr)
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        out[i] = arr[lo:hi].mean()
    return out

def find_peaks(arr, height=None, distance=None):
    """
    Find local maxima (equivalent to scipy.signal.find_peaks with
    height and distance parameters).
    """
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    peaks = []
    for i in range(1, n - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            if height is None or arr[i] >= height:
                peaks.append(i)
    # Enforce minimum distance: keep only the strongest peak in each window
    if distance and len(peaks) > 1:
        kept = []
        peaks.sort(key=lambda p: arr[p], reverse=True)
        used = set()
        for p in peaks:
            if not any(abs(p - k) < distance for k in used):
                kept.append(p)
                used.add(p)
        peaks = sorted(kept)
    return np.array(peaks), {}

# =============================================================
# CUBE DETECTOR v3 — Final
# Works on any photo with red/green/blue cubes, including
# same-color stacks.
#
# Key improvements over v2:
#   - Sobel-peak seam detection: finds seams using horizontal
#     edge strength in the original image, not mask projection.
#     Works even when top/bottom faces are the same color.
#   - 15% interior exclusion: rejects top/bottom face boundary
#     peaks so only true inter-cube seams are counted.
#   - Width-based cube_unit: blob widths are not inflated by
#     stacking, making the unit estimate robust when all blobs
#     are stacks with no isolated single cubes.
#   - Clean output: only ID number on each cube; all details
#     (color, hex, size) are shown in a legend panel.
#   - IDs assigned top-left → bottom-right.
# =============================================================

# 1. LOAD
image_path = os.path.expanduser("~/Downloads/cubes1.png")   # ← update path
img = cv2.imread(image_path)
if img is None:
    print("Error: image not found at", image_path); exit()

h_img, w_img, _ = img.shape
output = img.copy()
hsv    = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Sobel edge maps — used to locate seams between stacked cubes
sobel_h = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))  # horizontal edges
sobel_v = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))  # vertical edges

# 2. RAW MASKS (pre-morphology) — used for left/right tightening
raw_red1  = cv2.inRange(hsv, np.array([0,   80, 60]), np.array([10,  255, 255]))
raw_red2  = cv2.inRange(hsv, np.array([160, 80, 60]), np.array([180, 255, 255]))
raw_red   = cv2.bitwise_or(raw_red1, raw_red2)
raw_green = cv2.inRange(hsv, np.array([40,  60, 60]), np.array([85,  255, 255]))
raw_blue  = cv2.inRange(hsv, np.array([90,  40, 60]), np.array([135, 255, 255]))
raw_masks = {"red": raw_red, "green": raw_green, "blue": raw_blue}

# 3. MORPHED MASKS — for blob detection
k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,  5))

def process(mask):
    m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)

mask_red   = process(raw_red)
mask_green = process(raw_green)
mask_blue  = process(raw_blue)

# 4. CONNECTED COMPONENTS
def get_blobs(mask, color, min_area=800):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < min_area: continue
        x = stats[i, cv2.CC_STAT_LEFT];  y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]; h = stats[i, cv2.CC_STAT_HEIGHT]
        blobs.append((x, y, x+w, y+h, color, a))
    return blobs

all_blobs = (get_blobs(mask_red,   "red")   +
             get_blobs(mask_green, "green") +
             get_blobs(mask_blue,  "blue"))
# Exclude: bottom-right corner (watermark) AND bottom strip of image.
# The bottom strip filter is critical: if the input image is a previous
# run's output, the legend panel at the bottom will contain colored text
# that gets detected as tiny blobs and corrupts the cube_unit estimate.
all_blobs = [b for b in all_blobs
             if not (b[0] > w_img * 0.88 and b[1] > h_img * 0.88)
             and (b[1]+b[3])/2 < h_img * 0.88]

# 5. CUBE UNIT ESTIMATION
# Use blob WIDTHS (not min-side) for the vertical-split unit.
# Width is unaffected by stacking, so it stays accurate even when
# every visible blob is a stack with no isolated single cubes.
# Use min-side only for horizontal col-splits.
# Only consider blobs with cube-like aspect ratio (0.4–2.5) AND
# minimum area of 5000px — filters out narrow text/shadow slivers.
cube_shaped = [(x1,y1,x2,y2,c,a) for (x1,y1,x2,y2,c,a) in all_blobs
               if 0.4 < (x2-x1) / max(1, y2-y1) < 2.5
               and a > 5000]
top_blobs    = [b for b in cube_shaped if (b[1]+b[3])/2 < h_img * 0.6]
bottom_blobs = [b for b in cube_shaped if (b[1]+b[3])/2 >= h_img * 0.6]

def p20w(blobs):   # 20th-percentile width
    v = [x2-x1 for (x1,y1,x2,y2,c,a) in blobs]
    return int(np.percentile(v, 20)) if v else None
def p20s(blobs):   # 20th-percentile min-side
    v = [min(x2-x1, y2-y1) for (x1,y1,x2,y2,c,a) in blobs]
    return int(np.percentile(v, 20)) if v else None

all_w = p20w(cube_shaped) or 100
all_s = p20s(cube_shaped) or 100

far_unit_w  = p20w(top_blobs)    or all_w
near_unit_w = p20w(bottom_blobs) or all_w
far_unit_s  = p20s(top_blobs)    or all_s
near_unit_s = p20s(bottom_blobs) or all_s

print(f"Perspective: far_w={far_unit_w}  near_w={near_unit_w}")

def unit_row(y):   # expected cube height at depth y (for deciding whether to split)
    t = min(1.0, max(0.0, y / h_img))
    return far_unit_w + t * (near_unit_w - far_unit_w)

def unit_col(y):   # expected cube width at depth y
    t = min(1.0, max(0.0, y / h_img))
    return far_unit_s + t * (near_unit_s - far_unit_s)

# 6. SOBEL SEAM DETECTION
# Each seam between stacked cubes produces a strong horizontal edge in the
# original image (the bottom edge of the upper cube / shadow line).
# We project the Sobel-H response across the blob and find peaks.
#
# 15% interior exclusion: peaks within the top/bottom 15% of the blob are
# the cube's own top-face or bottom-face boundary — not inter-cube seams.
SPLIT_THRESH = 1.35

def find_seams(sobel_slice, length, n_seams):
    """
    Find exactly n_seams seam positions in a 1D Sobel projection.
    Uses unit-based n_seams count so false Sobel peaks can't inflate
    the cube count. Picks the n_seams strongest interior peaks, falling
    back to even spacing if Sobel doesn't find enough.
    """
    smooth   = uniform_filter1d(sobel_slice, size=max(5, length // 20))
    min_dist = max(10, length // (n_seams + 2))
    # Collect all interior candidate peaks (15% exclusion zone at each edge)
    candidates = [i for i in range(1, length-1)
                  if smooth[i] > smooth[i-1] and smooth[i] > smooth[i+1]
                  and smooth[i] >= smooth.mean() * 0.8
                  and 0.15 * length < i < 0.85 * length]
    # Enforce min distance between candidates
    kept = []
    candidates.sort(key=lambda p: smooth[p], reverse=True)
    used = set()
    for p in candidates:
        if not any(abs(p - k) < min_dist for k in used):
            kept.append(p)
            used.add(p)
    seams = sorted(kept[:n_seams])
    # Pad with evenly-spaced fallbacks if Sobel didn't find enough
    for k in range(1, n_seams + 1):
        if len(seams) >= n_seams:
            break
        candidate = int(length * k / (n_seams + 1))
        if not any(abs(candidate - s) < min_dist for s in seams):
            seams.append(candidate)
    return sorted(seams[:n_seams])

def split_blob(x1, y1, x2, y2, color):
    w, h = x2-x1, y2-y1
    cy   = (y1+y2) / 2
    uv   = unit_row(cy)
    uh   = unit_col(cy)

    # Number of cubes decided by unit size; Sobel finds where to cut
    n_rows = round(h / uv) if h / uv > SPLIT_THRESH else 1
    n_cols = round(w / uh) if w / uh > SPLIT_THRESH else 1
    n_rows, n_cols = max(1, n_rows), max(1, n_cols)

    # --- ROW SPLITS ---
    if n_rows > 1:
        proj_h    = np.mean(sobel_h[y1:y2, x1:x2], axis=1)
        row_seams = find_seams(proj_h, h, n_rows - 1)
    else:
        row_seams = []

    # --- COL SPLITS ---
    if n_cols > 1:
        proj_v    = np.mean(sobel_v[y1:y2, x1:x2], axis=0)
        col_seams = find_seams(proj_v, w, n_cols - 1)
    else:
        col_seams = []

    rb = [0] + row_seams + [h]
    cb = [0] + col_seams + [w]

    return [(x1+cb[c], y1+rb[r], x1+cb[c+1], y1+rb[r+1], color)
            for r in range(len(rb)-1) for c in range(len(cb)-1)]

detections = []
for (x1, y1, x2, y2, c, a) in all_blobs:
    detections.extend(split_blob(x1, y1, x2, y2, c))

# 7. NMS
def nms(boxes, iou_thresh=0.3):
    if not boxes: return []
    boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    suppressed = [False] * len(boxes)
    keep = []
    for i, (x1, y1, x2, y2, c) in enumerate(boxes):
        if suppressed[i]: continue
        keep.append(boxes[i])
        for j in range(i+1, len(boxes)):
            if suppressed[j]: continue
            bx1,by1,bx2,by2,_ = boxes[j]
            inter = (max(0, min(x2,bx2)-max(x1,bx1)) *
                     max(0, min(y2,by2)-max(y1,by1)))
            if inter == 0: continue
            iou = inter / ((x2-x1)*(y2-y1)+(bx2-bx1)*(by2-by1)-inter+1e-6)
            if iou > iou_thresh: suppressed[j] = True
    return keep

detections = nms(detections)

# 8. TIGHTEN LEFT/RIGHT (trim dark side-face overshoot)
def tighten_lr(x1, y1, x2, y2, color, thresh=0.25):
    mask = raw_masks[color]
    roi  = mask[y1:y2, x1:x2]
    h, w = roi.shape
    if h == 0 or w == 0: return x1, y1, x2, y2
    col_sums = np.sum(roi > 0, axis=0)
    mc = max(2, h * thresh)
    left  = 0
    while left < w-1 and col_sums[left] < mc:   left  += 1
    right = w-1
    while right > left and col_sums[right] < mc: right -= 1
    return (x1+left, y1, x1+right+1, y2)

boxes = [(*tighten_lr(x1,y1,x2,y2,c), c) for (x1,y1,x2,y2,c) in detections]

# 9. SORT top-left → bottom-right (row band then x-center)
avg_unit = int(unit_row(h_img * 0.6))
boxes.sort(key=lambda b: ((b[1]+b[3])//2 // avg_unit, (b[0]+b[2])//2))

print(f"Detected: {len(boxes)} cubes")

# 10. RENDER — ID number only on each cube
draw_colors = {"red": (0, 0, 220), "green": (0, 200, 0), "blue": (210, 90, 0)}

for i, (x1, y1, x2, y2, color) in enumerate(boxes):
    dc = draw_colors[color]
    cv2.rectangle(output, (x1, y1), (x2, y2), dc, 3)
    label = str(i+1)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    lx, ly = x1+5, y1+5
    # Dark chip so the number is readable over any background
    cv2.rectangle(output, (lx-2, ly-2), (lx+tw+4, ly+th+4), (10,10,10), -1)
    cv2.putText(output, label, (lx+2, ly+th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, dc, 2, cv2.LINE_AA)

# 11. LEGEND PANEL — bottom-right corner
lfont   = cv2.FONT_HERSHEY_SIMPLEX
lscale  = 0.45
lthick  = 1
pad     = 13
line_h  = 21
n_cols  = 2
col_w   = 210
n_rows  = (len(boxes) + n_cols - 1) // n_cols
pw      = col_w * n_cols + pad * 2
ph      = n_rows * line_h + pad * 2 + 26

lx = w_img - pw - 18
ly = h_img - ph - 18

ov = output.copy()
cv2.rectangle(ov, (lx, ly), (lx+pw, ly+ph), (12,12,12), -1)
cv2.addWeighted(ov, 0.80, output, 0.20, 0, output)
cv2.rectangle(output, (lx, ly), (lx+pw, ly+ph), (65,65,65), 1)
cv2.putText(output, "LEGEND", (lx+pad, ly+pad+15),
            lfont, 0.52, (185,185,185), 1, cv2.LINE_AA)

for idx, (x1, y1, x2, y2, color) in enumerate(boxes):
    cx, cy = (x1+x2)//2, (y1+y2)//2
    roi    = img[max(0,cy-8):cy+8, max(0,cx-8):cx+8]
    avg    = np.mean(roi, axis=(0,1))
    hx     = "#{:02X}{:02X}{:02X}".format(int(avg[2]), int(avg[1]), int(avg[0]))
    px_w   = x2-x1

    ci = idx % n_cols;  ri = idx // n_cols
    tx = lx + pad + ci * col_w
    ty = ly + 26 + pad + ri * line_h
    dc = draw_colors[color]
    cv2.putText(output, f"#{idx+1}  {color[0].upper()}  {hx}  {px_w}px",
                (tx, ty), lfont, lscale, dc, lthick, cv2.LINE_AA)

# 12. SAVE + OPEN IN LIVE SERVER
script_dir = os.path.dirname(os.path.abspath(__file__))
out_path   = os.path.join(script_dir, "cubes_output.png")
html_path  = os.path.join(script_dir, "index.html")

cv2.imwrite(out_path, output)
print(f"SUCCESS -> {out_path}")

with open(html_path, "w") as f:
    f.write(f"""<!DOCTYPE html>
<html>
<head>
  <title>Cube Detection</title>
  <style>
    body {{ margin:0; background:#1a1a1a; display:flex; flex-direction:column;
            align-items:center; padding:30px; font-family:monospace; color:#ccc; }}
    h2   {{ margin-bottom:6px; font-size:20px; }}
    p    {{ margin:4px 0 16px; font-size:13px; color:#777; }}
    img  {{ max-width:90vw; border:2px solid #333; border-radius:8px;
            box-shadow:0 4px 20px rgba(0,0,0,.5); }}
  </style>
  <meta http-equiv="refresh" content="2">
</head>
<body>
  <h2>Cube Detection Result</h2>
  <p>{len(boxes)} cubes &nbsp;|&nbsp; auto-refreshes every 2s</p>
  <img src="cubes_output.png">
</body>
</html>""")

webbrowser.open("http://127.0.0.1:5500/index.html")

# =============================================================
# TUNING GUIDE
# ─────────────────────────────────────────────────────────────
# Same-color stack not splitting:   lower  SPLIT_THRESH (1.3)
#                                    or    reduce min_dist in find_seams
# False splits on single cubes:     raise  SPLIT_THRESH (1.6)
#                                    or    raise interior exclusion (0.20)
# Boxes still overlap:              lower  iou_thresh   (0.2)
# Side-face overshoot remains:      raise  tighten_lr thresh (0.35)
# Missing small/distant cubes:      lower  min_area     (500)
# Same-color cubes merging:         lower  k_close size (9x9)
# =============================================================