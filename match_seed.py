import cv2
import numpy as np
import sys
import os
import re
import glob


# ── Coordinate parsing ────────────────────────────────────────────────────────

def parse_xy_from_filename(filename: str) -> tuple[float, float] | None:
    """
    Extract (x, y) physical coordinates from a filename like '(3.94,-1.34).JPG'.
    Returns (x, y) as floats, or None if not parseable.
    """
    match = re.search(r'\(([^,]+),([^,)]+)', filename)
    if match:
        try:
            return float(match.group(1)), float(match.group(2))
        except ValueError:
            return None
    return None


# ── Feature matching ──────────────────────────────────────────────────────────

def match_features_flann(des1: np.ndarray, des2: np.ndarray, ratio: float = 0.80) -> list:
    """Match descriptors using FLANN + Lowe's ratio test."""
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return []
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    return [m for m, n in matches if m.distance < ratio * n.distance]


# ── Source image GCP matching ─────────────────────────────────────────────────

def find_source_image(seed_gray: np.ndarray,
                      sift: cv2.SIFT,
                      source_dir: str = "received_images") -> dict:
    """
    Match the seed image against all source images in source_dir.
    Returns the best-matching source with its physical (x, y) and inlier count.
    """
    extensions = ("*.JPG", "*.jpg", "*.JPEG", "*.jpeg", "*.PNG", "*.png")
    source_paths = []
    for ext in extensions:
        source_paths.extend(glob.glob(os.path.join(source_dir, ext)))

    if not source_paths:
        print(f"  WARNING: No source images found in '{source_dir}'")
        return {}

    best = {"file": None, "inliers": 0, "physical_xy": None}

    kp_seed, des_seed = sift.detectAndCompute(seed_gray, None)
    if des_seed is None or len(kp_seed) < 4:
        return best

    for src_path in source_paths:
        src_img = cv2.imread(src_path)
        if src_img is None:
            continue

        # Downscale source image to max 1000px for speed
        sh, sw = src_img.shape[:2]
        scale = min(1.0, 1000.0 / max(sh, sw))
        if scale < 1.0:
            src_img = cv2.resize(src_img, (int(sw * scale), int(sh * scale)),
                                 interpolation=cv2.INTER_AREA)

        src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        kp_src, des_src = sift.detectAndCompute(src_gray, None)

        good = match_features_flann(des_seed, des_src, ratio=0.80)
        if len(good) < 4:
            continue

        # RANSAC to count inliers
        src_pts = np.float32([kp_seed[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_src[m.trainIdx].pt  for m in good]).reshape(-1, 1, 2)
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        inliers = int(mask.sum()) if mask is not None else 0

        if inliers > best["inliers"]:
            xy = parse_xy_from_filename(os.path.basename(src_path))
            best = {
                "file":        os.path.basename(src_path),
                "inliers":     inliers,
                "physical_xy": xy,
            }

    return best


# ── Main matching function ────────────────────────────────────────────────────

def match_seeds_with_panorama(
        thumb_path:    str       = "panorama_128x128.jpg",
        fullres_path:  str       = "panorama_stitcher.jpg",
        seed_paths:    list[str] = None,
        source_dir:    str       = "received_images",
        output_result_path: str  = "result.jpg",
) -> list[dict]:
    """
    1. Match each seed against the full-res panorama → pixel location.
    2. Match each seed against all source images    → physical (x, y) in metres.
    3. Draw centre dots on the 128x128 thumbnail.
    """
    if seed_paths is None:
        seed_paths = ["seeded_image_1.png", "seeded_image_2.png", "seeded_image_3.png","seeded_image_4.png"]

    # ── Load thumbnail ────────────────────────────────────────────────────────
    if not os.path.exists(thumb_path):
        print(f"Error: '{thumb_path}' not found!"); return []
    thumb = cv2.imread(thumb_path)
    if thumb is None:
        print(f"Error: Could not load '{thumb_path}'"); return []
    ph_t, pw_t = thumb.shape[:2]

    # ── Load full-res panorama ────────────────────────────────────────────────
    if not os.path.exists(fullres_path):
        print(f"Error: '{fullres_path}' not found!"); return []
    panorama_full = cv2.imread(fullres_path)
    if panorama_full is None:
        print(f"Error: Could not load '{fullres_path}'"); return []
    ph_f, pw_f = panorama_full.shape[:2]

    # Downscale panorama to 1500px wide for SIFT matching
    match_w  = 1500
    scale_pan = match_w / pw_f
    match_h  = int(ph_f * scale_pan)
    pan_match = cv2.resize(panorama_full, (match_w, match_h), interpolation=cv2.INTER_AREA)
    pan_gray  = cv2.cvtColor(pan_match, cv2.COLOR_BGR2GRAY)

    # Scale factors: match-space → 128x128 thumb
    sx = pw_t / match_w
    sy = ph_t / match_h

    # ── SIFT on panorama ──────────────────────────────────────────────────────
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.004, edgeThreshold=50)
    kp_pan, des_pan = sift.detectAndCompute(pan_gray, None)

    # ── Visualization canvas (4x thumbnail) ──────────────────────────────────
    scale_vis = 4
    vis = cv2.resize(thumb, (pw_t * scale_vis, ph_t * scale_vis),
                     interpolation=cv2.INTER_NEAREST)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0),(0, 0, 255)]   # R, G, B (BGR)

    print("\n" + "=" * 70)
    print("   SIFT MATCHING: Seeded Images -> Panorama + Physical Location")
    print("=" * 70)
    print(f"Thumbnail          : {thumb_path}  ({pw_t}x{ph_t} px)")
    print(f"Full-Res Panorama  : {fullres_path}  ({pw_f}x{ph_f} px)")
    print(f"Matching at        : {match_w}x{match_h} px")
    print(f"Panorama Features  : {len(kp_pan)} keypoints")
    print(f"Source images dir  : {source_dir}")

    results = []

    for i, seed_path in enumerate(seed_paths):
        color        = colors[i % len(colors)]
        seed_basename = os.path.basename(seed_path)

        print(f"\n{'-' * 70}")
        print(f"[Seed {i + 1}] {seed_basename}")

        if not os.path.exists(seed_path):
            print(f"  ERROR: '{seed_path}' not found. Skipping."); continue

        seed_img = cv2.imread(seed_path)
        if seed_img is None:
            print(f"  ERROR: Could not read '{seed_path}'. Skipping."); continue

        # Upscale seed to 512x512 for better SIFT coverage
        seed_match = cv2.resize(seed_img, (512, 512), interpolation=cv2.INTER_CUBIC)
        seed_gray  = cv2.cvtColor(seed_match, cv2.COLOR_BGR2GRAY)

        kp_seed, des_seed = sift.detectAndCompute(seed_gray, None)
        print(f"  Seed Features     : {len(kp_seed)} keypoints")

        if des_seed is None or len(kp_seed) < 4:
            print(f"  ERROR: Not enough features. Skipping."); continue

        # ── Step 1: Match seed → panorama (pixel location) ───────────────────
        good_matches = match_features_flann(des_seed, des_pan, ratio=0.80)
        print(f"  Panorama Matches  : {len(good_matches)}")

        cx_t = cy_t = cx_f = cy_f = None
        pan_status = "INSUFFICIENT_MATCHES"

        if len(good_matches) >= 4:
            src_pts = np.float32([kp_seed[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_pan[m.trainIdx].pt  for m in good_matches]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is not None and mask is not None:
                pan_inliers   = int(mask.sum())
                inlier_dst    = dst_pts[mask.ravel() == 1].reshape(-1, 2)
                cx_m          = float(inlier_dst[:, 0].mean())
                cy_m          = float(inlier_dst[:, 1].mean())
                cx_t          = float(np.clip(cx_m * sx, 0, pw_t - 1))
                cy_t          = float(np.clip(cy_m * sy, 0, ph_t - 1))
                cx_f          = cx_m / scale_pan
                cy_f          = cy_m / scale_pan
                pan_status    = "OK"
                print(f"  Panorama Inliers  : {pan_inliers} / {len(good_matches)}")
                print(f"  Pixel Location    : ({cx_f:.1f}, {cy_f:.1f}) px  "
                      f"[thumb: ({cx_t:.2f}, {cy_t:.2f})]")
            else:
                print(f"  WARNING: Homography failed for panorama match.")
        else:
            print(f"  WARNING: Too few matches for panorama.")

        # ── Step 2: Match seed → source images (physical location) ───────────
        print(f"  Searching source images for physical location...")
        gcp = find_source_image(seed_gray, sift, source_dir)

        physical_xy  = gcp.get("physical_xy")
        source_file  = gcp.get("file", "Unknown")
        gcp_inliers  = gcp.get("inliers", 0)

        if physical_xy:
            print(f"  Best Source Match : {source_file}  ({gcp_inliers} inliers)")
            print(f"  Physical Location : x = {physical_xy[0]:.3f} m,  "
                  f"y = {physical_xy[1]:.3f} m")
        else:
            print(f"  WARNING: Could not determine physical location.")

        result = {
            "seed_index":       i + 1,
            "seed_file":        seed_basename,
            "panorama_status":  pan_status,
            "pixel_fullres":    (round(cx_f, 1), round(cy_f, 1)) if cx_f is not None else None,
            "pixel_thumb":      (round(cx_t, 2), round(cy_t, 2)) if cx_t is not None else None,
            "source_file":      source_file,
            "source_inliers":   gcp_inliers,
            "physical_x":       physical_xy[0] if physical_xy else None,
            "physical_y":       physical_xy[1] if physical_xy else None,
        }
        results.append(result)

        # Draw centre dot on visualization
        if cx_t is not None:
            cx_s = int(cx_t * scale_vis)
            cy_s = int(cy_t * scale_vis)
            cv2.circle(vis, (cx_s, cy_s), 7, color, -1)
            cv2.circle(vis, (cx_s, cy_s), 10, (255, 255, 255), 2)
            label = (f"S{i+1} ({physical_xy[0]:.2f},{physical_xy[1]:.2f})m"
                     if physical_xy else f"S{i+1}")
            cv2.putText(vis, label, (cx_s + 12, cy_s - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  SUMMARY — Seeded Image Physical Locations")
    print(f"{'=' * 70}")
    print(f"  {'Seed':<6} {'Source File':<24} {'Pixel (fullres)':<20} "
          f"{'Physical X (m)':<16} {'Physical Y (m)':<16} {'Status'}")
    print(f"  {'-'*6} {'-'*24} {'-'*20} {'-'*16} {'-'*16} {'-'*10}")
    for r in results:
        px_str  = (f"({r['pixel_fullres'][0]}, {r['pixel_fullres'][1]})"
                   if r['pixel_fullres'] else "N/A")
        phx_str = f"{r['physical_x']:.3f}" if r['physical_x'] is not None else "N/A"
        phy_str = f"{r['physical_y']:.3f}" if r['physical_y'] is not None else "N/A"
        print(f"  S{r['seed_index']:<5} {r['source_file']:<24} {px_str:<20} "
              f"{phx_str:<16} {phy_str:<16} {r['panorama_status']}")
    print(f"{'=' * 70}\n")

    cv2.imwrite(output_result_path, vis)
    print(f"Saved visualization -> {output_result_path}")
    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    thumb   = "panorama_128x128.jpg"
    fullres = "panorama_stitcher.jpg"
    src_dir = "received_images"

    if len(sys.argv) > 1: thumb   = sys.argv[1]
    if len(sys.argv) > 2: fullres = sys.argv[2]
    if len(sys.argv) > 3: src_dir = sys.argv[3]

    seeds = ["seeded_image_1.png", "seeded_image_2.png", "seeded_image_3.png","seeded_image_4.png"]
    match_seeds_with_panorama(thumb, fullres, seeds, src_dir)