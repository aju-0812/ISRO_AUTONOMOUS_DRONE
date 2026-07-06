import cv2
import numpy as np
import glob
import os
import time
from collections import deque


def match_features_ransac(kp1, des1, kp2, des2):
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return 0, None
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    try:
        matches = flann.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.85 * n.distance]
        if len(good) < 4:
            return 0, None
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        if M is None or mask is None:
            return 0, None
        inliers = int(mask.sum())
        # Convert 2x3 affine matrix to 3x3 homography matrix so the rest of the pipeline is compatible
        H = np.eye(3, dtype=np.float64)
        H[:2, :] = M
        return inliers, H
    except Exception as e:
        return 0, None


def get_distance_weight_map(h: int, w: int) -> np.ndarray:
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    y = np.linspace(-cy, h - 1 - cy, h, dtype=np.float32)[:, None]
    x = np.linspace(-cx, w - 1 - cx, w, dtype=np.float32)[None, :]
    dist = np.sqrt(x ** 2 + y ** 2)
    dist_max = dist.max()
    if dist_max == 0:
        return np.ones((h, w), dtype=np.float32)
    return 1.0 - dist / dist_max


def find_largest_inner_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    h, w = mask.shape
    binary = (mask == 255).astype(np.int32)
    heights = np.zeros(w, dtype=np.int32)
    max_area = 0
    best_box = (0, 0, w, h)
    for r in range(h):
        heights = np.where(binary[r] == 1, heights + 1, 0)
        stack = []
        for c in range(w + 1):
            val = heights[c] if c < w else 0
            start = c
            while stack and stack[-1][1] > val:
                pos, height = stack.pop()
                width = c - pos
                area = height * width
                if area > max_area:
                    max_area = area
                    best_box = (pos, r - height + 1, c, r + 1)
                start = pos
            stack.append((start, val))
    return best_box


def main():
    # Resolve received_images directory dynamically (relative to script or fallback)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    received_dir = os.path.join(script_dir, "received_images")
    if not os.path.isdir(received_dir):
        received_dir = ("received_images"
                        "")
    # Match multiple file formats (uppercase & lowercase)
    extensions = ("*.JPG", "*.jpg", "*.JPEG", "*.jpeg", "*.PNG", "*.png")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(received_dir, ext)))
    # Deduplicate and sort paths
    image_paths = list(sorted(set(image_paths)))
    print(f"Found {len(image_paths)} images.")

    # 1. Feature Detection (SIFT) on low-res images for speed and efficiency
    sift = cv2.SIFT_create(nfeatures=2000)
    kp_list = []
    des_list = []
    orig_sizes = []
    S_list = []

    target_max_dim = 1000
    for p in image_paths:
        img = cv2.imread(p)
        h, w = img.shape[:2]
        orig_sizes.append((w, h))

        # Scale matrix for high-res reconstruction
        scale = target_max_dim / max(w, h)
        if scale >= 1.0:
            S = np.eye(3, dtype=np.float64)
            new_w, new_h = w, h
        else:
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            S = np.array([
                [new_w / w, 0.0, 0.0],
                [0.0, new_h / h, 0.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)

        S_list.append(S)
        img_low = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img_low, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        kp_list.append(kp)
        des_list.append(des)
        print(f"Detected features for {os.path.basename(p)}")

    # 2. Match features and estimate pairwise homographies
    n = len(image_paths)
    adj_list = {i: [] for i in range(n)}

    print("\nPairwise matching...")
    for i in range(n):
        for j in range(i + 1, n):
            inliers, H_ji = match_features_ransac(kp_list[j], des_list[j], kp_list[i], des_list[i])
            if inliers >= 75:  # Higher threshold to filter out false texture matches
                try:
                    H_ij = np.linalg.inv(H_ji)
                    adj_list[i].append((j, inliers, H_ji))
                    adj_list[j].append((i, inliers, H_ij))
                    print(
                        f"Match {os.path.basename(image_paths[i])} <-> {os.path.basename(image_paths[j])}: {inliers} inliers")
                except np.linalg.LinAlgError:
                    continue

    # 3. Find Anchor image (highest degree node or middle node)
    degrees = [len(adj_list[i]) for i in range(n)]
    ref_idx = int(np.argmax(degrees))
    print(f"\nSelected anchor image index {ref_idx}: {os.path.basename(image_paths[ref_idx])}")

    # 4. Traverse graph using BFS to find paths to anchor and chain homographies
    # We want to find for each image i, the homography H_i to the anchor ref_idx (so pts_ref = H_i @ pts_i)
    H_chain_low = [None] * n
    H_chain_low[ref_idx] = np.eye(3, dtype=np.float64)

    queue = deque([ref_idx])
    visited = {ref_idx}

    while queue:
        u = queue.popleft()
        # Look at neighbors of u
        for v, inliers, H_vu in adj_list[u]:
            if v not in visited:
                # v is a neighbor of u, H_vu warps v to u.
                # Since H_chain_low[u] warps u to ref, then H_chain_low[u] @ H_vu warps v to ref.
                H_chain_low[v] = H_chain_low[u] @ H_vu
                visited.add(v)
                queue.append(v)

    # Check if any images are not connected to the main component
    unconnected = [i for i in range(n) if H_chain_low[i] is None]
    if unconnected:
        print(
            f"Warning: The following images could not be matched to the panorama and are excluded: {[os.path.basename(image_paths[i]) for i in unconnected]}")

    # Scale the homographies to full resolution
    H_chain_full = []
    S_ref_inv = np.linalg.inv(S_list[ref_idx])
    for i in range(n):
        if H_chain_low[i] is None:
            H_chain_full.append(None)
            continue
        H_full = S_ref_inv @ H_chain_low[i] @ S_list[i]
        if H_full[2, 2] != 0:
            H_full /= H_full[2, 2]
        H_chain_full.append(H_full)

    # 5. Canvas size computation
    # Project corners of all connected images
    all_corners = []
    valid_indices = []
    for i in range(n):
        if H_chain_full[i] is None:
            continue
        w, h = orig_sizes[i]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        all_corners.append(cv2.perspectiveTransform(corners, H_chain_full[i]))
        valid_indices.append(i)

    all_corners = np.concatenate(all_corners, axis=0)
    x_min = int(np.floor(all_corners[:, 0, 0].min()))
    y_min = int(np.floor(all_corners[:, 0, 1].min()))
    x_max = int(np.ceil(all_corners[:, 0, 0].max()))
    y_max = int(np.ceil(all_corners[:, 0, 1].max()))

    offset = np.array(
        [[1.0, 0.0, -x_min],
         [0.0, 1.0, -y_min],
         [0.0, 0.0, 1.0]], dtype=np.float64
    )
    canvas_w = x_max - x_min
    canvas_h = y_max - y_min
    print(f"Canvas computed size: {canvas_w}x{canvas_h}")

    # 6. Warp & Blend (optimized localized warping)
    output_scale = 0.4
    S_out = np.array([
        [output_scale, 0.0, 0.0],
        [0.0, output_scale, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    offset_scaled = S_out @ offset
    canvas_w_out = max(1, int(round(canvas_w * output_scale)))
    canvas_h_out = max(1, int(round(canvas_h * output_scale)))

    result = np.zeros((canvas_h_out, canvas_w_out, 3), dtype=np.float32)
    weight_sum = np.zeros((canvas_h_out, canvas_w_out), dtype=np.float32)

    for i in valid_indices:
        H_adjusted = offset_scaled @ H_chain_full[i]
        img = cv2.imread(image_paths[i])
        h, w = img.shape[:2]

        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, H_adjusted)

        x_min_l = int(np.floor(warped_corners[:, 0, 0].min()))
        y_min_l = int(np.floor(warped_corners[:, 0, 1].min()))
        x_max_l = int(np.ceil(warped_corners[:, 0, 0].max()))
        y_max_l = int(np.ceil(warped_corners[:, 0, 1].max()))

        x_min_l = max(0, x_min_l)
        y_min_l = max(0, y_min_l)
        x_max_l = min(canvas_w_out, x_max_l)
        y_max_l = min(canvas_h_out, y_max_l)

        if x_min_l >= x_max_l or y_min_l >= y_max_l:
            continue

        T_shift = np.array([
            [1.0, 0.0, -x_min_l],
            [0.0, 1.0, -y_min_l],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        H_local = T_shift @ H_adjusted
        W_local = x_max_l - x_min_l
        H_local_size = y_max_l - y_min_l

        warped_local = cv2.warpPerspective(img, H_local, (W_local, H_local_size)).astype(np.float32)
        weight = get_distance_weight_map(h, w)
        weight_warped_local = cv2.warpPerspective(weight, H_local, (W_local, H_local_size))

        result[y_min_l:y_max_l, x_min_l:x_max_l] += warped_local * weight_warped_local[:, :, np.newaxis]
        weight_sum[y_min_l:y_max_l, x_min_l:x_max_l] += weight_warped_local

    safe_weight = weight_sum.copy()
    safe_weight[safe_weight == 0] = 1.0
    blended = (result / safe_weight[:, :, np.newaxis]).astype(np.uint8)

    # 7. Render (crop black borders using largest inner rectangle to remove black margins completely)
    gray = cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    h_m, w_m = mask.shape
    target_crop_dim = 500
    scale = target_crop_dim / max(h_m, w_m)
    if scale < 1.0:
        new_w = int(round(w_m * scale))
        new_h = int(round(h_m * scale))
        mask_low = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    else:
        mask_low = mask.copy()
        scale = 1.0

    kernel = np.ones((3, 3), np.uint8)
    mask_low = cv2.erode(mask_low, kernel, iterations=1)

    x_min_low, y_min_low, x_max_low, y_max_low = find_largest_inner_rectangle(mask_low)
    x_min_c = int(np.floor(x_min_low / scale))
    y_min_c = int(np.floor(y_min_low / scale))
    x_max_c = int(np.ceil(x_max_low / scale))
    y_max_c = int(np.ceil(y_max_low / scale))

    x_min_c = max(0, x_min_c)
    y_min_c = max(0, y_min_c)
    x_max_c = min(w_m, x_max_c)
    y_max_c = min(h_m, y_max_c)

    if x_min_c < x_max_c and y_min_c < y_max_c:
        blended = blended[y_min_c:y_max_c, x_min_c:x_max_c]

    cv2.imwrite("panorama_stitcher.jpg", blended)
    print("Done! Saved to panorama_stitcher.jpg")


if __name__ == "__main__":
    main()
