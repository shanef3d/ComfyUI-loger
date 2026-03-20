import math
import torch
import torch.nn.functional as F
import numpy as np
import plyfile


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def resize_images_for_loger(images_bhwc: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """
    Resize a (B, H, W, C) ComfyUI IMAGE tensor to (S, C, target_h, target_w).
    Caller adds [None] for the batch dimension expected by the model.
    """
    # (B, H, W, C) -> (B, C, H, W)
    x = images_bhwc.permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return x  # (S, C, H, W)


def normalize_depth_for_viz(depth_s_h_w: torch.Tensor) -> torch.Tensor:
    """
    Per-frame min-max normalize depth to [0, 1].
    Input:  (S, H, W)
    Output: (S, H, W, 1)
    """
    S, H, W = depth_s_h_w.shape
    d = depth_s_h_w.reshape(S, -1)
    d_min = d.min(dim=-1).values.view(S, 1, 1)
    d_max = d.max(dim=-1).values.view(S, 1, 1)
    normalized = (depth_s_h_w - d_min) / (d_max - d_min + 1e-8)
    return normalized.unsqueeze(-1)  # (S, H, W, 1)


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------

def compute_intrinsics(H: int, W: int, fov_degrees: float):
    """Returns (fx, fy, cx, cy) from a symmetric horizontal FOV."""
    fov_rad = math.radians(fov_degrees)
    fx = fy = (W / 2.0) / math.tan(fov_rad / 2.0)
    cx = W / 2.0
    cy = H / 2.0
    return fx, fy, cx, cy


# ---------------------------------------------------------------------------
# Quaternion helpers — copied from LoGeR-github/loger/utils/rotation.py
# ---------------------------------------------------------------------------

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def mat_to_quat_xyzw(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to quaternions (scalar-last, XYZW).
    Input:  (..., 3, 3)
    Output: (..., 4)  — [X, Y, Z, W]
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))

    # Convert from rijk → ijkr (XYZW)
    out = out[..., [1, 2, 3, 0]]
    out = standardize_quaternion(out)
    return out


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """Ensure scalar (W) part is non-negative (XYZW order, scalar-last)."""
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def mat_to_quat_wxyz(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to quaternions in COLMAP's WXYZ (scalar-first) order.
    Input:  (..., 3, 3)
    Output: (..., 4)  — [W, X, Y, Z]
    """
    q_xyzw = mat_to_quat_xyzw(matrix)
    # Reorder [X, Y, Z, W] -> [W, X, Y, Z]
    return q_xyzw[..., [3, 0, 1, 2]]


# ---------------------------------------------------------------------------
# COLMAP parsers  (for reading back cameras.txt / images.txt)
# ---------------------------------------------------------------------------

def _quat_wxyz_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """WXYZ quaternion → 3×3 rotation matrix (float64)."""
    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if n < 1e-10:
        return np.eye(3, dtype=np.float64)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)


def parse_colmap_cameras_txt(cameras_txt: str):
    """
    Parse a COLMAP cameras.txt string (PINHOLE model).
    Returns (W, H, fx, fy, cx, cy).
    """
    for line in cameras_txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 8 and parts[1] == "PINHOLE":
            return (
                int(parts[2]), int(parts[3]),
                float(parts[4]), float(parts[5]),
                float(parts[6]), float(parts[7]),
            )
    raise ValueError("No PINHOLE camera entry found in cameras.txt content")


def parse_colmap_images_txt(images_txt: str) -> list:
    """
    Parse a COLMAP images.txt string.
    Returns list of dicts: {img_id, R (3×3 ndarray, float64), t (3,), name}
    COLMAP stores world-to-camera (R_cw, t_cw).
    """
    frames = []
    lines = images_txt.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        img_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz     = float(parts[5]), float(parts[6]), float(parts[7])
        name = parts[9]
        R = _quat_wxyz_to_matrix(qw, qx, qy, qz)
        t = np.array([tx, ty, tz], dtype=np.float64)
        frames.append({"img_id": img_id, "R": R, "t": t, "name": name})
        if i < len(lines):   # skip the mandatory empty POINTS2D line
            i += 1
    return frames


# ---------------------------------------------------------------------------
# COLMAP / PLY writers
# ---------------------------------------------------------------------------

def write_colmap_cameras_txt(path: str, H: int, W: int, fx: float, fy: float, cx: float, cy: float):
    content = (
        "# Camera list with one line of data per camera:\n"
        "#   CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n"
        "# Number of cameras: 1, mean reprojection error: 0.000\n"
        f"1 PINHOLE {W} {H} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n"
    )
    with open(path, "w") as f:
        f.write(content)
    return content


def write_colmap_images_txt(path: str, camera_poses_twc_np: np.ndarray, image_names: list) -> str:
    """
    Write COLMAP images.txt.

    camera_poses_twc_np: (S, 4, 4) numpy array  — camera-to-world (Twc)
    image_names: list of S strings
    """
    lines = [
        "# Image list with two lines of data per image:\n",
        "#   IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)\n",
        f"# Number of images: {len(image_names)}, mean reprojection error: 0.000\n",
    ]

    for idx, (Twc, name) in enumerate(zip(camera_poses_twc_np, image_names), start=1):
        R_wc = Twc[:3, :3]
        t_wc = Twc[:3, 3]

        # Twc → Tcw
        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc

        q_wxyz = mat_to_quat_wxyz(torch.from_numpy(R_cw.astype(np.float32))).numpy()
        QW, QX, QY, QZ = q_wxyz
        TX, TY, TZ = t_cw

        lines.append(
            f"{idx} {QW:.9f} {QX:.9f} {QY:.9f} {QZ:.9f} "
            f"{TX:.9f} {TY:.9f} {TZ:.9f} 1 {name}\n"
        )
        lines.append("\n")  # mandatory empty POINTS2D line

    content = "".join(lines)
    with open(path, "w") as f:
        f.write(content)
    return content


def write_pointcloud_ply(
    path: str,
    points: torch.Tensor,
    colors: torch.Tensor,
    conf: torch.Tensor,
    conf_threshold: float,
    downsample: int = 1,
    keep_random_points: float = 1.0,
):
    """
    Write a PLY point cloud filtered by confidence, with optional spatial
    downsampling and random subsampling.

    points: (S, H, W, 3)  world coordinates
    colors: (S, H, W, 3)  float32 [0, 1]
    conf:   (S, H, W, 1)  confidence [0, 1]
    downsample: keep every Nth pixel in H and W (1 = no downsampling)
    keep_random_points: fraction of surviving points to keep (1.0 = all)

    Each point carries: x y z red green blue confidence view_id
    """
    S, H, W, _ = points.shape

    # --- spatial downsampling on the grid ---
    step = max(1, int(downsample))
    pts  = points[:, ::step, ::step, :]   # (S, H', W', 3)
    cols = colors[:, ::step, ::step, :]   # (S, H', W', 3)
    cfs  = conf[  :, ::step, ::step, :]   # (S, H', W', 1)

    Sp, Hp, Wp, _ = pts.shape

    # build per-point view_id: frame index repeated for each spatial position
    view_ids = np.repeat(np.arange(Sp, dtype=np.int32), Hp * Wp)  # (S*H'*W',)

    pts_flat  = pts.reshape(-1, 3).numpy()
    col_flat  = (cols.reshape(-1, 3).numpy() * 255).clip(0, 255).astype(np.uint8)
    conf_flat = cfs.reshape(-1).numpy()

    # --- confidence mask ---
    mask = conf_flat > conf_threshold
    pts_flat  = pts_flat[mask]
    col_flat  = col_flat[mask]
    conf_vals = conf_flat[mask]
    view_ids  = view_ids[mask]

    # --- random subsampling ---
    frac = float(keep_random_points)
    if 0.0 < frac < 1.0 and len(pts_flat) > 0:
        n_keep = max(1, int(len(pts_flat) * frac))
        idx = np.random.choice(len(pts_flat), size=n_keep, replace=False)
        pts_flat  = pts_flat[idx]
        col_flat  = col_flat[idx]
        conf_vals = conf_vals[idx]
        view_ids  = view_ids[idx]

    n = len(pts_flat)
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("confidence", "f4"),
        ("view_id", "i4"),
    ]
    if n == 0:
        vertex_data = np.zeros(0, dtype=dtype)
    else:
        vertex_data = np.empty(n, dtype=dtype)
        vertex_data["x"]          = pts_flat[:, 0]
        vertex_data["y"]          = pts_flat[:, 1]
        vertex_data["z"]          = pts_flat[:, 2]
        vertex_data["red"]        = col_flat[:, 0]
        vertex_data["green"]      = col_flat[:, 1]
        vertex_data["blue"]       = col_flat[:, 2]
        vertex_data["confidence"] = conf_vals
        vertex_data["view_id"]    = view_ids

    vertex_el = plyfile.PlyElement.describe(vertex_data, "vertex")
    plyfile.PlyData([vertex_el], text=False).write(path)
