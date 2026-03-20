import sys
import os
import inspect
import yaml
import torch
import numpy as np
from PIL import Image as PILImage

import folder_paths
from comfy_api.latest import ComfyExtension, io

# Default path: ComfyUI/models/LoGeR
_DEFAULT_CKPTS_PATH = os.path.join(folder_paths.models_dir, "LoGeR")

# ---------------------------------------------------------------------------
# Make bundled loger/ package importable
# ---------------------------------------------------------------------------
_NODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _NODE_DIR not in sys.path:
    sys.path.insert(0, _NODE_DIR)

from .utils import (
    resize_images_for_loger,
    normalize_depth_for_viz,
    compute_intrinsics,
    write_colmap_cameras_txt,
    write_colmap_images_txt,
    write_pointcloud_ply,
)

# ---------------------------------------------------------------------------
# Custom type handles
# ---------------------------------------------------------------------------
LoGeRModel = io.Custom("LOGER_MODEL")
LoGeROutput = io.Custom("LOGER_OUTPUT")

# ---------------------------------------------------------------------------
# Module-level model cache  {variant::base_path -> model}
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict = {}


# ===========================================================================
# Node 1 — LoGeRModelLoader
# ===========================================================================

class LoGeRModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoGeRModelLoader",
            display_name="LoGeR Model (Down)Loader",
            category="LoGeR",
            description="Load a LoGeR or LoGeR_star model from local weights.",
            inputs=[
                io.Combo.Input(
                    "model_variant",
                    options=["LoGeR", "LoGeR_star"],
                    default="LoGeR_star",
                ),
                io.String.Input(
                    "ckpts_base_path",
                    default=_DEFAULT_CKPTS_PATH,
                    multiline=False,
                ),
            ],
            outputs=[
                LoGeRModel.Output("LOGER_MODEL"),
            ],
        )

    @classmethod
    def _download_if_missing(cls, ckpts_base_path: str, model_variant: str):
        """Download config + weights from HuggingFace if not present locally."""
        config_path  = os.path.join(ckpts_base_path, model_variant, "original_config.yaml")
        weights_path = os.path.join(ckpts_base_path, model_variant, "latest.pt")
        files_needed = []
        if not os.path.isfile(config_path):
            files_needed.append(f"{model_variant}/original_config.yaml")
        if not os.path.isfile(weights_path):
            files_needed.append(f"{model_variant}/latest.pt")
        if not files_needed:
            return  # nothing to download

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "[LoGeR] huggingface_hub is required for automatic download. "
                "Run: pip install huggingface_hub"
            )

        os.makedirs(os.path.join(ckpts_base_path, model_variant), exist_ok=True)
        for filename in files_needed:
            print(f"[LoGeR] Downloading {filename} from Junyi42/LoGeR ...")
            hf_hub_download(
                repo_id="Junyi42/LoGeR",
                filename=filename,
                local_dir=ckpts_base_path,
            )
            print(f"[LoGeR] Downloaded: {filename}")

    @classmethod
    def execute(cls, model_variant: str, ckpts_base_path: str):
        cache_key = f"{model_variant}::{ckpts_base_path}"
        if cache_key in _MODEL_CACHE:
            print(f"[LoGeR] Using cached model for {cache_key}")
            return io.NodeOutput(_MODEL_CACHE[cache_key])

        config_path  = os.path.join(ckpts_base_path, model_variant, "original_config.yaml")
        weights_path = os.path.join(ckpts_base_path, model_variant, "latest.pt")

        cls._download_if_missing(ckpts_base_path, model_variant)

        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config not found after download attempt: {config_path}")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Weights not found after download attempt: {weights_path}")

        print(f"[LoGeR] Loading config from: {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Lazy import (only after sys.path is set)
        from loger.models.pi3 import Pi3

        model_config = config.get("model", {})
        pi3_signature = inspect.signature(Pi3.__init__)
        valid_kwargs = {
            name
            for name, param in pi3_signature.parameters.items()
            if name not in {"self", "args", "kwargs"}
            and param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }

        def _maybe_parse_sequence(value):
            if isinstance(value, str):
                stripped = value.strip()
                if stripped.startswith("[") and stripped.endswith("]"):
                    try:
                        parsed = yaml.safe_load(stripped)
                        if isinstance(parsed, (list, tuple)):
                            return list(parsed)
                    except Exception:
                        pass
            return value

        model_kwargs = {}
        for key in sorted(valid_kwargs):
            if key in model_config:
                value = model_config[key]
                if key in {"ttt_insert_after", "attn_insert_after"}:
                    value = _maybe_parse_sequence(value)
                model_kwargs[key] = value

        print(f"[LoGeR] Model kwargs: {model_kwargs}")
        model = Pi3(**model_kwargs)

        print(f"[LoGeR] Loading weights from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k[7:] if k.startswith("module.") else k] = v

        model.load_state_dict(new_state_dict, strict=True)
        model.eval()
        print(f"[LoGeR] Model loaded successfully.")

        result = {"model": model, "config": config, "variant": model_variant}
        _MODEL_CACHE[cache_key] = result
        return io.NodeOutput(result)


# ===========================================================================
# Node 2 — LoGeRInference
# ===========================================================================

class LoGeRInference(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoGeRInference",
            display_name="LoGeR Inference",
            category="LoGeR",
            description="Run LoGeR inference on a batch of images (treated as a video sequence).",
            inputs=[
                LoGeRModel.Input("loger_model"),
                io.Image.Input("images"),
                io.Int.Input("window_size", default=64, min=-1, max=512),
                io.Int.Input("overlap_size", default=3, min=0, max=64),
                io.Boolean.Input("filter_edges", default=True),
                io.Int.Input("target_height", default=280, min=14, max=2048, step=14),
                io.Int.Input("target_width", default=504, min=14, max=2048, step=14),
                io.Boolean.Input("use_sim3", default=False),
            ],
            outputs=[
                LoGeROutput.Output("LOGER_OUTPUT"),
            ],
        )

    @classmethod
    def execute(
        cls,
        loger_model,
        images: torch.Tensor,
        window_size: int,
        overlap_size: int,
        filter_edges: bool,
        target_height: int,
        target_width: int,
        use_sim3: bool,
    ):
        from loger.utils.geometry import depth_edge

        model = loger_model["model"]
        config = loger_model["config"]

        # Snap to multiples of 14
        H = (target_height // 14) * 14
        W = (target_width // 14) * 14

        # images: (B, H, W, C) → (S, C, H, W)
        imgs = resize_images_for_loger(images, H, W)  # (S, C, H, W)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.get_device_capability(device)[0] >= 8
            else torch.float16
        )

        model = model.to(device)
        imgs = imgs.to(device)

        # sim3 and se3 are mutually exclusive; sim3 takes priority when enabled
        se3 = config.get("model", {}).get("se3", False) and not use_sim3

        # ComfyUI disables memory-efficient SDP globally to use its own attention.
        # We temporarily re-enable it for LoGeR: without it, global attention
        # materializes an O(N²) matrix that OOMs at any practical window size.
        _mem_eff_was_enabled = torch.backends.cuda.mem_efficient_sdp_enabled()
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        try:
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype, enabled=torch.cuda.is_available()):
                preds = model(
                    imgs[None],          # (1, S, C, H, W)
                    window_size=window_size,
                    overlap_size=overlap_size,
                    sim3=use_sim3,
                    se3=se3,
                )
        finally:
            torch.backends.cuda.enable_mem_efficient_sdp(_mem_eff_was_enabled)

        # Post-process
        preds["conf"] = torch.sigmoid(preds["conf"])

        if filter_edges and "local_points" in preds:
            edge = depth_edge(preds["local_points"][..., 2], rtol=0.03)
            preds["conf"][edge] = 0.0

        # images: (1, S, C, H, W) → (1, S, H, W, C)
        preds["images"] = imgs[None].permute(0, 1, 3, 4, 2)

        # Move everything to CPU float32
        output = {}
        for k, v in preds.items():
            if torch.is_tensor(v):
                output[k] = v.detach().cpu().float()

        return io.NodeOutput(output)


# ===========================================================================
# Node 3 — LoGeRDepthOutput
# ===========================================================================

class LoGeRDepthOutput(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoGeRDepthOutput",
            display_name="LoGeR Depth Output",
            category="LoGeR",
            description="Extract per-frame depth maps and confidence maps from LoGeR output.",
            inputs=[
                LoGeROutput.Input("loger_output"),
            ],
            outputs=[
                io.Image.Output("depth_maps"),
                io.Image.Output("confidence_maps"),
            ],
        )

    @classmethod
    def execute(cls, loger_output):
        # local_points: (1, S, H, W, 3) — z channel is depth
        local_points = loger_output["local_points"]  # (1, S, H, W, 3)
        conf = loger_output["conf"]                  # (1, S, H, W, 1)

        depth_s_h_w = local_points[0, ..., 2]       # (S, H, W)
        depth_norm = normalize_depth_for_viz(depth_s_h_w)  # (S, H, W, 1)
        depth_rgb = depth_norm.expand(-1, -1, -1, 3)        # (S, H, W, 3)

        conf_s = conf[0]                             # (S, H, W, 1)
        conf_rgb = conf_s.expand(-1, -1, -1, 3)     # (S, H, W, 3)

        return io.NodeOutput(depth_rgb, conf_rgb)


# ===========================================================================
# Node 4 — LoGeRToPointcloud
# ===========================================================================

class LoGeRToPointcloud(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoGeRToPointcloud",
            display_name="LoGeR to Pointcloud",
            category="LoGeR",
            description=(
                "Export a LoGeR reconstruction: pointcloud.ply and optionally "
                "COLMAP cameras.txt / images.txt."
            ),
            inputs=[
                LoGeROutput.Input("loger_output"),
                io.String.Input("filename_prefix", default="loger", multiline=False),
                io.Boolean.Input("save_colmap_txt", default=False),
                io.Float.Input("fov_degrees", default=60.0, min=1.0, max=179.0, step=0.5),
                io.Float.Input("conf_threshold", default=0.1, min=0.0, max=1.0, step=0.01),
                io.Int.Input("downsample", default=1, min=1, max=16),
                io.Float.Input("keep_random_points", default=1.0, min=0.01, max=1.0, step=0.01),
            ],
            outputs=[
                io.String.Output("cameras_txt"),
                io.String.Output("images_txt"),
                io.String.Output("ply_path"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        loger_output,
        filename_prefix: str,
        fov_degrees: float,
        conf_threshold: float,
        downsample: int,
        keep_random_points: float,
        save_colmap_txt: bool,
    ):
        points = loger_output["points"]         # (1, S, H, W, 3)
        conf = loger_output["conf"]             # (1, S, H, W, 1)
        images_t = loger_output["images"]       # (1, S, H, W, 3)
        camera_poses = loger_output["camera_poses"]  # (1, S, 4, 4)

        # Remove batch dim
        points_s = points[0]         # (S, H, W, 3)
        conf_s = conf[0]             # (S, H, W, 1)
        images_s = images_t[0]       # (S, H, W, 3)
        poses_s = camera_poses[0]    # (S, 4, 4)

        S, H, W, _ = points_s.shape

        # Compute intrinsics from assumed FOV
        fx, fy, cx, cy = compute_intrinsics(H, W, fov_degrees)

        # Resolve output directory: ComfyUI/output/<filename_prefix>
        rel_path = filename_prefix.replace("\\", "/").strip("/")
        output_dir = os.path.join(folder_paths.get_output_directory(), *rel_path.split("/")) if rel_path else folder_paths.get_output_directory()

        # Write PLY (always)
        os.makedirs(output_dir, exist_ok=True)
        ply_path = os.path.join(output_dir, "pointcloud.ply")
        write_pointcloud_ply(ply_path, points_s, images_s, conf_s, conf_threshold,
                             downsample=downsample, keep_random_points=keep_random_points)
        ply_path_abs = os.path.abspath(ply_path)

        cameras_content = ""
        images_content = ""

        if save_colmap_txt:
            # Save frame JPEGs
            images_dir = os.path.join(output_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            image_names = []
            for i in range(S):
                name = f"frame_{i:06d}.jpg"
                image_names.append(name)
                frame_np = (images_s[i].numpy() * 255).clip(0, 255).astype(np.uint8)
                PILImage.fromarray(frame_np).save(os.path.join(images_dir, name))

            cameras_content = write_colmap_cameras_txt(
                os.path.join(output_dir, "cameras.txt"), H, W, fx, fy, cx, cy)
            images_content = write_colmap_images_txt(
                os.path.join(output_dir, "images.txt"), poses_s.numpy(), image_names)

        print(f"[LoGeR] Pointcloud written to: {output_dir}")
        return io.NodeOutput(cameras_content, images_content, ply_path_abs)


# ===========================================================================
# Node 5 — LoGeRToHoudiniScript
# ===========================================================================

class LoGeRToHoudiniScript(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="LoGeRToHoudiniScript",
            display_name="LoGeR to Houdini Script",
            category="LoGeR",
            description=(
                "Generate a Houdini Python script that imports LoGeR camera poses "
                "as keyframes, aligned to the exported pointcloud.ply."
            ),
            inputs=[
                LoGeROutput.Input("loger_output"),
                io.String.Input("filename_prefix",   default="loger",                       multiline=False),
                io.String.Input("script_filename",   default="loger_camera.txt",            multiline=False),
                io.Float.Input("fov_degrees",        default=60.0, min=1.0,   max=179.0,   step=0.5),
                io.Float.Input("scene_scale",        default=1.0,  min=0.001, max=1000.0,  step=0.1),
                io.Int.Input(  "start_frame_offset", default=0,    min=0,     max=100000),
            ],
            outputs=[
                io.String.Output("script_path"),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        loger_output,
        fov_degrees: float,
        filename_prefix: str,
        script_filename: str,
        scene_scale: float,
        start_frame_offset: int,
    ):
        # --- Extract data from LOGER_OUTPUT ---
        camera_poses = loger_output["camera_poses"]  # (1, S, 4, 4)  Twc, camera-to-world
        local_points = loger_output["local_points"]  # (1, S, H, W, 3)

        poses_s = camera_poses[0]                    # (S, 4, 4)
        S = poses_s.shape[0]
        H = local_points.shape[2]
        W = local_points.shape[3]

        fx, _, _, _ = compute_intrinsics(H, W, fov_degrees)

        # Convert Twc tensors to plain Python lists for embedding in the script
        frames_py = [
            {"frame_idx": i, "Twc": poses_s[i].numpy().tolist()}
            for i in range(S)
        ]
        intrinsics_py = {"W": W, "H": H, "fx": float(fx)}

        # --- Resolve output directory ---
        rel_path = filename_prefix.replace("\\", "/").strip("/")
        output_dir = (
            os.path.join(folder_paths.get_output_directory(), *rel_path.split("/"))
            if rel_path else folder_paths.get_output_directory()
        )
        os.makedirs(output_dir, exist_ok=True)

        # --- Generate Houdini Python script ---
        #
        # Math: LoGeR outputs Twc (camera-to-world, column-vector, OpenCV convention:
        #   +X right, +Y down, +Z forward / depth).
        # Houdini uses row-vector matrices and its own convention (+Y up, -Z forward).
        #
        # To place the camera correctly relative to the raw PLY (which is in LoGeR world
        # space), we only adjust the camera's LOCAL axes — we do NOT transform world space.
        #
        #   C2W_houdini_row = flip_local * Twc^T
        #
        # where flip_local = diag(1, -1, -1, 1) flips the camera's Y and Z local axes:
        #   - Y: OpenCV down  →  Houdini up
        #   - Z: OpenCV +Z fwd  →  Houdini -Z fwd (Houdini cameras aim at -Z)
        #
        # In Houdini code: final_m = flip_local * m_raw.transposed()
        # (m_raw holds Twc stored row-by-row, so m_raw.transposed() = Twc^T)

        script_content = f"""\
import hou

# -----------------------------------------------------------------------
# LoGeR Camera Import  —  generated by ComfyUI-loger
# Paste this script into Houdini's Python Script editor and run it.
# Import the matching pointcloud.ply without any axis transform.
# -----------------------------------------------------------------------

SCENE_SCALE        = {scene_scale}
START_FRAME_OFFSET = {start_frame_offset}
APERTURE_MM        = 36.0   # 35 mm full-frame equivalent film width (mm)

intrinsics  = {repr(intrinsics_py)}
frames_data = {repr(frames_py)}

W        = intrinsics["W"]
H        = intrinsics["H"]
focal_mm = (intrinsics["fx"] / W) * APERTURE_MM

# Flip camera-local Y and Z axes: OpenCV (+Y down, +Z fwd) -> Houdini (+Y up, -Z fwd).
# Applied as a left-multiply to Twc^T so only camera axes are adjusted;
# the world / PLY coordinate space is left unchanged.
flip_local = hou.Matrix4((
    (1,  0,  0, 0),
    (0, -1,  0, 0),
    (0,  0, -1, 0),
    (0,  0,  0, 1),
))

obj      = hou.node("/obj")
cam_name = "LoGeR_Camera"
cam      = obj.node(cam_name)
if not cam:
    cam = obj.createNode("cam", cam_name)

cam.parm("resx").set(W)
cam.parm("resy").set(H)
cam.parm("aperture").set(APERTURE_MM)


def set_key(parm_name, value, frame):
    k = hou.Keyframe()
    k.setValue(value)
    k.setFrame(frame)
    cam.parm(parm_name).setKeyframe(k)


for fd in frames_data:
    Twc       = fd["Twc"]   # 4x4 camera-to-world (column-vector, OpenCV)
    frame_num = fd["frame_idx"] + START_FRAME_OFFSET  # 0-based, matches PLY view_id

    # Store Twc row-by-row into a hou.Matrix4; transposing gives Twc^T in row-vector form
    m_raw   = hou.Matrix4(Twc)
    final_m = flip_local * m_raw.transposed()   # C2W in Houdini row-vector space

    tr  = final_m.extractTranslates()
    rot = final_m.extractRotates()

    set_key("tx",    tr[0]  * SCENE_SCALE, frame_num)
    set_key("ty",    tr[1]  * SCENE_SCALE, frame_num)
    set_key("tz",    tr[2]  * SCENE_SCALE, frame_num)
    set_key("rx",    rot[0],               frame_num)
    set_key("ry",    rot[1],               frame_num)
    set_key("rz",    rot[2],               frame_num)
    set_key("focal", focal_mm,             frame_num)

print(f"LoGeR camera import done: {{len(frames_data)}} frames -> {{cam_name}}")
"""

        # --- Save ---
        safe_filename = script_filename.strip() or "loger_houdini.py"
        out_path = os.path.join(output_dir, safe_filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(script_content)

        print(f"[LoGeR] Houdini script written to: {out_path}")
        return io.NodeOutput(out_path)
