# ComfyUI-loger

A ComfyUI custom node pack for running **LoGeR** (Long-sequence Generalizable Reconstruction) inference directly inside ComfyUI workflows. LoGeR estimates per-frame depth maps, confidence maps, and full 3-D point clouds from a sequence of images (treated as a video).

---

## Nodes

| Node | Description |
|---|---|
| **LoGeR Model (Down)Loader** | Loads `LoGeR` or `LoGeR_star` weights. Downloads automatically from HuggingFace (`Junyi42/LoGeR`) if not present locally. |
| **LoGeR Inference** | Runs the reconstruction model on an image batch and returns a `LOGER_OUTPUT` bundle. |
| **LoGeR Depth Output** | Extracts per-frame depth maps and confidence maps as `IMAGE` tensors. |
| **LoGeR to Pointcloud** | Exports the reconstruction as a `pointcloud.ply` and optionally COLMAP `cameras.txt` / `images.txt`. |
| **LoGeR to Houdini Script** | Generates a Houdini Python script that imports camera poses as keyframes, aligned to the exported PLY. |

---

## Requirements

- ComfyUI with the V3 node API (`comfy_api.latest`)
- Python packages: see `requirements.txt`
  - `plyfile`
  - `huggingface_hub` (for automatic weight download)
  - Standard ML stack: `torch`, `numpy`, `Pillow`, `pyyaml`

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Installation

1. Clone this repository into your `ComfyUI/custom_nodes/` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/ComfyUI-loger
```

2. Restart ComfyUI. The LoGeR nodes will appear in the **LoGeR** category.

---

## Model Weights

Weights are downloaded automatically on first use from [Junyi42/LoGeR](https://huggingface.co/Junyi42/LoGeR) via `huggingface_hub`. They are stored under `ComfyUI/models/LoGeR/` by default. You can override the path in the **LoGeR Model (Down)Loader** node.

Two variants are available:

- `LoGeR` — base model
- `LoGeR_star` — improved variant (default)

---

## Typical Workflow

```
Load Images → LoGeR Model Loader → LoGeR Inference → LoGeR Depth Output
                                                    → LoGeR to Pointcloud
                                                    → LoGeR to Houdini Script
```

### LoGeR Inference parameters

| Parameter | Default | Description |
|---|---|---|
| `window_size` | 64 | Sliding-window size (-1 = full sequence) |
| `overlap_size` | 3 | Frame overlap between windows |
| `filter_edges` | true | Zero out depth-edge confidence |
| `target_height` | 280 | Inference height (snapped to multiples of 14) |
| `target_width` | 504 | Inference width (snapped to multiples of 14) |
| `use_sim3` | false | Use Sim3 alignment instead of SE3 |

### LoGeR to Pointcloud parameters

| Parameter | Default | Description |
|---|---|---|
| `filename_prefix` | `loger` | Output sub-folder inside `ComfyUI/output/` |
| `save_colmap_txt` | false | Also write COLMAP cameras.txt / images.txt |
| `fov_degrees` | 60.0 | Assumed horizontal FOV for intrinsics |
| `conf_threshold` | 0.1 | Minimum confidence to include a point |
| `downsample` | 1 | Spatial grid downsampling factor |
| `keep_random_points` | 1.0 | Random fraction of points to keep |

---

## Houdini Integration

The **LoGeR to Houdini Script** node writes a self-contained Python script. Open Houdini's **Python Script** editor and run it to create a `LoGeR_Camera` object with per-frame keyframes. Import `pointcloud.ply` without any axis transform — the script handles the OpenCV → Houdini convention conversion internally.

---

## Credits

- LoGeR model: [Junyi42/LoGeR](https://huggingface.co/Junyi42/LoGeR)
- ComfyUI custom node integration: this repository

---

## License

MIT — see [LICENSE](LICENSE).
