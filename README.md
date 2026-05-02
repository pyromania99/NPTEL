# COLMAP Pipeline Setup

This directory contains `colmap_pipeline.py`, a ROS 2 node and COLMAP runner that records PX4 odometry-aligned images, writes COLMAP priors, and can produce a fused dense point cloud at `dense/fused.ply`.

## What the pipeline needs

### System requirements

- Ubuntu 22.04
- Python 3.10
- ROS 2 Humble with a sourced workspace that provides:
  - `rclpy`
  - `sensor_msgs`
  - `px4_msgs`
  - `cv_bridge`
- COLMAP installed with CUDA support if you want GPU acceleration
- An NVIDIA driver and CUDA-capable GPU for the CUDA build

### Python packages

Install the runtime Python dependencies used by the script:

```bash
python3 -m pip install "numpy<2" opencv-python
```

If your ROS environment does not already provide the message packages above, source the appropriate ROS and PX4 workspaces before launching the script.

## CUDA COLMAP setup

The pipeline looks for COLMAP in this order:

1. `COLMAP_BIN`
2. a local CUDA build at `../../colmap_cuda_src/build_cuda/src/colmap/exe/colmap`
3. `/opt/colmap/bin/colmap`
4. `/usr/local/bin/colmap`
5. `/usr/bin/colmap`
6. `colmap` on `PATH`

For a CUDA build, set `COLMAP_BIN` explicitly if you want to avoid ambiguity:

```bash
export COLMAP_BIN=/absolute/path/to/colmap
```

The local build is expected to be compiled with CUDA enabled. If COLMAP was built without CUDA, the script will fall back to CPU mode automatically for feature extraction and matching.

### Recommended build notes

- Build COLMAP from the `colmap_cuda_src` workspace with CUDA enabled.
- Confirm the binary responds to `colmap help` and that the build advertises CUDA support.
- The script probes COLMAP CLI option names and adapts to either the newer or older GPU flag names used by different COLMAP versions.

## Run modes

### 1. Collect a fresh dataset and reconstruct it

This mode subscribes to the live ROS topics, records images with a minimum baseline, writes priors, and then runs COLMAP:

```bash
source /opt/ros/humble/setup.bash
source /path/to/your/px4_workspace/install/setup.bash
python3 launch/colmap_pipeline.py --dataset-dir /path/to/colmap_dataset
```

Press `Ctrl+C` to stop collection. On shutdown the script writes the COLMAP prior model and then launches matching, mapping, and optional dense reconstruction.

### 2. Reconstruct an existing dataset without ROS

Use this when `images/`, `custom_matches.txt`, and `sparse/model/` already exist:

```bash
python3 launch/colmap_pipeline.py --reconstruct-existing --dataset-dir /path/to/colmap_dataset
```

### 3. Enable dense reconstruction

Add `--dense` to run the dense pipeline after sparse mapping:

```bash
python3 launch/colmap_pipeline.py --reconstruct-existing --dense --dataset-dir /path/to/colmap_dataset
```

Dense modes are available with `--dense-mode`:

- `normal`
- `low-memory`
- `ultra-low-memory`
- `fusion-only`

`fusion-only` reuses existing stereo depth and normal maps and only reruns `stereo_fusion`.

### 4. Dense only on an existing sparse model

If you already have a sparse reconstruction in `sparse/reconstruction`, rerun only the dense stage:

```bash
python3 launch/colmap_pipeline.py --dense-only --dataset-dir /path/to/colmap_dataset
```

## Output layout

The dataset root defaults to `colmap_dataset/` and ends up with this structure:

```text
colmap_dataset/
  images/                  # captured camera frames
  custom_matches.txt       # spatial pair list used by COLMAP
  db.db                    # COLMAP database
  sparse/
    model/                 # PX4 prior cameras.txt / images.txt / points3D.txt
    reconstruction/        # mapper output
    snapshots/             # mapper snapshots for inspection
  dense/
    fused.ply              # final dense point cloud
```

The sparse prior model is written to `sparse/model/`, while mapper output is written to `sparse/reconstruction/`. The dense point cloud is saved as `dense/fused.ply`.

## Inputs expected from ROS

The live collector subscribes to:

- `/mono_cam/image_raw`
- `/mono_cam/camera_info`
- `/fmu/out/vehicle_odometry`

The collector stores images only when the motion baseline exceeds the configured threshold, which keeps the dataset from filling up with nearly identical frames.

## Common issues

- If COLMAP fails to start, verify `COLMAP_BIN` points to an executable binary.
- If the script reports that ROS packages are missing, source the ROS and PX4 environment before running it.
- If dense reconstruction fails due to memory pressure, retry with `--dense-mode low-memory` or `--dense-mode ultra-low-memory`.
- If you want to inspect the sparse result, open COLMAP GUI with the generated `db.db` and `images/` directory.
