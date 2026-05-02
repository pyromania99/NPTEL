# COLMAP Pipeline Repro Guide

This guide keeps the COLMAP dataset folder clean and makes pipeline runs reproducible in host or Docker setups.

## 1) Recommended dataset layout

Use a single dataset root (default: `colmap_dataset`):

- `images/` (input images)
- `custom_matches.txt` (pair list)
- `sparse/model/` (PX4 prior model: `cameras.txt`, `images.txt`, `points3D.txt`)

Generated artifacts (safe to regenerate):

- `db.db`
- `dense/`
- `sparse/0/`
- `sparse/triangulated/`

## 2) Clean before reruns

Preserve source images and priors:

```bash
./launch/cleanup_colmap_dataset.sh --dataset-dir /data/colmap_dataset
```

Full reset including source images and priors:

```bash
./launch/cleanup_colmap_dataset.sh --dataset-dir /data/colmap_dataset --full
```

## 3) Portable COLMAP binary selection

`launch/colmap_pipeline.py` resolves COLMAP in this order:

1. `COLMAP_BIN` environment variable
2. local build: `../colmap_cuda_src/build_cuda/src/colmap/exe/colmap`
3. `/opt/colmap/bin/colmap`
4. `/usr/local/bin/colmap`
5. `/usr/bin/colmap`
6. `colmap` from `PATH`

For Docker, set `COLMAP_BIN` explicitly to remove ambiguity.

## 6) New features (summary)

This pipeline has been extended to improve robustness and observability when running COLMAP. Highlights:

- **Mapper snapshotting:** periodic mapper snapshots are supported and written to `sparse/snapshots` so you can inspect BA progress and intermediate models.
- **GPU flags detection:** the script probes COLMAP subcommand help to pick the correct GPU-related flags for feature extraction and matching across different builds.
- **Camera prior compatibility:** prior camera models from `sparse/model` are applied to image reading to prevent camera-model mismatch crashes.
- **Memory-tuned dense reconstruction:** select `normal`, `low-memory`, or `ultra-low-memory` dense modes and the pipeline will retry with lower-memory settings on failures.
- **Rerun controls:** `--dense-only` will re-run only the dense stage; `--reconstruct-existing` avoids re-running mapping when a sparse model already exists.
- **Fusion-only rerun:** `--dense-only --dense-mode fusion-only` runs only `stereo_fusion` and reuses existing `dense/stereo` depth/normal maps (no undistorter or patch-match rewrite).
- **Samples export:** small visual exports of features and example matches are written to `samples/` for quick review.

Refer to `launch/colmap_pipeline.py` CLI help for up-to-date flag names and usage.

## 4) Example Docker run

```bash
docker run --rm -it \
  --gpus all \
  -v "$PWD":/workspace \
  -v "$PWD/colmap_dataset":/data/colmap_dataset \
  -e COLMAP_BIN=/opt/colmap/bin/colmap \
  your-image:tag \
  bash -lc "cd /workspace/PX4-Autopilot && ./launch/cleanup_colmap_dataset.sh --dataset-dir /data/colmap_dataset && python3 launch/colmap_pipeline.py --reconstruct-existing --dataset-dir /data/colmap_dataset"
```

## 5) Repro tips

- Pin your Docker image tag and COLMAP version/commit.
- Keep only one active dataset root per experiment run.
- Archive `images/`, `custom_matches.txt`, and `sparse/model/` as immutable inputs.
