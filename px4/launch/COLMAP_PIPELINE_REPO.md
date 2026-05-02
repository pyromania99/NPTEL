# COLMAP Pipeline Repo Guide

This guide keeps the COLMAP dataset folder clean and makes pipeline runs reproducible in host or Docker setups.

## 1) Recommended dataset layout

Use a single dataset root (default: `colmap_dataset`):

- `images/` (input images)
- `custom_matches.txt` (pair list)
- `sparse/model/` (PX4 prior model: `cameras.txt`, `images.txt`, `points3D.txt`)
- `sparse/reconstruction/` (COLMAP mapper output)

Generated artifacts (safe to regenerate):

- `db.db`
- `dense/`
- `sparse/reconstruction/`

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

This pipeline script has been extended to improve robustness and observability when running COLMAP at scale. Key additions:

- **Mapper snapshotting & BA playback:** the mapper can write periodic snapshot models for step-by-step bundle-adjustment inspection. Use `--Mapper.snapshot_path` and `--Mapper.snapshot_frames_freq` (configured by the script) and inspect snapshots under `sparse/snapshots`.
- **GPU option probing:** COLMAP CLI option names vary between builds (e.g. `--FeatureExtraction.use_gpu` vs `--SiftExtraction.use_gpu`). The script auto-detects supported option names and uses the correct flags for feature extraction and matching.
- **Camera-model alignment:** the script reads prior camera models from `sparse/model` and passes the matching `--ImageReader.camera_model` to COLMAP to avoid SIGABRTs caused by model mismatches.
- **Dense modes & OOM handling:** dense reconstruction supports `normal`, `low-memory`, and `ultra-low-memory` modes. The pipeline will retry with lower-memory configurations on OOM and can skip dense processing if desired.
- **Dense-only reruns & reconstruct-existing:** use `--dense-only` to re-run the dense pipeline without re-running mapping, and `--reconstruct-existing` to run mapper only when an existing sparse reconstruction is present.
- **Fusion-only dense rerun:** use `--dense-only --dense-mode fusion-only` to run only `stereo_fusion` and reuse existing `dense/stereo` depth/normal maps without rewriting photometric/geometric stereo outputs.
- **Feature/match sample export:** the script can export small visual samples of detected features and example match pairs to `samples/features` and `samples/matches` under the dataset dir for quick inspection.

See the CLI reference below for exact flags and behavior notes.

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
