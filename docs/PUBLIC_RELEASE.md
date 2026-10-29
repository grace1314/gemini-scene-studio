# Public example provenance and privacy

The repository's `examples/desk-scene` dataset is derived from a private local capture, but the source scene and its identifiers are not published.

Before publication:

- People and work screens are strongly pixelated in both rectified views.
- Readable printed card text is pixelated where it does not occlude the reconstruction target.
- Camera serial numbers, frame timestamps, local source paths, and raw device identifiers are removed.
- Image dimensions, camera intrinsics, stereo baseline, and rectified geometry are not changed.
- All 20 public stereo pairs were visually reviewed after redaction.

The public copy was then rerun from scratch. COLMAP registered 20/20 frames with 1.08 px mean reprojection error, followed by RGB-D point-cloud fusion, Open3D TSDF extraction, and the included gsplat baseline. See `examples/desk-scene/results/metrics.json` for the recorded outputs.

Precomputed outputs are usable offline immediately. Reproducing the neural stages requires a one-time download of third-party source and model weights with `scripts/fetch_deps.sh` and `scripts/fetch_models.sh`; these large upstream assets are intentionally not redistributed in Git.

`apps/public_demo.cpp` and `examples/desk-scene/privacy_masks.json` make the redaction process auditable. The tool refuses to overwrite its output. The original private images remain outside the Git repository.
