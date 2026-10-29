#!/usr/bin/env python3
"""Replace COLMAP SIFT matches with ALIKED + LightGlue RGB-only matches."""
from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import torch


def generate_matches(database_path: Path, image_dir: Path, match_file: Path,
                     progress=None) -> dict:
    from lightglue import ALIKED, LightGlue
    from lightglue.utils import load_image, rbd

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("ALIKED + LightGlue requires CUDA in the integrated pipeline")
    torch.cuda.reset_peak_memory_stats()
    extractor = ALIKED(max_num_keypoints=4096).eval().to(device)
    matcher = LightGlue(features="aliked", depth_confidence=-1, width_confidence=-1).eval().to(device)
    paths = sorted(image_dir.glob("*.png"))
    features = {}
    with torch.inference_mode():
        for index, path in enumerate(paths):
            features[path.name] = extractor.extract(load_image(path).to(device))
            print(f"features {path.name}: {features[path.name]['keypoints'].shape[1]}", flush=True)
            if progress:
                progress(0.08 + 0.18 * (index + 1) / len(paths),
                         f"ALIKED 特征 {index + 1}/{len(paths)}")

    database = sqlite3.connect(database_path)
    image_ids = {name: image_id for image_id, name in database.execute("SELECT image_id,name FROM images")}
    for table in ("keypoints", "descriptors", "matches", "two_view_geometries"):
        database.execute(f"DELETE FROM {table}")
    for path in paths:
        keypoints = rbd(features[path.name])["keypoints"].detach().cpu().numpy().astype(np.float32)
        database.execute("INSERT INTO keypoints(image_id,rows,cols,data) VALUES(?,?,?,?)",
                         (image_ids[path.name], len(keypoints), 2, keypoints.tobytes()))
    database.commit()
    database.close()

    total = 0
    pair_count = len(paths) * (len(paths) - 1) // 2
    processed = 0
    written_pairs = 0
    with match_file.open("w") as output, torch.inference_mode():
        for i, first in enumerate(paths):
            for second in paths[i + 1:]:
                prediction = rbd(matcher({"image0": features[first.name],
                                          "image1": features[second.name]}))
                matches = prediction["matches"].detach().cpu().numpy()
                if len(matches) < 15:
                    processed += 1
                    continue
                output.write(f"{first.name} {second.name}\n")
                np.savetxt(output, matches, fmt="%d")
                output.write("\n")
                total += len(matches)
                written_pairs += 1
                processed += 1
                print(f"matches {first.name} {second.name}: {len(matches)}", flush=True)
                if progress and (processed % 10 == 0 or processed == pair_count):
                    progress(0.26 + 0.28 * processed / pair_count,
                             f"LightGlue 匹配 {processed}/{pair_count}")
    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f"wrote {total} raw matches to {match_file}")
    return {"raw_matches": total, "matched_pairs": written_pairs,
            "peak_allocated_mib": peak,
            "features_per_frame": {path.name: int(features[path.name]["keypoints"].shape[1])
                                   for path in paths}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_database", type=Path)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("output_database", type=Path)
    parser.add_argument("match_file", type=Path)
    args = parser.parse_args()
    shutil.copy2(args.source_database, args.output_database)
    generate_matches(args.output_database, args.image_dir, args.match_file)


if __name__ == "__main__":
    main()
