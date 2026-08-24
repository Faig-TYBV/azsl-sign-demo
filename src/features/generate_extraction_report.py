"""
Final feature-extraction report for AzSLD_Words_200.

Scans all videos vs. per-video .npz feature files under data/features/full/,
parses the append-only extraction_metadata.jsonl, and writes
outputs/reports/full_extraction_report.json (+ human-readable .txt).

Usage:
    python src/features/generate_extraction_report.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.run_full_extraction import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_DIR,
    collect_videos,
    npz_is_valid,
)

REPORT_DIR = Path("outputs/reports")


def main() -> None:
    t0 = time.time()
    videos = collect_videos(DEFAULT_DATA_ROOT)
    total = len(videos)

    meta_lines = []
    meta_path = DEFAULT_OUTPUT_DIR / "extraction_metadata.jsonl"
    if meta_path.is_file():
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        meta_lines.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    failed_meta = [m for m in meta_lines if m.get("status") == "failed"]
    ok_meta = [m for m in meta_lines if m.get("status") != "failed"]

    valid_npz, invalid_npz = [], []
    for vp in videos:
        out_path = DEFAULT_OUTPUT_DIR / vp.parent.name / f"{vp.stem}.npz"
        (valid_npz if npz_is_valid(out_path) else invalid_npz).append(str(vp))

    detection_rates = [m["detection_rate"] for m in ok_meta if "detection_rate" in m]
    proc_times = [m["processing_time_sec"] for m in ok_meta if "processing_time_sec" in m]

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_videos": total,
        "successfully_extracted": len(valid_npz),
        "skipped_existing_valid_files": len(valid_npz),
        "failed_videos_count": len(failed_meta),
        "final_feature_count": len(valid_npz),
        "invalid_or_missing_features_remaining": len(invalid_npz) - len(failed_meta),
        "feature_shape_expected": [26, 126],
        "avg_detection_rate": round(float(np.mean(detection_rates)), 4) if detection_rates else None,
        "min_detection_rate": round(float(np.min(detection_rates)), 4) if detection_rates else None,
        "avg_processing_time_sec_per_video": round(float(np.mean(proc_times)), 3) if proc_times else None,
        "total_processing_time_sec": round(sum(proc_times), 1),
        "report_generation_time_sec": round(time.time() - t0, 2),
        "failure_list": [
            {"video_path": m.get("video_path"), "error": m.get("error")}
            for m in failed_meta
        ],
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "full_extraction_report.json"
    txt_path = REPORT_DIR / "full_extraction_report.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        "=" * 60,
        "  FULL FEATURE EXTRACTION REPORT — AzSLD_Words_200",
        "=" * 60,
        f"Generated:                 {report['generated_at']}",
        f"Total videos:              {report['total_videos']}",
        f"Successfully extracted:    {report['successfully_extracted']}",
        f"Skipped existing valid:    {report['skipped_existing_valid_files']}",
        f"Failed videos:             {report['failed_videos_count']}",
        f"Final feature count:       {report['final_feature_count']}",
        f"Feature shape:             {tuple(report['feature_shape_expected'])}",
        f"Avg detection rate:        {report['avg_detection_rate']}",
        f"Min detection rate:        {report['min_detection_rate']}",
        f"Avg sec/video:             {report['avg_processing_time_sec_per_video']}",
        f"Total processing time:     {report['total_processing_time_sec']} s "
        f"({report['total_processing_time_sec'] / 3600:.2f} h)",
        "-" * 60,
        "Failure list:",
    ]
    lines += [
        f"  - {m.get('video_path')}: {m.get('error')}" for m in failed_meta
    ] or ["  (none)"]
    lines.append("=" * 60)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
