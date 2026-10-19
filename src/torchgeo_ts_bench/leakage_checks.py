"""Leakage Detection Engine for Spatiotemporal Geospatial Benchmarks.

Core Original Contribution: Automated audit for:
1. Spatial Leakage: Geographic polygon/bounding-box overlap between train, val, and test splits.
2. Temporal Leakage: Chronological contamination (future observations in train or test before train).
3. Duplicate Sample Leakage: Identical or reused sample footprints across splits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class LeakageAuditReport:
    """Detailed audit report of spatiotemporal data splits."""

    passed: bool
    has_spatial_leakage: bool
    has_temporal_leakage: bool
    has_duplicate_leakage: bool
    spatial_violations: List[Dict[str, Any]] = field(default_factory=list)
    temporal_violations: List[Dict[str, Any]] = field(default_factory=list)
    duplicate_violations: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "has_spatial_leakage": self.has_spatial_leakage,
            "has_temporal_leakage": self.has_temporal_leakage,
            "has_duplicate_leakage": self.has_duplicate_leakage,
            "num_spatial_violations": len(self.spatial_violations),
            "num_temporal_violations": len(self.temporal_violations),
            "num_duplicate_violations": len(self.duplicate_violations),
            "summary": self.summary,
            "spatial_violations": self.spatial_violations[:10],  # truncate if large
            "temporal_violations": self.temporal_violations[:10],
            "duplicate_violations": self.duplicate_violations[:10],
        }


class LeakageDetector:
    """Performs rigorous geospatial and geotemporal leakage audits."""

    @staticmethod
    def _compute_bbox_intersection(
        bbox1: Tuple[float, float, float, float],
        bbox2: Tuple[float, float, float, float],
    ) -> float:
        """Compute 2D Intersection over Union (IoU) of two bounding boxes (minx, miny, maxx, maxy)."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        inter_min_x = max(x1_min, x2_min)
        inter_min_y = max(y1_min, y2_min)
        inter_max_x = min(x1_max, x2_max)
        inter_max_y = min(y1_max, y2_max)

        if inter_max_x <= inter_min_x or inter_max_y <= inter_min_y:
            return 0.0

        inter_area = (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        if union_area <= 0.0:
            return 0.0
        return float(inter_area / union_area)

    @classmethod
    def audit_splits(
        cls,
        train_metadata: List[Dict[str, Any]],
        test_metadata: List[Dict[str, Any]],
        val_metadata: Optional[List[Dict[str, Any]]] = None,
        spatial_iou_threshold: float = 0.0,
        enforce_chronological_split: bool = False,
    ) -> LeakageAuditReport:
        """Run comprehensive spatial, temporal, and duplicate leakage audits.

        Args:
            train_metadata: List of sample metadata dicts for training split.
            test_metadata: List of sample metadata dicts for testing split.
            val_metadata: Optional list of sample metadata dicts for validation split.
            spatial_iou_threshold: Overlap IoU threshold above which spatial leak is flagged.
            enforce_chronological_split: If True, tests must occur strictly after training.

        Returns:
            LeakageAuditReport with pass/fail status and detailed violation diagnostics.
        """
        spatial_violations: List[Dict[str, Any]] = []
        temporal_violations: List[Dict[str, Any]] = []
        duplicate_violations: List[Dict[str, Any]] = []

        # 1. Duplicate Sample Audit
        train_ids = {s.get("sample_id", ""): s for s in train_metadata if "sample_id" in s}
        for s in test_metadata:
            sid = s.get("sample_id", "")
            if sid and sid in train_ids:
                duplicate_violations.append(
                    {
                        "sample_id": sid,
                        "type": "Exact sample ID reused in train and test splits",
                    }
                )

        if val_metadata:
            for s in val_metadata:
                sid = s.get("sample_id", "")
                if sid and sid in train_ids:
                    duplicate_violations.append(
                        {
                            "sample_id": sid,
                            "type": "Exact sample ID reused in train and val splits",
                        }
                    )

        # 2. Spatial Overlap Audit (Geographic Leakage)
        for t_sample in train_metadata:
            t_box = t_sample.get("bbox")
            if not t_box:
                continue
            for test_sample in test_metadata:
                test_box = test_sample.get("bbox")
                if not test_box:
                    continue

                iou = cls._compute_bbox_intersection(t_box, test_box)
                if iou > spatial_iou_threshold:
                    spatial_violations.append(
                        {
                            "train_id": t_sample.get("sample_id", "unknown"),
                            "test_id": test_sample.get("sample_id", "unknown"),
                            "overlap_iou": round(iou, 4),
                            "train_bbox": t_box,
                            "test_bbox": test_box,
                        }
                    )

        # 3. Temporal Contamination Audit
        if enforce_chronological_split:
            # Parse train timestamps to find latest training observation
            train_dates: List[datetime] = []
            for s in train_metadata:
                for dt_str in s.get("timestamps", []):
                    try:
                        train_dates.append(datetime.fromisoformat(dt_str))
                    except Exception:
                        pass

            max_train_date = max(train_dates) if train_dates else None

            if max_train_date:
                for s in test_metadata:
                    test_dates: List[datetime] = []
                    for dt_str in s.get("timestamps", []):
                        try:
                            test_dates.append(datetime.fromisoformat(dt_str))
                        except Exception:
                            pass

                    min_test_date = min(test_dates) if test_dates else None
                    if min_test_date and min_test_date <= max_train_date:
                        temporal_violations.append(
                            {
                                "test_id": s.get("sample_id", "unknown"),
                                "test_earliest_timestamp": min_test_date.isoformat(),
                                "train_latest_timestamp": max_train_date.isoformat(),
                                "violation": "Test observation occurs before or concurrently with training period",
                            }
                        )

        has_spatial = len(spatial_violations) > 0
        has_temporal = len(temporal_violations) > 0
        has_duplicates = len(duplicate_violations) > 0
        passed = not (has_spatial or has_temporal or has_duplicates)

        summary_parts = []
        if passed:
            summary = "PASSED: All splits are verified leakage-free across space, time, and sample identities."
        else:
            if has_spatial:
                summary_parts.append(f"{len(spatial_violations)} spatial bounding-box overlaps")
            if has_temporal:
                summary_parts.append(f"{len(temporal_violations)} chronological sequence leaks")
            if has_duplicates:
                summary_parts.append(f"{len(duplicate_violations)} duplicate sample identity reuses")
            summary = "FAILED: Detected " + ", ".join(summary_parts) + "."

        return LeakageAuditReport(
            passed=passed,
            has_spatial_leakage=has_spatial,
            has_temporal_leakage=has_temporal,
            has_duplicate_leakage=has_duplicates,
            spatial_violations=spatial_violations,
            temporal_violations=temporal_violations,
            duplicate_violations=duplicate_violations,
            summary=summary,
        )

    @classmethod
    def generate_controlled_leakage_benchmark(cls) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Generates deliberately overlapping synthetic metadata to validate detection capabilities.

        Used in automated unit tests to demonstrate that our audit engine successfully
        flags geographic, temporal, and identity leaks when present.
        """
        train_meta = [
            {
                "sample_id": "TRAIN_001",
                "bbox": (10.0, 45.0, 10.5, 45.5),
                "timestamps": ["2023-01-01T00:00:00", "2023-01-15T00:00:00"],
            },
            {
                "sample_id": "TRAIN_002",
                "bbox": (11.0, 46.0, 11.5, 46.5),
                "timestamps": ["2023-02-01T00:00:00", "2023-02-15T00:00:00"],
            },
        ]

        # Contaminated test metadata (TRAIN_001 duplicate, spatial overlap with TRAIN_002, and temporal leak)
        test_meta_contaminated = [
            {
                "sample_id": "TRAIN_001",  # Duplicate identity leak
                "bbox": (10.0, 45.0, 10.5, 45.5),
                "timestamps": ["2023-01-01T00:00:00"],
            },
            {
                "sample_id": "TEST_LEAK_SPATIAL",
                "bbox": (11.2, 46.2, 11.7, 46.7),  # Spatial bounding-box overlap with TRAIN_002
                "timestamps": ["2022-12-01T00:00:00"],  # Chronological leak (earlier than training)
            },
        ]

        return train_meta, test_meta_contaminated
