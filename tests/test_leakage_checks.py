"""Unit tests for the Leakage Detection Engine.

Includes the controlled software test with deliberately overlapping sample metadata
to mathematically verify that the checking module detects spatial, temporal,
and duplicate leaks as required by Section 3A.
"""

from torchgeo_ts_bench.leakage_checks import LeakageDetector


def test_clean_splits_pass_leakage_audit():
    """Verify that completely separated spatial and temporal splits pass without violations."""
    train_meta = [
        {
            "sample_id": "TRAIN_01",
            "bbox": (10.0, 45.0, 10.1, 45.1),
            "timestamps": ["2023-01-01T00:00:00", "2023-01-15T00:00:00"],
        },
        {
            "sample_id": "TRAIN_02",
            "bbox": (10.2, 45.0, 10.3, 45.1),
            "timestamps": ["2023-01-01T00:00:00", "2023-01-15T00:00:00"],
        },
    ]

    test_meta = [
        {
            "sample_id": "TEST_01",
            "bbox": (11.0, 46.0, 11.1, 46.1),  # Completely different geographic region
            "timestamps": ["2023-06-01T00:00:00", "2023-06-15T00:00:00"],  # Later time period
        }
    ]

    report = LeakageDetector.audit_splits(
        train_metadata=train_meta,
        test_metadata=test_meta,
        enforce_chronological_split=True,
    )

    assert report.passed is True
    assert report.has_spatial_leakage is False
    assert report.has_temporal_leakage is False
    assert report.has_duplicate_leakage is False
    assert "PASSED" in report.summary


def test_controlled_leakage_detection_novelty_proof():
    """CONTROLLED SOFTWARE TEST: Deliberately injected leaks to prove detection capability.

    Label: Controlled software test (not real-world model performance).
    Validates detection of:
    1. Geographic overlap between train and test bounding boxes.
    2. Chronological sequence contamination (test observations occurring before train).
    3. Reused sample identity (duplicate sample ID in train and test).
    """
    train_meta, test_meta_contaminated = LeakageDetector.generate_controlled_leakage_benchmark()

    report = LeakageDetector.audit_splits(
        train_metadata=train_meta,
        test_metadata=test_meta_contaminated,
        enforce_chronological_split=True,
    )

    # Must fail because leaks were deliberately injected
    assert report.passed is False
    assert report.has_duplicate_leakage is True, "Failed to detect duplicate sample identity reuse!"
    assert report.has_spatial_leakage is True, "Failed to detect geographic bounding box overlap!"
    assert report.has_temporal_leakage is True, "Failed to detect chronological sequence contamination!"

    assert len(report.duplicate_violations) >= 1
    assert len(report.spatial_violations) >= 1
    assert len(report.temporal_violations) >= 1
    assert "FAILED" in report.summary
