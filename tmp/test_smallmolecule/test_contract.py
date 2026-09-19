from pathlib import Path
import unittest

import numpy as np

from ops.small_molecule_test.build_manifest import (
    _assert_unique_candidate_keys,
    _target_from_intersections,
)
from ops.small_molecule_test.ranking import build_topk_curve


class SmallMoleculeContractTest(unittest.TestCase):
    def test_duplicate_candidate_key_fails_closed(self) -> None:
        duplicate = [
            {"pdb_id": "x", "source_blob_index": 1},
            {"pdb_id": "x", "source_blob_index": 1},
        ]
        with self.assertRaises(ValueError):
            _assert_unique_candidate_keys("duplicate", duplicate)

    def test_unique_target_uses_overlap_then_occurrence_id(self) -> None:
        matched, candidate_coverage, occurrence_coverage, score = _target_from_intersections(
            np.asarray([7, 3, 5]),
            np.asarray([40, 40, 10]),
            100,
            np.asarray([100, 100, 20]),
        )
        self.assertEqual(matched, 3)
        self.assertEqual(candidate_coverage, 0.4)
        self.assertEqual(occurrence_coverage, 0.4)
        self.assertEqual(score, 0.4)

    def test_non_stage3_foreground_is_free_but_background_spends_attempt(self) -> None:
        candidates = [
            {
                "pdb_id": "9rd8",
                "source_blob_index": 0,
                "centered_box_index": 0,
                "candidate_selected": True,
                "gaussian_score": 3.0,
                "target_scope": "non_stage3_foreground",
                "matched_occurrence_id": 0,
                "target_smiles": "C",
            },
            {
                "pdb_id": "9rd8",
                "source_blob_index": 1,
                "centered_box_index": 1,
                "candidate_selected": True,
                "gaussian_score": 2.0,
                "target_scope": "background",
                "matched_occurrence_id": None,
                "target_smiles": None,
            },
            {
                "pdb_id": "9rd8",
                "source_blob_index": 2,
                "centered_box_index": 2,
                "candidate_selected": False,
                "gaussian_score": 9.0,
                "target_scope": "stage3_foreground",
                "matched_occurrence_id": 1,
                "target_smiles": "CC",
            },
        ]
        curve, traces = build_topk_curve(
            "contract",
            Path("frozen.jsonl"),
            ["9rd8"],
            candidates,
            "gaussian_score",
            lambda candidate: candidate["source_blob_index"] == 2,
        )
        self.assertEqual(curve["hit_count"][:2], [0, 1])
        self.assertEqual(
            [record["attempt_index"] for record in traces["9rd8"]], [None, 1, 2]
        )
        self.assertEqual(traces["9rd8"][0]["action"], "skip_non_stage3_foreground")
        self.assertEqual(traces["9rd8"][2]["raw_rank"], 3)

    def test_selected_segment_always_precedes_unselected_segment(self) -> None:
        candidates = [
            {
                "pdb_id": "x",
                "source_blob_index": 8,
                "centered_box_index": 8,
                "candidate_selected": False,
                "gaussian_score": 100.0,
                "target_scope": "background",
                "matched_occurrence_id": None,
                "target_smiles": None,
            },
            {
                "pdb_id": "x",
                "source_blob_index": 2,
                "centered_box_index": 2,
                "candidate_selected": True,
                "gaussian_score": 1.0,
                "target_scope": "background",
                "matched_occurrence_id": None,
                "target_smiles": None,
            },
        ]
        _, traces = build_topk_curve(
            "selected-first",
            Path("frozen.jsonl"),
            ["x"],
            candidates,
            "gaussian_score",
            lambda candidate: False,
        )
        self.assertEqual(
            [record["source_blob_index"] for record in traces["x"]], [2, 8]
        )


if __name__ == "__main__":
    unittest.main()
