from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from test_smallmolecule.build_manifest import _target_from_intersections
from test_smallmolecule.evaluate_stage1 import evaluate_stage1
from test_smallmolecule.ranking import build_topk_curve


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _candidate(
    blob_index: int,
    *,
    selected: bool,
    score: float,
    target_scope: str,
) -> dict:
    return {
        "pdb_id": "9rd8",
        "source_blob_index": blob_index,
        "centered_box_index": blob_index,
        "candidate_selected": selected,
        "gaussian_score": score,
        "source_probability_mean": score,
        "target_scope": target_scope,
        "matched_occurrence_id": 1 if target_scope != "background" else None,
        "target_smiles": "CC" if target_scope != "background" else None,
    }


def test_unique_target_uses_overlap_then_occurrence_id() -> None:
    target = _target_from_intersections(
        np.asarray([7, 3, 5]),
        np.asarray([40, 40, 10]),
        100,
        np.asarray([100, 100, 20]),
    )
    assert target == (3, 0.4, 0.4, 0.4)


def test_attempt_order_keeps_selected_prefix_and_free_skip() -> None:
    candidates = [
        _candidate(
            0,
            selected=True,
            score=3.0,
            target_scope="non_stage3_foreground",
        ),
        _candidate(1, selected=True, score=2.0, target_scope="background"),
        _candidate(
            2,
            selected=False,
            score=9.0,
            target_scope="stage3_foreground",
        ),
    ]
    curve, traces = build_topk_curve(
        ["9rd8"],
        candidates,
        "gaussian_score",
        lambda candidate: candidate["source_blob_index"] == 2,
    )
    assert curve["hit_count"][:2] == [0, 1]
    assert [record["source_blob_index"] for record in traces["9rd8"]] == [0, 1, 2]
    assert [record["attempt_index"] for record in traces["9rd8"]] == [None, 1, 2]


def test_stage1_product_contains_only_scientific_results(tmp_path: Path) -> None:
    manifest_root = tmp_path / "manifest"
    evaluation_root = tmp_path / "evaluation"
    manifest_root.mkdir()
    files = {
        "real_candidates": "real_candidates.jsonl",
        "cryoatom2_candidates": "cryoatom2_candidates.jsonl",
        "emap2lig_candidates": "emap2lig_candidates.jsonl",
    }
    (manifest_root / "manifest.json").write_text(
        json.dumps(
            {
                "coverage_threshold": 0.3,
                "pdb_ids": ["9rd8"],
                "files": files,
            }
        ),
        encoding="utf-8",
    )
    candidate = _candidate(
        0,
        selected=True,
        score=1.0,
        target_scope="stage3_foreground",
    )
    for filename in files.values():
        _write_jsonl(manifest_root / filename, [candidate])
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "outputs:\n"
        f"  manifest_root: {manifest_root.as_posix()}\n"
        f"  evaluation_root: {evaluation_root.as_posix()}\n",
        encoding="utf-8",
    )

    result_path = evaluate_stage1(config_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert set(result) == {"coverage_threshold", "curves", "emap2lig", "traces"}
    assert set(result["curves"]["gaussian"]["real_receptor"]) == {
        "pdb_ids",
        "top_k",
        "hit_count",
        "hit_ratio",
        "pdb_hit",
    }
