"""验证模式一清单只在样本收集边界过滤零 occurrence。"""

from __future__ import annotations

import json

import numpy as np

from matcher.manifest import build_ground_truth_manifest, build_stage1_pointer


def test_ground_truth_manifest_freezes_occurrence_ids(tmp_path) -> None:
    data_root = tmp_path / "data"
    for pdb_id, rows in (("one", [{"candidate_id": 7}]), ("zero", [])):
        path = data_root / "parse" / pdb_id / "occurrences.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text(json.dumps(["one", "zero"]), encoding="utf-8")
    output = tmp_path / "manifest.json"
    result = build_ground_truth_manifest(
        data_root=data_root,
        split_files={"train": split, "validation": split, "calibration": split},
        output=output,
    )
    assert result["splits"]["train"] == [
        {"pdb_id": "one", "occurrence_ids": [7], "num_occurrences": 1}
    ]
    assert any(row["reason"] == "zero_occurrence" for row in result["excluded"])


def test_stage1_pointer_freezes_occurrence_ids_and_counts(tmp_path) -> None:
    data_root = tmp_path / "data"
    occurrence_path = data_root / "parse" / "one" / "occurrences.jsonl"
    occurrence_path.parent.mkdir(parents=True)
    occurrence_path.write_text(
        json.dumps({"candidate_id": 8}) + "\n" + json.dumps({"candidate_id": 3}) + "\n",
        encoding="utf-8",
    )
    pdb_root = tmp_path / "artifacts" / "Find_0" / "train" / "one"
    (pdb_root / "status" / "F1_centered").mkdir(parents=True)
    (pdb_root / "centered").mkdir()
    (pdb_root / "probability").mkdir()
    (pdb_root / "status" / "F1_centered" / "_COMPLETE").write_text("", encoding="utf-8")
    np.savez(pdb_root / "centered" / "F1_centered.npz", centered_box_index=np.arange(4))
    np.savez(pdb_root / "probability" / "probability_map.npz", probability_map=np.zeros(1))
    (pdb_root / "probability" / "geometry.json").write_text("{}", encoding="utf-8")

    result = build_stage1_pointer(
        artifact_root=tmp_path / "artifacts",
        data_root=data_root,
        producer="Find_0",
        splits=("train",),
        role="F1_centered",
        output=tmp_path / "pointer.json",
    )

    assert result["schema_version"] == 2
    assert result["records"][0]["num_candidates"] == 4
    assert result["records"][0]["occurrence_ids"] == [8, 3]
