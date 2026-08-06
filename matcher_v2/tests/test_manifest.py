"""验证模式一清单只在样本收集边界过滤零 occurrence。"""

from __future__ import annotations

import json

from matcher_v2.manifest import build_ground_truth_manifest


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
