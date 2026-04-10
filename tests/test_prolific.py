import json

from llm_judge_audit.datasets.prolific import (
    apply_prolific_annotations,
    build_annotation_progress,
)


def test_apply_prolific_annotations_merges_and_computes_majority(tmp_path):
    anchor = tmp_path / "anchor.json"
    annotations = tmp_path / "annotations.csv"
    output = tmp_path / "anchor_annotated.json"

    anchor.write_text(
        json.dumps(
            {
                "version": "1.0",
                "items": [
                    {
                        "item_id": "item-1",
                        "domain": "code",
                        "difficulty": "easy",
                        "prompt": "P1",
                        "response_a": "A1",
                        "response_b": "B1",
                        "human_annotations": [],
                        "majority_preference": "Tie",
                    },
                    {
                        "item_id": "item-2",
                        "domain": "factual",
                        "difficulty": "easy",
                        "prompt": "P2",
                        "response_a": "A2",
                        "response_b": "B2",
                        "human_annotations": [],
                        "majority_preference": "Tie",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    annotations.write_text(
        "\n".join(
            [
                "item_id,participant_id,preference,confidence,rationale",
                "item-1,p1,A,5,good",
                "item-1,p2,B,4,ok",
                "item-1,p3,A,3,better",
                "item-2,p1,A,5,first",
                "item-2,p2,A,4,second",
                "item-2,p3,A,5,third",
            ]
        ),
        encoding="utf-8",
    )

    apply_prolific_annotations(
        anchor_path=anchor,
        annotations_csv_path=annotations,
        output_path=output,
        rater_id_col="participant_id",
    )

    merged = json.loads(output.read_text(encoding="utf-8"))
    item1 = next(i for i in merged["items"] if i["item_id"] == "item-1")
    item2 = next(i for i in merged["items"] if i["item_id"] == "item-2")

    assert len(item1["human_annotations"]) == 3
    assert item1["majority_preference"] == "A"
    assert item1["is_gold_standard"] is False

    assert len(item2["human_annotations"]) == 3
    assert item2["majority_preference"] == "A"
    assert item2["is_gold_standard"] is True


def test_apply_prolific_annotations_dedupes_same_rater_latest_row_wins(tmp_path):
    anchor = tmp_path / "anchor.json"
    annotations = tmp_path / "annotations.csv"
    output = tmp_path / "anchor_annotated.json"

    anchor.write_text(
        json.dumps(
            {
                "version": "1.0",
                "items": [
                    {
                        "item_id": "item-1",
                        "domain": "creative",
                        "difficulty": "easy",
                        "prompt": "P1",
                        "response_a": "A1",
                        "response_b": "B1",
                        "human_annotations": [],
                        "majority_preference": "Tie",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    annotations.write_text(
        "\n".join(
            [
                "item_id,participant_id,preference,confidence,rationale",
                "item-1,p1,A,5,first",
                "item-1,p1,B,2,updated",
                "item-1,p2,B,3,other",
                "item-1,p3,B,4,third",
                "item-1,p4,invalid,4,ignored",
            ]
        ),
        encoding="utf-8",
    )

    apply_prolific_annotations(
        anchor_path=anchor,
        annotations_csv_path=annotations,
        output_path=output,
        rater_id_col="participant_id",
    )

    merged = json.loads(output.read_text(encoding="utf-8"))
    item = merged["items"][0]
    assert len(item["human_annotations"]) == 3
    assert item["majority_preference"] == "B"


def test_apply_prolific_annotations_allow_partial(tmp_path):
    anchor = tmp_path / "anchor.json"
    annotations = tmp_path / "annotations.csv"
    output = tmp_path / "anchor_partial.json"

    anchor.write_text(
        json.dumps(
            {
                "version": "1.0",
                "items": [
                    {
                        "item_id": "item-1",
                        "domain": "code",
                        "difficulty": "easy",
                        "prompt": "P1",
                        "response_a": "A1",
                        "response_b": "B1",
                        "human_annotations": [],
                        "majority_preference": "Tie",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    annotations.write_text(
        "\n".join(
            [
                "item_id,participant_id,preference",
                "item-1,p1,A",
                "item-1,p2,A",
            ]
        ),
        encoding="utf-8",
    )

    apply_prolific_annotations(
        anchor_path=anchor,
        annotations_csv_path=annotations,
        output_path=output,
        rater_id_col="participant_id",
        require_min_annotations=False,
    )
    merged = json.loads(output.read_text(encoding="utf-8"))
    item = merged["items"][0]
    assert len(item["human_annotations"]) == 2
    assert item["majority_preference"] == "A"


def test_build_annotation_progress(tmp_path):
    anchor = tmp_path / "anchor.json"
    annotations = tmp_path / "annotations.csv"
    anchor.write_text(
        json.dumps(
            {
                "version": "1.0",
                "items": [
                    {
                        "item_id": "code-1",
                        "domain": "code",
                        "difficulty": "easy",
                        "prompt": "P1",
                        "response_a": "A1",
                        "response_b": "B1",
                        "human_annotations": [],
                        "majority_preference": "Tie",
                    },
                    {
                        "item_id": "fact-1",
                        "domain": "factual",
                        "difficulty": "easy",
                        "prompt": "P2",
                        "response_a": "A2",
                        "response_b": "B2",
                        "human_annotations": [],
                        "majority_preference": "Tie",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    annotations.write_text(
        "\n".join(
            [
                "item_id,participant_id,preference",
                "code-1,p1,A",
                "code-1,p2,A",
                "code-1,p3,B",
                "fact-1,p1,B",
            ]
        ),
        encoding="utf-8",
    )

    progress = build_annotation_progress(
        anchor_path=anchor,
        annotations_csv_path=annotations,
        rater_id_col="participant_id",
    )

    assert progress["total_items"] == 2
    assert progress["items_with_at_least_1_annotation"] == 2
    assert progress["items_with_at_least_3_annotations"] == 1
    assert progress["domains"]["code"]["items_with_at_least_3_annotations"] == 1
