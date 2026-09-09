from __future__ import annotations

import json

import pytest

from models.arc_data import (
    ArcPair,
    ArcTask,
    inverse_transform_task,
    load_arc_split,
    parse_arc_context,
    serialize_arc_context,
    transform_task,
    validate_task,
    verify_split_counts,
)


@pytest.fixture
def sample_task() -> ArcTask:
    return ArcTask(
        task_id="sample",
        train=(
            ArcPair(input=((0, 1, 2), (3, 4, 5)), output=((5, 4), (3, 2), (1, 0))),
            ArcPair(input=((6, 7), (8, 9)), output=((9, 8), (7, 6))),
        ),
        test=(ArcPair(input=((1, 0, 2),), output=((2,), (0,), (1,))),),
    )


def test_all_dihedral_transforms_are_invertible(sample_task):
    transformed_forms = set()
    for transform_id in range(8):
        transformed = transform_task(sample_task, transform_id)
        assert inverse_transform_task(transformed, transform_id) == sample_task
        transformed_forms.add(transformed.train[0].input)
    assert len(transformed_forms) == 8


def test_context_serialization_round_trips(sample_task):
    text = serialize_arc_context(sample_task, query_index=0)
    train, query = parse_arc_context(text)
    assert train == sample_task.train
    assert query == sample_task.test[0].input
    assert text.endswith("[Q][I]102[/I][O]")


@pytest.mark.parametrize(
    "bad_grid",
    [(), ((0, 1), (2,)), ((10,),), tuple((0,) for _ in range(31))],
)
def test_validation_rejects_invalid_grids(sample_task, bad_grid):
    broken = ArcTask(
        task_id=sample_task.task_id,
        train=(ArcPair(input=bad_grid, output=((0,),)),),
        test=sample_task.test,
    )
    with pytest.raises(ValueError):
        validate_task(broken)


def test_load_split_rejects_duplicate_ids_and_reads_official_schema(tmp_path):
    payload = {
        "train": [{"input": [[0, 1]], "output": [[1, 0]]}],
        "test": [{"input": [[2]], "output": [[2]]}],
    }
    (tmp_path / "a.json").write_text(json.dumps(payload))
    named_payload = {**payload, "name": "b"}
    (tmp_path / "b.json").write_text(json.dumps(named_payload))
    tasks = load_arc_split(tmp_path)
    assert [task.task_id for task in tasks] == ["a", "b"]
    assert tasks[0].train[0].input == ((0, 1),)

    with pytest.raises(ValueError, match="overlap"):
        verify_split_counts(tasks, tasks, expected_training=2, expected_evaluation=2)


def test_verify_split_counts_requires_exact_sizes(sample_task):
    evaluation = ArcTask("other", sample_task.train, sample_task.test)
    verify_split_counts(
        [sample_task], [evaluation], expected_training=1, expected_evaluation=1
    )
    with pytest.raises(ValueError, match="training tasks"):
        verify_split_counts(
            [sample_task], [evaluation], expected_training=400, expected_evaluation=1
        )
