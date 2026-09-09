"""Strict ARC-AGI-1 task loading, augmentation, and compact serialization."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
from typing import Iterable, Sequence
from urllib.request import urlretrieve

Grid = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ArcPair:
    input: Grid
    output: Grid


@dataclass(frozen=True)
class ArcTask:
    task_id: str
    train: tuple[ArcPair, ...]
    test: tuple[ArcPair, ...]


def _grid(value: Sequence[Sequence[int]]) -> Grid:
    return tuple(tuple(int(cell) for cell in row) for row in value)


def validate_grid(grid: Grid) -> None:
    if not grid or len(grid) > 30:
        raise ValueError("grid height must be within 1..30")
    width = len(grid[0])
    if width < 1 or width > 30 or any(len(row) != width for row in grid):
        raise ValueError("grid rows must be rectangular with width within 1..30")
    if any(cell < 0 or cell > 9 for row in grid for cell in row):
        raise ValueError("grid colors must be integers within 0..9")


def validate_task(task: ArcTask) -> None:
    if not task.task_id:
        raise ValueError("task_id must be non-empty")
    if not task.train or not task.test:
        raise ValueError("ARC tasks require training and test pairs")
    for pair in (*task.train, *task.test):
        validate_grid(pair.input)
        validate_grid(pair.output)


def load_arc_task(path: str | Path) -> ArcTask:
    path = Path(path)
    payload = json.loads(path.read_text())
    keys = set(payload)
    if not {"train", "test"}.issubset(keys) or keys - {"train", "test", "name"}:
        raise ValueError(f"{path}: expected train/test and optional name keys")
    if "name" in payload and str(payload["name"]) != path.stem:
        raise ValueError(f"{path}: embedded name does not match filename")

    def pair(value: dict) -> ArcPair:
        if set(value) != {"input", "output"}:
            raise ValueError(f"{path}: pair must contain input/output")
        return ArcPair(input=_grid(value["input"]), output=_grid(value["output"]))

    task = ArcTask(
        task_id=path.stem,
        train=tuple(pair(item) for item in payload["train"]),
        test=tuple(pair(item) for item in payload["test"]),
    )
    validate_task(task)
    return task


def load_arc_split(path: str | Path) -> list[ArcTask]:
    paths = sorted(Path(path).glob("*.json"))
    tasks = [load_arc_task(item) for item in paths]
    ids = [task.task_id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task IDs in split")
    return tasks


def verify_split_counts(
    training: Sequence[ArcTask],
    evaluation: Sequence[ArcTask],
    *,
    expected_training: int = 400,
    expected_evaluation: int = 400,
) -> None:
    if len(training) != expected_training:
        raise ValueError(f"expected {expected_training} training tasks, got {len(training)}")
    if len(evaluation) != expected_evaluation:
        raise ValueError(
            f"expected {expected_evaluation} evaluation tasks, got {len(evaluation)}"
        )
    overlap = {task.task_id for task in training} & {task.task_id for task in evaluation}
    if overlap:
        raise ValueError(f"training/evaluation task ID overlap: {sorted(overlap)[:5]}")


def _rotate_clockwise(grid: Grid) -> Grid:
    return tuple(tuple(row[col] for row in reversed(grid)) for col in range(len(grid[0])))


def _mirror_left_right(grid: Grid) -> Grid:
    return tuple(tuple(reversed(row)) for row in grid)


def transform_grid(grid: Grid, transform_id: int) -> Grid:
    if transform_id not in range(8):
        raise ValueError("transform_id must be within 0..7")
    result = _mirror_left_right(grid) if transform_id >= 4 else grid
    for _ in range(transform_id % 4):
        result = _rotate_clockwise(result)
    return result


def inverse_transform_grid(grid: Grid, transform_id: int) -> Grid:
    inverse_id = transform_id if transform_id >= 4 else (-transform_id) % 4
    return transform_grid(grid, inverse_id)


def _map_task(task: ArcTask, fn) -> ArcTask:
    def map_pair(pair: ArcPair) -> ArcPair:
        return ArcPair(input=fn(pair.input), output=fn(pair.output))

    return ArcTask(
        task_id=task.task_id,
        train=tuple(map_pair(pair) for pair in task.train),
        test=tuple(map_pair(pair) for pair in task.test),
    )


def transform_task(task: ArcTask, transform_id: int) -> ArcTask:
    return _map_task(task, lambda grid: transform_grid(grid, transform_id))


def inverse_transform_task(task: ArcTask, transform_id: int) -> ArcTask:
    return _map_task(task, lambda grid: inverse_transform_grid(grid, transform_id))


def _serialize_grid(grid: Grid) -> str:
    validate_grid(grid)
    return "/".join("".join(str(cell) for cell in row) for row in grid)


def _parse_grid(text: str) -> Grid:
    if not text:
        raise ValueError("empty serialized grid")
    grid = tuple(tuple(int(cell) for cell in row) for row in text.split("/"))
    validate_grid(grid)
    return grid


def serialize_arc_context(task: ArcTask, query_index: int) -> str:
    validate_task(task)
    if query_index < 0 or query_index >= len(task.test):
        raise IndexError("query_index outside test pairs")
    demos = "".join(
        f"[D][I]{_serialize_grid(pair.input)}[/I][O]{_serialize_grid(pair.output)}[/O]"
        for pair in task.train
    )
    query = _serialize_grid(task.test[query_index].input)
    return f"{demos}[Q][I]{query}[/I][O]"


_DEMO_RE = re.compile(r"\[D\]\[I\](.*?)\[/I\]\[O\](.*?)\[/O\]")
_QUERY_RE = re.compile(r"\[Q\]\[I\](.*?)\[/I\]\[O\]$")


def parse_arc_context(text: str) -> tuple[tuple[ArcPair, ...], Grid]:
    query_match = _QUERY_RE.search(text)
    if query_match is None:
        raise ValueError("serialized ARC context has no terminal query")
    prefix = text[: query_match.start()]
    matches = list(_DEMO_RE.finditer(prefix))
    if not matches or "".join(match.group(0) for match in matches) != prefix:
        raise ValueError("serialized ARC demonstrations are malformed")
    demos = tuple(
        ArcPair(input=_parse_grid(match.group(1)), output=_parse_grid(match.group(2)))
        for match in matches
    )
    return demos, _parse_grid(query_match.group(1))


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if root != target and root not in target.parents:
            raise ValueError(f"unsafe archive path: {member.name}")
    archive.extractall(destination)


def prepare_official_arc(
    root: str | Path,
    source_url: str,
    source_ref: str,
) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="arc-agi1-") as tmp_name:
        tmp = Path(tmp_name)
        archive_path = tmp / "source.tar.gz"
        urlretrieve(source_url, archive_path)
        with tarfile.open(archive_path, "r:gz") as archive:
            _safe_extract(archive, tmp / "extract")
        data_roots = list((tmp / "extract").glob("*/data"))
        if len(data_roots) != 1:
            raise ValueError("official archive must contain one top-level data directory")
        source_data = data_roots[0]
        staged = tmp / "staged"
        for split in ("training", "evaluation"):
            shutil.copytree(source_data / split, staged / split)
        training = load_arc_split(staged / "training")
        evaluation = load_arc_split(staged / "evaluation")
        verify_split_counts(training, evaluation)
        for split in ("training", "evaluation"):
            target = root / split
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(staged / split, target)

    manifest = {
        "dataset": "ARC-AGI-1",
        "source_url": source_url,
        "source_ref": source_ref,
        "training_tasks": 400,
        "evaluation_tasks": 400,
        "training_ids": [task.task_id for task in training],
        "evaluation_ids": [task.task_id for task in evaluation],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
