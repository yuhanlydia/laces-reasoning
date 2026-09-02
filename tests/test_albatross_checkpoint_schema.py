"""Red-first test contract for scripts/tools/inspect_albatross_checkpoint.py.

All tests intentionally fail until Task 7 implements the CLI because the tool
script does not exist yet.  The failures must be ``FileNotFoundError`` /
``AssertionError`` for a missing script, NOT syntax, import, or fixture errors.

Synthetic ``.pth`` checkpoints are created with ``torch.save`` inside
``tmp_path`` using representative Albatross-model keys (``blocks.N.att.x_r``,
``blocks.N.att.w0``, ``blocks.N.ffn.key.weight``, ``emb.weight``,
``ln_out.weight``, ``head.weight``, etc.).

    python -m pytest tests/test_albatross_checkpoint_schema.py -v
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOOL_PATH = os.path.join(
    REPO_ROOT, "scripts", "tools", "inspect_albatross_checkpoint.py"
)

# Small synthetic dimensions — keep cpu-only, no GPU requirement.
EMBD = 64
N_LAYER = 2
VOCAB = 128


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _run_tool(
    *args: str,
    cwd: str | None = None,
) -> tuple[int, str, str]:
    """Execute the inspection CLI and return (returncode, stdout, stderr)."""
    cmd = [sys.executable, TOOL_PATH, *args]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd or REPO_ROOT,
    )
    return result.returncode, result.stdout, result.stderr


def _make_valid_ckpt(tmp_path: str, name: str = "albatross_valid.pth") -> str:
    """Create a synthetic Albatross-like checkpoint with representative keys.

    Includes the keys listed in the task spec plus enough coverage to
    exercise the ``recognized`` grouping logic of the future tool.
    """
    state: dict[str, torch.Tensor] = {}
    for layer in range(N_LAYER):
        blk = f"blocks.{layer}"
        state[f"{blk}.att.x_r"] = torch.randn(1, 1, EMBD)
        state[f"{blk}.att.w0"] = torch.randn(1, 1, EMBD)
        state[f"{blk}.att.w1"] = torch.randn(1, 1, EMBD)
        state[f"{blk}.att.w2"] = torch.randn(1, 1, EMBD)
        state[f"{blk}.att.a0"] = torch.randn(1, 1, EMBD)
        state[f"{blk}.att.a1"] = torch.randn(1, 1, EMBD)
        state[f"{blk}.att.a2"] = torch.randn(1, 1, EMBD)
        state[f"{blk}.att.g0"] = torch.randn(1, 1, EMBD)
        state[f"{blk}.att.g1"] = torch.randn(1, 1, EMBD)
        state[f"{blk}.att.g2"] = torch.randn(1, 1, EMBD)
        state[f"{blk}.att.k_k"] = torch.randn(1, 1, EMBD)
        state[f"{blk}.att.k_a"] = torch.randn(1, 1, EMBD)
        state[f"{blk}.att.r_k"] = torch.randn(1, 1, EMBD)
        state[f"{blk}.att.receptance.weight"] = torch.randn(EMBD, EMBD)
        state[f"{blk}.att.key.weight"] = torch.randn(EMBD, EMBD)
        state[f"{blk}.att.value.weight"] = torch.randn(EMBD, EMBD)
        state[f"{blk}.att.gate.weight"] = torch.randn(EMBD, EMBD)
        state[f"{blk}.att.ln_x.weight"] = torch.randn(EMBD)
        state[f"{blk}.att.ln_x.bias"] = torch.randn(EMBD)
        state[f"{blk}.ffn.key.weight"] = torch.randn(EMBD * 4, EMBD)
        state[f"{blk}.ffn.value.weight"] = torch.randn(EMBD, EMBD * 4)
        state[f"{blk}.ffn.receptance.weight"] = torch.randn(EMBD, EMBD)
        state[f"{blk}.ln0.weight"] = torch.randn(EMBD)
        state[f"{blk}.ln0.bias"] = torch.randn(EMBD)
        state[f"{blk}.ln1.weight"] = torch.randn(EMBD)
        state[f"{blk}.ln1.bias"] = torch.randn(EMBD)
        state[f"{blk}.ln2.weight"] = torch.randn(EMBD)
        state[f"{blk}.ln2.bias"] = torch.randn(EMBD)

    state["emb.weight"] = torch.randn(VOCAB, EMBD)
    state["ln_out.weight"] = torch.randn(EMBD)
    state["ln_out.bias"] = torch.randn(EMBD)
    state["head.weight"] = torch.randn(VOCAB, EMBD)

    path = os.path.join(tmp_path, name)
    torch.save(state, path)
    return path


def _make_missing_ckpt(tmp_path: str) -> str:
    """Checkpoint MISSING required top-level fields (ln_out, head, ffn.key)."""
    state: dict[str, torch.Tensor] = {}
    for layer in range(N_LAYER):
        blk = f"blocks.{layer}"
        state[f"{blk}.att.x_r"] = torch.randn(1, 1, EMBD)
        state[f"{blk}.att.w0"] = torch.randn(1, 1, EMBD)
        state[f"{blk}.att.receptance.weight"] = torch.randn(EMBD, EMBD)
        state[f"{blk}.att.key.weight"] = torch.randn(EMBD, EMBD)
        state[f"{blk}.att.value.weight"] = torch.randn(EMBD, EMBD)
        state[f"{blk}.att.gate.weight"] = torch.randn(EMBD, EMBD)
        state[f"{blk}.ffn.receptance.weight"] = torch.randn(EMBD, EMBD)
        state[f"{blk}.ln0.weight"] = torch.randn(EMBD)
        state[f"{blk}.ln1.weight"] = torch.randn(EMBD)
        state[f"{blk}.ln2.weight"] = torch.randn(EMBD)

    state["emb.weight"] = torch.randn(VOCAB, EMBD)
    # intentionally missing:  ln_out.weight, head.weight, blocks.*.ffn.key.weight

    path = os.path.join(tmp_path, "albatross_missing_keys.pth")
    torch.save(state, path)
    return path


WRONG_EMBD = 128  # intentionally different from EMBD=64


def _make_wrong_embd_ckpt(tmp_path: str) -> str:
    """Synthetic checkpoint where several tensors have dimension WRONG_EMBD
    instead of EMBD, so ``--expected-embd EMBD`` should flag mismatches."""
    state: dict[str, torch.Tensor] = {}

    for layer in range(N_LAYER):
        blk = f"blocks.{layer}"
        # Albatross vectors — last dim = WRONG_EMBD
        for vec_key in ("x_r", "w0", "w1", "w2", "a0", "a1", "a2",
                        "g0", "g1", "g2", "k_k", "k_a", "r_k"):
            state[f"{blk}.att.{vec_key}"] = torch.randn(1, 1, WRONG_EMBD)
        # Square projection weights — both dims = WRONG_EMBD
        for proj_key in ("receptance", "key", "value", "gate"):
            state[f"{blk}.att.{proj_key}.weight"] = torch.randn(WRONG_EMBD, WRONG_EMBD)
        state[f"{blk}.att.ln_x.weight"] = torch.randn(WRONG_EMBD)
        state[f"{blk}.att.ln_x.bias"] = torch.randn(WRONG_EMBD)
        # FFN: key last dim = WRONG_EMBD, value first dim = WRONG_EMBD
        state[f"{blk}.ffn.key.weight"] = torch.randn(WRONG_EMBD * 4, WRONG_EMBD)
        state[f"{blk}.ffn.value.weight"] = torch.randn(WRONG_EMBD, WRONG_EMBD * 4)
        state[f"{blk}.ffn.receptance.weight"] = torch.randn(WRONG_EMBD, WRONG_EMBD)
        # Layer norms
        for ln_i in ("ln0", "ln1", "ln2"):
            state[f"{blk}.{ln_i}.weight"] = torch.randn(WRONG_EMBD)
            state[f"{blk}.{ln_i}.bias"] = torch.randn(WRONG_EMBD)

    state["emb.weight"] = torch.randn(VOCAB, WRONG_EMBD)
    state["ln_out.weight"] = torch.randn(WRONG_EMBD)
    state["ln_out.bias"] = torch.randn(WRONG_EMBD)
    state["head.weight"] = torch.randn(VOCAB, WRONG_EMBD)

    path = os.path.join(tmp_path, "albatross_wrong_embd.pth")
    torch.save(state, path)
    return path


# ---------------------------------------------------------------------------
# red-first guard — every test fails cleanly when the tool is absent
# ---------------------------------------------------------------------------

def _require_tool() -> None:
    """Fail with a clear, targeted message when the tool has not been created.

    This is the ONLY red-first failure trigger; the tests themselves are
    syntactically valid, type-clean, and do not import the missing script.
    """
    if not os.path.isfile(TOOL_PATH):
        pytest_msg = (
            f"Red-first: {TOOL_PATH} does not exist.\n"
            "This contract will remain red until Task 7 creates the inspection CLI.\n"
            "No syntax / import / fixture errors — the test file itself is valid."
        )
        raise AssertionError(pytest_msg)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


class TestAlbatrossCheckpointSchemaContract:
    """Pytest contract for ``scripts/tools/inspect_albatross_checkpoint.py``."""

    # -- schema compliance -------------------------------------------------

    def test_valid_checkpoint_produces_expected_json_fields(self, tmp_path):
        """Valid Albatross .pth → JSON report with all 5 mandatory top-level keys.

        Expected keys: recognized, missing_required, unexpected, shape_summary,
        diff_rwkv_mapping_preview.
        """
        _require_tool()
        ckpt_path = _make_valid_ckpt(str(tmp_path), "valid.pth")
        output_json = os.path.join(str(tmp_path), "report.json")

        rc, stdout, stderr = _run_tool(
            "--input", ckpt_path,
            "--output-json", output_json,
            "--expected-layers", str(N_LAYER),
            "--expected-embd", str(EMBD),
        )
        assert rc == 0, f"Tool exited non-zero (rc={rc}):\n{stderr}"
        assert os.path.isfile(output_json), (
            f"Output JSON not created.\nstdout={stdout}\nstderr={stderr}"
        )

        with open(output_json) as f:
            report = json.load(f)

        required = [
            "recognized",
            "missing_required",
            "unexpected",
            "shape_summary",
            "diff_rwkv_mapping_preview",
        ]
        for field in required:
            assert field in report, (
                f"Missing mandatory JSON field '{field}'. Got keys: {sorted(report)}"
            )

        # Valid checkpoint → recognized non-empty, missing_required empty
        assert isinstance(report["recognized"], list)
        assert len(report["recognized"]) > 0, "recognized must not be empty for a valid checkpoint"
        assert len(report["missing_required"]) == 0, (
            "missing_required must be empty for a valid checkpoint"
        )
        assert isinstance(report["shape_summary"], dict)
        assert len(report["shape_summary"]) > 0

    def test_valid_checkpoint_recognized_groups_and_shape_summary(self, tmp_path):
        """Recognized groups contain representative Albatross key categories.

        shape_summary maps at least emb, head, and ln_out keys.
        """
        _require_tool()
        ckpt_path = _make_valid_ckpt(str(tmp_path), "valid2.pth")
        output_json = os.path.join(str(tmp_path), "report2.json")

        rc, _, stderr = _run_tool(
            "--input", ckpt_path,
            "--output-json", output_json,
        )
        assert rc == 0, f"Tool failed: {stderr}"

        with open(output_json) as f:
            report = json.load(f)

        shapes = report["shape_summary"]
        assert len(shapes) > 1

        # At least one of emb / head / ln_out must appear in the shape summary.
        flat_keys = {str(k).lower() for k in shapes}
        assert any("emb" in k for k in flat_keys), (
            f"shape_summary should contain embedding key; got {sorted(flat_keys)}"
        )
        assert any("head" in k for k in flat_keys), (
            f"shape_summary should contain head weight key; got {sorted(flat_keys)}"
        )

        # recognized groups should mention blocks, att, ffn, emb, or head
        groups_lower = [g.lower() for g in report["recognized"]]
        assert any(
            "block" in g or "att" in g or "ffn" in g for g in groups_lower
        ), f"recognized groups should mention block/attention/ffn; got {report['recognized']}"

    def test_missing_required_key_strict_mode_nonzero_exit(self, tmp_path):
        """--strict mode exits non-zero when required keys are absent.

        missing_required in the JSON report is non-empty.
        """
        _require_tool()
        ckpt_path = _make_missing_ckpt(str(tmp_path))
        output_json = os.path.join(str(tmp_path), "report_missing.json")

        rc, stdout, stderr = _run_tool(
            "--input", ckpt_path,
            "--output-json", output_json,
            "--strict",
        )
        assert rc != 0, (
            f"--strict must exit non-zero on missing required keys "
            f"(got rc={rc}).\nstdout={stdout}\nstderr={stderr}"
        )
        assert os.path.isfile(output_json), "JSON report must still be written on error"

        with open(output_json) as f:
            report = json.load(f)

        missing = report["missing_required"]
        assert isinstance(missing, list)
        assert len(missing) > 0, (
            f"missing_required must be non-empty; got {missing}"
        )
        # Expect at least ln_out / head / ffn.key to appear in missing_required
        missing_lower = [str(m).lower() for m in missing]
        assert any("ln_out" in m for m in missing_lower) or any("head" in m for m in missing_lower), (
            f"Expected ln_out or head in missing_required; got {missing}"
        )

    def test_diff_rwkv_mapping_preview_contains_block_mappings(self, tmp_path):
        """diff_rwkv_mapping_preview maps Albatross keys → DiffRwkv keys."""
        _require_tool()
        ckpt_path = _make_valid_ckpt(str(tmp_path), "valid3.pth")
        output_json = os.path.join(str(tmp_path), "report3.json")

        rc, _, stderr = _run_tool(
            "--input", ckpt_path,
            "--output-json", output_json,
        )
        assert rc == 0, f"Tool failed: {stderr}"

        with open(output_json) as f:
            report = json.load(f)

        preview = report["diff_rwkv_mapping_preview"]
        assert isinstance(preview, (dict, list))

        if isinstance(preview, dict):
            assert len(preview) > 0, "mapping preview must not be empty"
            # Keys should be Albatross → DiffRwkv style mappings
            for alb_key, diff_kv_key in preview.items():
                assert isinstance(alb_key, str)
                assert isinstance(diff_kv_key, str)

    # -- report-only behaviour ---------------------------------------------

    def test_no_model_pt_or_converted_checkpoint_created(self, tmp_path):
        """The inspection tool is report-only — it must not write model.pt
        or any converted checkpoint file."""
        _require_tool()
        ckpt_path = _make_valid_ckpt(str(tmp_path), "input.pth")
        output_json = os.path.join(str(tmp_path), "report_no_write.json")

        before = set(os.listdir(str(tmp_path)))

        rc, _, stderr = _run_tool(
            "--input", ckpt_path,
            "--output-json", output_json,
        )
        assert rc == 0, f"Tool failed: {stderr}"

        after = set(os.listdir(str(tmp_path)))
        new = after - before - {"report_no_write.json"}

        forbidden = [f for f in new if "model" in f.lower() or f != "report_no_write.json"]
        assert not forbidden, (
            f"Tool wrote unexpected output: {forbidden}. "
            "Checkpoint inspection must be report-only."
        )

    def test_no_extra_pth_output_created(self, tmp_path):
        """No .pth files beyond the original synthetic input are created."""
        _require_tool()
        ckpt_path = _make_valid_ckpt(str(tmp_path), "input2.pth")
        output_json = os.path.join(str(tmp_path), "report_no_pth.json")

        rc, _, stderr = _run_tool(
            "--input", ckpt_path,
            "--output-json", output_json,
        )
        assert rc == 0, f"Tool failed: {stderr}"

        all_pth = [
            f for f in os.listdir(str(tmp_path))
            if f.endswith(".pth")
        ]
        # Only the synthetic input .pth files are expected.
        extra = [f for f in all_pth if f not in ("input2.pth", "input.pth")]
        assert not extra, f"Unexpected .pth files created: {extra}"

    # -- CLI flag contract -------------------------------------------------

    def test_cli_accepts_expected_layers_and_embd_flags(self, tmp_path):
        """--expected-layers and --expected-embd are accepted and do not
        cause the tool to reject a valid checkpoint."""
        _require_tool()
        ckpt_path = _make_valid_ckpt(str(tmp_path), "valid_flags.pth")
        output_json = os.path.join(str(tmp_path), "report_flags.json")

        rc, _, stderr = _run_tool(
            "--input", ckpt_path,
            "--output-json", output_json,
            "--expected-layers", "2",
            "--expected-embd", "64",
        )
        assert rc == 0, (
            f"Tool must accept --expected-layers / --expected-embd.\n{stderr}"
        )

    def test_cli_accepts_strict_flag(self, tmp_path):
        """--strict is accepted and does not crash on a valid checkpoint."""
        _require_tool()
        ckpt_path = _make_valid_ckpt(str(tmp_path), "valid_strict.pth")
        output_json = os.path.join(str(tmp_path), "report_strict.json")

        rc, _, stderr = _run_tool(
            "--input", ckpt_path,
            "--output-json", output_json,
            "--strict",
        )
        # Valid checkpoint + strict → should still exit 0
        assert rc == 0, f"Tool must accept --strict flag.\n{stderr}"

    # -- expected-embd validation -------------------------------------------

    def test_expected_embd_valid_checkpoint_empty_mismatches(self, tmp_path):
        """--expected-embd on a valid checkpoint produces empty shape_mismatches."""
        _require_tool()
        ckpt_path = _make_valid_ckpt(str(tmp_path), "valid_embd.pth")
        output_json = os.path.join(str(tmp_path), "report_embd.json")

        rc, _, stderr = _run_tool(
            "--input", ckpt_path,
            "--output-json", output_json,
            "--expected-embd", str(EMBD),
        )
        assert rc == 0, f"Tool failed: {stderr}"

        with open(output_json) as f:
            report = json.load(f)

        assert "shape_mismatches" in report, (
            "shape_mismatches field must be present in JSON report"
        )
        assert len(report["shape_mismatches"]) == 0, (
            f"Expected empty shape_mismatches for valid checkpoint, "
            f"got {report['shape_mismatches']}"
        )

    def test_expected_embd_wrong_checkpoint_detects_mismatches(self, tmp_path):
        """--expected-embd on a wrong-dimension checkpoint populates shape_mismatches."""
        _require_tool()
        ckpt_path = _make_wrong_embd_ckpt(str(tmp_path))
        output_json = os.path.join(str(tmp_path), "report_wrong_embd.json")

        rc, _, stderr = _run_tool(
            "--input", ckpt_path,
            "--output-json", output_json,
            "--expected-embd", str(EMBD),
            "--expected-layers", str(N_LAYER),
        )
        assert rc == 0, (
            f"Tool should exit 0 without --strict even with mismatches; "
            f"got rc={rc}\n{stderr}"
        )
        assert os.path.isfile(output_json), "JSON report must be written"

        with open(output_json) as f:
            report = json.load(f)

        mismatches = report["shape_mismatches"]
        assert isinstance(mismatches, list)
        assert len(mismatches) > 0, (
            f"Expected non-empty shape_mismatches when embd={EMBD} but "
            f"tensors use {WRONG_EMBD}"
        )

        # All mismatches should reference the correct expected/actual values
        for m in mismatches:
            assert m["expected"] == EMBD
            assert m["actual"] == WRONG_EMBD

    def test_expected_embd_strict_exits_nonzero_on_mismatches(self, tmp_path):
        """--strict + --expected-embd exits non-zero on incompatible shapes,
        while still writing the JSON report."""
        _require_tool()
        ckpt_path = _make_wrong_embd_ckpt(str(tmp_path))
        output_json = os.path.join(str(tmp_path), "report_strict_embd.json")

        rc, stdout, stderr = _run_tool(
            "--input", ckpt_path,
            "--output-json", output_json,
            "--expected-embd", str(EMBD),
            "--strict",
        )
        assert rc != 0, (
            f"--strict must exit non-zero when embd dimension is incompatible; "
            f"got rc={rc}\nstdout={stdout}\nstderr={stderr}"
        )
        assert os.path.isfile(output_json), (
            "JSON report must still be written on strict shape mismatch"
        )

        with open(output_json) as f:
            report = json.load(f)

        mismatches = report["shape_mismatches"]
        assert len(mismatches) > 0, (
            "shape_mismatches must be non-empty when dimensions are wrong"
        )
