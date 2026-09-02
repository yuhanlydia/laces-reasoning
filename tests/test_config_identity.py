"""TDD tests for config-name and checkpoint identity validation.

RED phase: tests that SHOULD FAIL before config_name fixes are applied.
All tests are CPU-safe — no CUDA, no model loading, no GPU memory allocation.

Usage:
    pytest tests/test_config_identity.py -v
    CUDA_VISIBLE_DEVICES="" pytest tests/test_config_identity.py -v
"""
import glob
import os
import sys

import pytest
import yaml
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Helper: config_name from YAML file stem
# ---------------------------------------------------------------------------

def _config_name_from_stem(filepath: str) -> str:
    """Extract expected config_name from filename stem (without .yaml)."""
    return os.path.splitext(os.path.basename(filepath))[0]


# ---------------------------------------------------------------------------
# Helper: checkpoint identity classifier (CPU-safe)
# ---------------------------------------------------------------------------

def classify_checkpoint_identity(config_yaml_str: str) -> str:
    """Classify a relay checkpoint config string.

    Returns one of:
        "13.3B validated"   — strong 13.3B identity signals (>2 indicators)
        "identity pending"  — missing config_name but model path looks plausible
        "misnamed/non-13.3B" — clearly wrong (0.4B path/name, or mismatched signals)
        "unknown"           — not enough info to classify
    """
    cfg = OmegaConf.create(config_yaml_str)

    # Collect signals
    signals_13b = 0
    signals_other = 0

    rwkv_path = str(cfg.model.get("rwkv_local_path", ""))
    rwkv_name = str(cfg.model.get("rwkv_name", ""))
    config_name = cfg.get("config_name", None)

    # Signal: model path contains 13.3B identifier
    if any(token in rwkv_path for token in ["13.3B", "13.3b", "RWKV7-G1f"]):
        signals_13b += 1
    elif any(token in rwkv_path for token in ["0.4B", "0.4b", "0.4B-world"]):
        signals_other += 1

    # Signal: model name contains 13B/g1 identifier
    if any(token in rwkv_name.lower() for token in ["g1", "13.3b", "13.3b"]):
        signals_13b += 1
    elif any(token in rwkv_name.lower() for token in ["0.4b"]):
        signals_other += 1

    # Signal: config_name (if present) contains 13.3B
    if config_name is not None:
        if "13.3B" in str(config_name) or "13.3b" in str(config_name):
            signals_13b += 1
        elif "0.4B" in str(config_name) or "0.4b" in str(config_name):
            signals_other += 1

    # Classification logic
    if signals_13b >= 2 and signals_other == 0:
        return "13.3B validated"
    elif signals_13b >= 1 and config_name is None:
        return "identity pending"
    elif signals_other > 0 and signals_13b == 0:
        return "misnamed/non-13.3B"
    elif signals_13b >= 1 and signals_other >= 1:
        return "misnamed/non-13.3B"  # contradictory signals
    else:
        return "unknown"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def configs_dir():
    """Path to the configs/ directory."""
    return os.path.join(os.path.dirname(__file__), "..", "configs")


@pytest.fixture(scope="module")
def all_vae32_configs(configs_dir):
    """All rwkv_relay_*state_hijack_dit_vae32*.yaml config files."""
    pattern = os.path.join(configs_dir, "rwkv_relay_*state_hijack_dit_vae32*.yaml")
    return sorted(glob.glob(pattern))


@pytest.fixture(scope="module")
def config_13_3b(configs_dir):
    """The 13.3B config as a dictionary."""
    path = os.path.join(configs_dir, "rwkv_relay_13.3B_state_hijack_dit_vae32.yaml")
    if not os.path.isfile(path):
        pytest.skip("13.3B config not found")
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def omega_config_13_3b(configs_dir):
    """The 13.3B config as OmegaConf DictConfig."""
    path = os.path.join(configs_dir, "rwkv_relay_13.3B_state_hijack_dit_vae32.yaml")
    if not os.path.isfile(path):
        pytest.skip("13.3B config not found")
    return OmegaConf.load(path)


# ---------------------------------------------------------------------------
# Test 1: Every vae32 config has config_name matching file stem
# RED PHASE: this WILL FAIL for 13.3B, 2.9B, and 2.9B_mlp_alpha configs
# ---------------------------------------------------------------------------

class TestConfigNameCompleteness:
    """Verify all relay vae32 configs declare a config_name matching their filename."""

    # Configs that are documented exceptions (e.g., Hydra override derivations
    # that intentionally inherit the parent's config_name)
    DOCUMENTED_EXCEPTIONS: set = {
        # kl01 is a high-KL variant reusing 0.4B parent config_name
        "rwkv_relay_0.4B_state_hijack_dit_vae32_kl01",
    }

    def test_config_name_field_exists(self, all_vae32_configs):
        """RED: Every vae32 config YAML must contain a 'config_name' top-level key.

        Fails for: 13.3B, 2.9B base, 2.9B mlp_alpha (all missing config_name).
        """
        missing = []
        for path in all_vae32_configs:
            with open(path) as f:
                data = yaml.safe_load(f)
            if "config_name" not in data:
                missing.append(os.path.basename(path))

        # Fail with clear message showing ALL missing files
        assert not missing, (
            f"config_name missing in {len(missing)} vae32 config(s): {', '.join(missing)}"
        )

    def test_config_name_matches_file_stem(self, all_vae32_configs):
        """RED: config_name value must match the YAML filename stem.

        Exceptions: explicitly documented inherited-name overrides (kl01, etc.).
        Fails for configs that have config_name but it mismatches the filename.
        """
        mismatched = []
        for path in all_vae32_configs:
            stem = _config_name_from_stem(path)
            if stem in self.DOCUMENTED_EXCEPTIONS:
                continue
            with open(path) as f:
                data = yaml.safe_load(f)
            config_name_val = data.get("config_name")
            if config_name_val is None:
                # Already caught by test_config_name_field_exists — skip here
                continue
            if config_name_val != stem:
                mismatched.append(f"{stem} → {config_name_val}")

        assert not mismatched, (
            f"config_name mismatch in {len(mismatched)} config(s): {', '.join(mismatched)}"
        )


# ---------------------------------------------------------------------------
# Test 2: 13.3B config identity — does NOT default to 0.4B fallback
# RED PHASE: config_name is missing, so the fallback triggers
# ---------------------------------------------------------------------------

class Test13_3BConfigIdentity:
    """Verify the 13.3B config resolves to 13.3B identity, not 0.4B."""

    def test_config_name_exists_in_13_3b(self, config_13_3b):
        """RED: 13.3B config must have a 'config_name' key.

        Currently missing — this assertion INTENTIONALLY fails in RED phase.
        After Task 3 adds config_name to the YAML, this will pass.
        """
        assert "config_name" in config_13_3b, (
            "13.3B config is missing 'config_name' field. "
            "This causes train_state_hijacking_dit.py line 153 to log "
            "the hardcoded fallback 'rwkv_relay_0.4B_state_hijack_dit_vae32' "
            "producing a misleading train.log entry for 13.3B runs."
        )

    def test_13_3b_config_name_is_not_0_4b(self, config_13_3b):
        """If config_name exists, it must NOT reference 0.4B."""
        cn = config_13_3b.get("config_name", "")
        assert "0.4B" not in str(cn), (
            f"13.3B config_name='{cn}' incorrectly references 0.4B"
        )
        assert "0.4b" not in str(cn).lower(), (
            f"13.3B config_name='{cn}' incorrectly references 0.4B"
        )

    def test_simulated_logging_line_produces_13_3b_not_0_4b(self, config_13_3b):
        """RED: Simulate train_state_hijacking_dit.py:153 logging behavior.

        The real code is:
            logger.info(f"Config: {config.config_name if 'config_name' in config
                        else 'rwkv_relay_0.4B_state_hijack_dit_vae32'}")

        For 13.3B config (missing config_name), this produces:
            "Config: rwkv_relay_0.4B_state_hijack_dit_vae32"  ← MISLEADING

        After fix: the logged string must contain "13.3B" or the real config name.
        """
        # Simulate the production logging line
        config_name_val = (
            config_13_3b["config_name"]
            if "config_name" in config_13_3b
            else "rwkv_relay_0.4B_state_hijack_dit_vae32"
        )
        logged = f"Config: {config_name_val}"

        assert "0.4B" not in logged, (
            f"13.3B config logged as: {logged}\n"
            "The hardcoded fallback 'rwkv_relay_0.4B_state_hijack_dit_vae32' "
            "is being used because config_name is missing from the 13.3B YAML."
        )
        assert "13.3B" in logged or "13.3b" in logged, (
            f"13.3B config logged without 13.3B identifier: {logged}"
        )

    def test_model_path_identifies_13_3b(self, config_13_3b):
        """Model path must reference 13.3B backbone, not 0.4B."""
        rwkv_path = config_13_3b.get("model", {}).get("rwkv_local_path", "")
        rwkv_name = config_13_3b.get("model", {}).get("rwkv_name", "")

        path_has_13b = "13.3B" in rwkv_path or "13.3b" in rwkv_path.lower()
        name_has_13b = "13.3B" in rwkv_name or "13.3b" in rwkv_name.lower() or "g1" in rwkv_name.lower()

        assert path_has_13b or name_has_13b, (
            f"Neither model.rwkv_local_path ({rwkv_path}) nor "
            f"model.rwkv_name ({rwkv_name}) identifies 13.3B."
        )


# ---------------------------------------------------------------------------
# Test 3: Checkpoint identity classifier
# ---------------------------------------------------------------------------

class TestCheckpointIdentityClassifier:
    """Verify the classify_checkpoint_identity helper works on various configs."""

    def test_13_3b_config_classifies_as_validated(self, config_13_3b):
        """RED: 13.3B config must classify as '13.3B validated' by the identity classifier.

        Currently fails because config_name is missing → 'identity pending'.
        After Task 3 adds config_name to the 13.3B YAML, the classifier will
        see >2 13.3B signals and return '13.3B validated'.
        """
        import yaml as _yaml
        yaml_str = _yaml.dump(config_13_3b)

        result = classify_checkpoint_identity(yaml_str)

        assert result == "13.3B validated", (
            f"13.3B config classified as '{result}', expected '13.3B validated'. "
            "Missing config_name causes 'identity pending' instead. "
            "After Task 3 adds config_name to the 13.3B YAML, this should pass."
        )

    def test_0_4b_config_classifies_correctly(self, all_vae32_configs):
        """0.4B config (with config_name) should classify as non-13.3B."""
        for path in all_vae32_configs:
            stem = _config_name_from_stem(path)
            if "0.4B" not in stem and "0.4b" not in stem.lower():
                continue
            with open(path) as f:
                yaml_str = f.read()
            result = classify_checkpoint_identity(yaml_str)
            # 0.4B configs should never be "13.3B validated"
            assert result != "13.3B validated", (
                f"0.4B config {stem} incorrectly classified as '13.3B validated'"
            )

    def test_identity_classifier_handles_missing_config_name(self):
        """Classifier handles configs with missing config_name gracefully."""
        minimal = (
            "model:\n"
            "  rwkv_local_path: /path/to/RWKV7-G1f-13.3B-HF\n"
            "  rwkv_name: BlinkDL/rwkv7-g1\n"
            "training:\n"
            "  stage: 3\n"
        )
        result = classify_checkpoint_identity(minimal)
        # Missing config_name → identity pending, but path says 13.3B
        assert result in ("13.3B validated", "identity pending"), (
            f"Minimal 13.3B config classified as '{result}', expected 13.3B or identity_pending"
        )

    def test_identity_classifier_detects_wrong_model_path(self):
        """Classifier detects 0.4B path even with plausible config_name."""
        misleading = (
            "config_name: rwkv_relay_13.3B_state_hijack_dit_vae32\n"
            "model:\n"
            "  rwkv_local_path: /path/to/rwkv7-0.4B-world\n"
            "  rwkv_name: fla-hub/rwkv7-0.4B-world\n"
            "training:\n"
            "  stage: 3\n"
        )
        result = classify_checkpoint_identity(misleading)
        assert result == "misnamed/non-13.3B", (
            f"Misleading config (13.3B name, 0.4B path) classified as '{result}', "
            "expected 'misnamed/non-13.3B'"
        )

    def test_identity_classifier_handles_empty_config(self):
        """Classifier handles empty/minimal configs gracefully."""
        result = classify_checkpoint_identity("model: {}\ntraining: {}")
        assert result == "unknown", (
            f"Empty config classified as '{result}', expected 'unknown'"
        )


# ---------------------------------------------------------------------------
# Test 4: OmegaConf loading — verifying the OmegaConf/Hydra behavior
# ---------------------------------------------------------------------------

class TestOmegaConfBehavior:
    """Verify OmegaConf correctly loads configs with proper config_name identity."""

    def test_13_3b_omega_conf_must_have_config_name(self, omega_config_13_3b):
        """RED: 13.3B config loaded via OmegaConf MUST contain config_name.

        The production code at train_state_hijacking_dit.py:153 uses:
            'config_name' in config

        For the 13.3B config loaded via OmegaConf, this currently evaluates to
        False, triggering the hardcoded 0.4B fallback. This is the bug.

        After Task 3 fix (add config_name to 13.3B YAML), the OmegaConf config
        must have 'config_name' present and equal to the expected value.
        """
        assert "config_name" in omega_config_13_3b, (
            "13.3B config loaded via OmegaConf is missing 'config_name'. "
            "Add 'config_name: rwkv_relay_13.3B_state_hijack_dit_vae32' "
            "to configs/rwkv_relay_13.3B_state_hijack_dit_vae32.yaml"
        )
        assert omega_config_13_3b.config_name == "rwkv_relay_13.3B_state_hijack_dit_vae32", (
            f"13.3B OmegaConf config_name='{omega_config_13_3b.config_name}', "
            "expected 'rwkv_relay_13.3B_state_hijack_dit_vae32'"
        )

    def test_0_4b_has_config_name_in_omega_conf(self, configs_dir):
        """0.4B config loads correctly with config_name in OmegaConf."""
        path = os.path.join(configs_dir, "rwkv_relay_0.4B_state_hijack_dit_vae32.yaml")
        if not os.path.isfile(path):
            pytest.skip("0.4B config not found")
        cfg = OmegaConf.load(path)
        assert "config_name" in cfg, "0.4B config must have config_name in OmegaConf"
        assert cfg.config_name == "rwkv_relay_0.4B_state_hijack_dit_vae32"
