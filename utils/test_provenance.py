import json
import re
import subprocess
import time

import numpy as np
import pytest

from utils.provenance import save_with_provenance

REQUIRED_KEYS = {
    "script", "git_commit", "git_dirty", "timestamp",
    "params", "config_hash", "outputs", "duration_s",
}


def _real_head_commit():
    out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def test_sidecar_has_all_required_keys(tmp_path):
    out_path = tmp_path / "result.npz"
    sidecar = save_with_provenance(out_path, {"x": np.array([1, 2, 3])}, {"a": 1}, script="dummy.py")
    record = json.loads(sidecar.read_text())
    assert REQUIRED_KEYS.issubset(record.keys())


def test_sidecar_naming_drops_original_suffix(tmp_path):
    out_path = tmp_path / "steady_state_map.npz"
    sidecar = save_with_provenance(out_path, {"x": np.array([1])}, {}, script="dummy.py")
    assert sidecar.name == "steady_state_map.provenance.json"


def test_git_commit_matches_real_head(tmp_path):
    out_path = tmp_path / "result.npz"
    sidecar = save_with_provenance(out_path, {"x": np.array([1])}, {}, script="dummy.py")
    record = json.loads(sidecar.read_text())
    assert record["git_commit"] == _real_head_commit()


def test_git_dirty_is_a_bool(tmp_path):
    out_path = tmp_path / "result.npz"
    sidecar = save_with_provenance(out_path, {"x": np.array([1])}, {}, script="dummy.py")
    record = json.loads(sidecar.read_text())
    assert isinstance(record["git_dirty"], bool)


def test_config_hash_is_a_sha256_digest(tmp_path):
    out_path = tmp_path / "result.npz"
    sidecar = save_with_provenance(out_path, {"x": np.array([1])}, {}, script="dummy.py")
    record = json.loads(sidecar.read_text())
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", record["config_hash"])


def test_config_hash_is_deterministic_for_the_same_config(tmp_path):
    a = save_with_provenance(tmp_path / "a.npz", {"x": np.array([1])}, {}, script="dummy.py")
    b = save_with_provenance(tmp_path / "b.npz", {"x": np.array([1])}, {}, script="dummy.py")
    assert json.loads(a.read_text())["config_hash"] == json.loads(b.read_text())["config_hash"]


def test_duration_s_reflects_elapsed_time_since_start_time(tmp_path):
    start = time.time() - 1.0
    out_path = tmp_path / "result.npz"
    sidecar = save_with_provenance(out_path, {"x": np.array([1])}, {}, script="dummy.py", start_time=start)
    record = json.loads(sidecar.read_text())
    assert record["duration_s"] >= 0.9


def test_duration_s_is_none_when_start_time_omitted(tmp_path):
    out_path = tmp_path / "result.npz"
    sidecar = save_with_provenance(out_path, {"x": np.array([1])}, {}, script="dummy.py")
    record = json.loads(sidecar.read_text())
    assert record["duration_s"] is None


def test_params_round_trip_exactly(tmp_path):
    params = {"F_range": [5, 35, 0.25], "betas": [1.0, 0.8, 0.6]}
    out_path = tmp_path / "result.npz"
    sidecar = save_with_provenance(out_path, {"x": np.array([1])}, params, script="dummy.py")
    record = json.loads(sidecar.read_text())
    assert record["params"] == params


def test_script_field_matches_given_value(tmp_path):
    out_path = tmp_path / "result.npz"
    sidecar = save_with_provenance(out_path, {"x": np.array([1])}, {}, script="evaluation/steady_state_map.py")
    record = json.loads(sidecar.read_text())
    assert record["script"] == "evaluation/steady_state_map.py"


def test_data_dict_is_saved_as_npz(tmp_path):
    out_path = tmp_path / "result.npz"
    save_with_provenance(out_path, {"x": np.array([1, 2, 3]), "y": np.array([4.0, 5.0])}, {}, script="dummy.py")
    loaded = np.load(out_path)
    np.testing.assert_array_equal(loaded["x"], [1, 2, 3])
    np.testing.assert_array_equal(loaded["y"], [4.0, 5.0])


def test_data_none_only_writes_sidecar_not_the_output_file(tmp_path):
    out_path = tmp_path / "figure.png"
    out_path.write_bytes(b"fake png bytes")
    sidecar = save_with_provenance(out_path, None, {}, script="dummy.py")
    assert out_path.read_bytes() == b"fake png bytes"
    assert sidecar.exists()


def test_outputs_field_lists_every_given_path(tmp_path):
    p1 = tmp_path / "a.png"
    p2 = tmp_path / "b.png"
    p1.write_bytes(b"1")
    p2.write_bytes(b"2")
    sidecar = save_with_provenance([p1, p2], None, {}, script="dummy.py")
    record = json.loads(sidecar.read_text())
    assert set(record["outputs"]) == {str(p1), str(p2)}


def test_extra_fields_are_merged_into_the_record(tmp_path):
    out_path = tmp_path / "result.npz"
    sidecar = save_with_provenance(out_path, {"x": np.array([1])}, {}, script="dummy.py",
                                    extra={"beta_star": 0.38505})
    record = json.loads(sidecar.read_text())
    assert record["beta_star"] == 0.38505


def test_script_given_as_this_files_dunder_file_is_stored_repo_relative(tmp_path):
    out_path = tmp_path / "result.npz"
    sidecar = save_with_provenance(out_path, {"x": np.array([1])}, {}, script=__file__)
    record = json.loads(sidecar.read_text())
    assert record["script"] == "rl-nmpc-cstr/utils/test_provenance.py"
    assert not record["script"].startswith("/")


def test_script_is_required():
    with pytest.raises(TypeError):
        save_with_provenance("out.npz", {"x": np.array([1])}, {})
