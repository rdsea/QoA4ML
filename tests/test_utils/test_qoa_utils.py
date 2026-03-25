import json
import pathlib

import numpy as np
import pytest
import yaml

from qoa4ml.utils.qoa_utils import (
    convert_to_gbyte,
    convert_to_kbyte,
    convert_to_mbyte,
    get_dict_at,
    get_file_dir,
    get_parent_dir,
    is_numpyarray,
    load_config,
    merge_report,
    set_logger_level,
    to_json,
    to_yaml,
)


class TestLoadConfig:
    def test_load_json_config(self, tmp_path):
        config = {"key": "value", "number": 42}
        filepath = str(tmp_path / "config.json")
        with open(filepath, "w") as f:
            json.dump(config, f)

        result = load_config(filepath)
        assert result == config

    def test_load_yaml_config(self, tmp_path):
        config = {"key": "value", "nested": {"a": 1}}
        filepath = str(tmp_path / "config.yaml")
        with open(filepath, "w") as f:
            yaml.dump(config, f)

        result = load_config(filepath)
        assert result == config

    def test_load_yml_config(self, tmp_path):
        config = {"key": "value"}
        filepath = str(tmp_path / "config.yml")
        with open(filepath, "w") as f:
            yaml.dump(config, f)

        result = load_config(filepath)
        assert result == config

    def test_load_unsupported_format_returns_none(self, tmp_path):
        filepath = str(tmp_path / "config.toml")
        with open(filepath, "w") as f:
            f.write("[section]\nkey = 'value'\n")

        result = load_config(filepath)
        assert result is None

    def test_load_nonexistent_file_returns_none(self):
        result = load_config("/nonexistent/path/config.json")
        assert result is None


class TestToJson:
    def test_to_json_writes_file(self, tmp_path):
        config = {"name": "test", "value": 123}
        filepath = str(tmp_path / "output.json")
        to_json(filepath, config)

        with open(filepath) as f:
            loaded = json.load(f)
        assert loaded == config

    def test_to_json_overwrites_existing(self, tmp_path):
        filepath = str(tmp_path / "output.json")
        to_json(filepath, {"old": True})
        to_json(filepath, {"new": True})

        with open(filepath) as f:
            loaded = json.load(f)
        assert loaded == {"new": True}


class TestToYaml:
    def test_to_yaml_writes_file(self, tmp_path):
        config = {"name": "test", "items": [1, 2, 3]}
        filepath = str(tmp_path / "output.yaml")
        to_yaml(filepath, config)

        with open(filepath) as f:
            loaded = yaml.safe_load(f)
        assert loaded == config


class TestConvertBytes:
    def test_convert_to_gbyte(self):
        one_gb_in_bytes = 1024.0**3
        assert convert_to_gbyte(one_gb_in_bytes) == pytest.approx(1.0)

    def test_convert_to_gbyte_zero(self):
        assert convert_to_gbyte(0) == 0.0

    def test_convert_to_mbyte(self):
        one_mb_in_bytes = 1024.0**2
        assert convert_to_mbyte(one_mb_in_bytes) == pytest.approx(1.0)

    def test_convert_to_mbyte_zero(self):
        assert convert_to_mbyte(0) == 0.0

    def test_convert_to_kbyte(self):
        one_kb_in_bytes = 1024.0
        assert convert_to_kbyte(one_kb_in_bytes) == pytest.approx(1.0)

    def test_convert_to_kbyte_zero(self):
        assert convert_to_kbyte(0) == 0.0

    def test_conversion_consistency(self):
        value = 1073741824  # 1 GB in bytes
        gb = convert_to_gbyte(value)
        mb = convert_to_mbyte(value)
        kb = convert_to_kbyte(value)
        assert mb == pytest.approx(gb * 1024)
        assert kb == pytest.approx(mb * 1024)


class TestMergeReport:
    def test_merge_disjoint_dicts(self):
        f = {"a": 1}
        i = {"b": 2}
        result = merge_report(f, i)
        assert result == {"a": 1, "b": 2}

    def test_merge_overlapping_dicts_prio_true(self):
        f = {"a": 1}
        i = {"a": 2}
        result = merge_report(f, i, prio=True)
        assert result["a"] == 1

    def test_merge_overlapping_dicts_prio_false(self):
        f = {"a": 1}
        i = {"a": 2}
        result = merge_report(f, i, prio=False)
        assert result["a"] == 2

    def test_merge_nested_dicts(self):
        f = {"outer": {"a": 1}}
        i = {"outer": {"b": 2}}
        result = merge_report(f, i)
        assert result == {"outer": {"a": 1, "b": 2}}

    def test_merge_empty_dicts(self):
        assert merge_report({}, {}) == {}

    def test_merge_f_empty(self):
        result = merge_report({}, {"a": 1})
        assert result == {"a": 1}

    def test_merge_i_empty(self):
        result = merge_report({"a": 1}, {})
        assert result == {"a": 1}


class TestGetDictAt:
    def test_get_first_element(self):
        d = {"first": 10, "second": 20}
        key, value = get_dict_at(d, 0)
        assert key == "first"
        assert value == 10

    def test_get_second_element(self):
        d = {"first": 10, "second": 20}
        key, value = get_dict_at(d, 1)
        assert key == "second"
        assert value == 20

    def test_default_index_zero(self):
        d = {"only": 42}
        key, value = get_dict_at(d)
        assert key == "only"
        assert value == 42

    def test_out_of_range_returns_none(self):
        d = {"a": 1}
        result = get_dict_at(d, 5)
        assert result is None


class TestIsNumpyarray:
    def test_numpy_array_returns_true(self):
        arr = np.array([1, 2, 3])
        assert is_numpyarray(arr) is True

    def test_list_returns_false(self):
        assert is_numpyarray([1, 2, 3]) is False

    def test_none_returns_false(self):
        assert is_numpyarray(None) is False

    def test_string_returns_false(self):
        assert is_numpyarray("array") is False

    def test_empty_numpy_array(self):
        assert is_numpyarray(np.array([])) is True

    def test_2d_numpy_array(self):
        assert is_numpyarray(np.zeros((3, 3))) is True


class TestGetFileDir:
    def test_returns_parent_directory(self):
        result = get_file_dir("/some/path/to/file.py")
        assert result == "/some/path/to"

    def test_returns_string_by_default(self):
        result = get_file_dir("/some/path/file.py")
        assert isinstance(result, str)

    def test_returns_path_object_when_requested(self):
        result = get_file_dir("/some/path/file.py", to_string=False)
        assert isinstance(result, pathlib.Path)


class TestGetParentDir:
    def test_one_level_up(self):
        result = get_parent_dir("/a/b/c/file.py", parent_level=1)
        assert result.endswith("/a/b")

    def test_two_levels_up(self):
        result = get_parent_dir("/a/b/c/file.py", parent_level=2)
        assert result.endswith("/a")

    def test_returns_string_by_default(self):
        result = get_parent_dir("/a/b/c/file.py")
        assert isinstance(result, str)

    def test_returns_path_object(self):
        result = get_parent_dir("/a/b/c/file.py", to_string=False)
        assert isinstance(result, pathlib.Path)


class TestSetLoggerLevel:
    def test_set_valid_levels(self):
        for level in range(6):
            set_logger_level(level)

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError, match="Error logging level"):
            set_logger_level(6)

    def test_negative_level_raises(self):
        with pytest.raises(ValueError, match="Error logging level"):
            set_logger_level(-1)
