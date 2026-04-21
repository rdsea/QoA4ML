import glob
import json
import logging
import os
import pathlib
import re
import shutil
import subprocess
from functools import lru_cache
from typing import Any

import numpy as np
import psutil
import yaml

from qoa4ml.utils.logger import qoa_logger

_REDUCER_MAP = {
    "dot": ".",
    "underscore": "_",
    "path": "/",
}


def _resolve_sep(sep: str) -> str:
    """Resolve a reducer name (e.g. 'dot') to its separator character."""
    return _REDUCER_MAP.get(sep, sep)


def flatten(d: dict, sep: str = ".", parent_key: str = "") -> dict:
    """
    Flatten a nested dictionary into a single-level dictionary with
    keys joined by the given separator.

    Parameters
    ----------
    d : dict
        The nested dictionary to flatten.
    sep : str, optional
        The separator used to join keys. Accepts reducer names such as
        "dot", "underscore", "path", or a literal separator character.
        Default is ".".
    parent_key : str, optional
        The prefix for keys (used in recursion), default is "".

    Returns
    -------
    dict
        A flattened dictionary.
    """
    resolved = _resolve_sep(sep)
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{resolved}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten(v, sep=sep, parent_key=new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten(d: dict, sep: str = ".") -> dict:
    """
    Unflatten a single-level dictionary (with keys containing the separator)
    back into a nested dictionary.

    Parameters
    ----------
    d : dict
        The flat dictionary to unflatten.
    sep : str, optional
        The separator used in the flat keys. Accepts reducer names such as
        "dot", "underscore", "path", or a literal separator character.
        Default is ".".

    Returns
    -------
    dict
        A nested dictionary.
    """
    resolved = _resolve_sep(sep)
    result: dict = {}
    for key, value in d.items():
        parts = key.split(resolved)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


def make_folder(temp_path: str) -> bool:
    """
    Create a folder if it doesn't already exist.

    Parameters
    ----------
    temp_path : str
        The path of the folder to be created.

    Returns
    -------
    bool
        True if the folder exists or is created successfully, False otherwise.

    Notes
    -----
    If the folder already exists, nothing is done.
    """
    try:
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        return True
    except OSError:
        return False


@lru_cache(maxsize=1)
def get_cgroup_version() -> str:
    """
    Retrieve the current cgroup version.

    Returns
    -------
    str
        The cgroup version ("v1" or "v2"). Defaults to "v1" on systems
        where `mount` cannot be executed (e.g., Windows, minimal images).

    Notes
    -----
    Result is cached; the subprocess only runs on first call. Prior
    versions invoked this at module import, which slowed startup and
    broke imports on systems without `mount`.
    """
    # Resolve `mount` against $PATH explicitly so we never execute a
    # PATH-injected binary if the process is launched with a hostile env.
    mount_bin = shutil.which("mount")
    if mount_bin is None:
        qoa_logger.debug("`mount` not found on PATH; defaulting cgroup version to v1")
        return "v1"
    try:
        proc = subprocess.run(
            [mount_bin],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError) as error:
        qoa_logger.debug(
            f"Could not detect cgroup version ({type(error).__name__}); defaulting to v1"
        )
        return "v1"
    return "v2" if "cgroup2" in proc.stdout else "v1"


def __getattr__(name: str) -> Any:
    # Backward-compat shim: code that used the module-level CGROUP_VERSION
    # constant still works, but the subprocess call is deferred to first
    # access rather than running at import time.
    if name == "CGROUP_VERSION":
        return get_cgroup_version()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def set_logger_level(logging_level: int) -> None:
    """
    Set the logging level for the application logger.

    Parameters
    ----------
    logging_level : int
        The desired logging level:
        0 - NOTSET
        1 - DEBUG
        2 - INFO
        3 - WARNING
        4 - ERROR
        5 - CRITICAL

    Raises
    ------
    ValueError
        If the logging level is not between 0 and 5.
    """
    log_levels = [
        logging.NOTSET,
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL,
    ]

    if not 0 <= logging_level < len(log_levels):
        raise ValueError(f"Error logging level {logging_level}")

    qoa_logger.setLevel(log_levels[logging_level])


def load_config(file_path: str) -> dict | None:
    """
    Load a configuration file.

    Parameters
    ----------
    file_path : str
        The path to the configuration file.

    Returns
    -------
    dict
        The loaded configuration dictionary.

    Notes
    -----
    Supports JSON and YAML file formats. Logs a warning if the format is unsupported.
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            if "json" in file_path:
                return json.load(f)
            elif "yaml" in file_path or "yml" in file_path:
                return yaml.safe_load(f)
            else:
                qoa_logger.warning("Unsupported format")
                return None
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as e:
        qoa_logger.error(f"Unable to load configuration: {e}")
        return None


def to_json(file_path: str, conf: dict) -> None:
    """
    Save a configuration to a JSON file.

    Parameters
    ----------
    file_path : str
        The path to the file where the configuration should be saved.
    conf : dict
        The configuration dictionary to save.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(conf, f)


def to_yaml(file_path: str, conf: dict) -> None:
    """
    Save a configuration to a YAML file.

    Parameters
    ----------
    file_path : str
        The path to the file where the configuration should be saved.
    conf : dict
        The configuration dictionary to save.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(conf, f)


def get_sys_cpu() -> dict:
    """
    Retrieve system CPU statistics and times.

    Returns
    -------
    dict
        Dictionary containing CPU stats and times.

    Notes
    -----
    Uses psutil to retrieve both CPU stats and times.
    """
    stats = psutil.cpu_stats()
    cpu_time = psutil.cpu_times()
    return {key: getattr(stats, key) for key in stats._fields} | {
        key: getattr(cpu_time, key) for key in cpu_time._fields
    }


def get_sys_cpu_util() -> dict:
    """
    Retrieve system CPU utilization for each core.

    Returns
    -------
    dict
        Dictionary containing CPU utilization for each core.
    """
    core_utils = psutil.cpu_percent(percpu=True)
    return {
        f"core_{core_num}": core_util for core_num, core_util in enumerate(core_utils)
    }


def get_sys_cpu_metadata() -> dict:
    """
    Retrieve metadata information about the system CPU.

    Returns
    -------
    dict
        Dictionary containing CPU frequency and thread count.
    """
    cpu_freq = psutil.cpu_freq()
    return {
        "frequency": {"value": cpu_freq.max / 1000, "unit": "GHz"},
        "thread": psutil.cpu_count(logical=True),
    }


def get_sys_mem() -> dict:
    """
    Retrieve system memory statistics.

    Returns
    -------
    dict
        Dictionary containing memory stats.
    """
    stats = psutil.virtual_memory()
    return {key: getattr(stats, key) for key in stats._fields}


def get_sys_net() -> dict:
    """
    Retrieve system network I/O statistics.

    Returns
    -------
    dict
        Dictionary containing network I/O stats.
    """
    net = psutil.net_io_counters()
    return {key: getattr(net, key) for key in net._fields}


def report_proc_cpu(process: psutil.Process) -> dict:
    """
    Retrieve CPU usage statistics for a given process.

    Parameters
    ----------
    process : psutil.Process
        The process to retrieve CPU stats for.

    Returns
    -------
    dict
        Dictionary containing CPU stats for the process.
    """
    cpu_time = process.cpu_times()
    context = process.num_ctx_switches()
    return (
        {key: getattr(cpu_time, key) for key in cpu_time._fields}
        | {key: getattr(context, key) for key in context._fields}
        | {"num_thread": process.num_threads()}
    )


def report_proc_child_cpu(process: psutil.Process) -> dict:
    """
    Retrieve CPU usage statistics for a given process and its children.

    Parameters
    ----------
    process : psutil.Process
        The process to retrieve CPU stats for.

    Returns
    -------
    dict
        Dictionary containing CPU stats for the process and its children.

    Notes
    -----
    This function can be time-consuming as it recursively evaluates all child processes.
    """
    child_processes = process.children(recursive=True)
    child_processes_cpu = {
        f"child_{id}": float(
            child_proc.cpu_times().user + child_proc.cpu_times().system
        )
        for id, child_proc in enumerate(child_processes)
    }

    process_cpu_time = process.cpu_times()
    main_process = float(
        process_cpu_time.user
        + process_cpu_time.system
        + process_cpu_time.children_user
        + process_cpu_time.children_system
    )
    total_cpu_usage = sum(child_processes_cpu.values()) + main_process

    return {
        "child_process": len(child_processes),
        "value": child_processes_cpu,
        "main_process": main_process,
        "total": total_cpu_usage,
        "unit": "cputime",
    }


def get_proc_cpu(pid: int | None = None) -> dict:
    """
    Retrieve CPU usage statistics for a given process and its children.

    Parameters
    ----------
    pid : int, optional
        The process ID to retrieve CPU stats for. If None, uses the current process ID.

    Returns
    -------
    dict
        Dictionary containing CPU stats for the process and its children.

    Notes
    -----
    - The main process's stats are keyed by its PID.
    - Each child process's stats are keyed by its PID with a "c" suffix.
    """
    if pid is None:
        pid = os.getpid()
    process = psutil.Process(pid)
    child_list = process.children()
    info: dict[int | str, dict] = {}
    info[pid] = report_proc_cpu(process)

    for child in child_list:
        info[f"{child.pid}c"] = report_proc_cpu(child)
    return info


def report_proc_mem(process: psutil.Process) -> dict:
    """
    Retrieve memory usage statistics for a given process.

    Parameters
    ----------
    process : psutil.Process
        The process to retrieve memory stats for.

    Returns
    -------
    dict
        Dictionary containing memory stats for the process.
    """
    mem_info = process.memory_info()
    return {key: getattr(mem_info, key) for key in mem_info._fields}


def get_proc_mem(pid: int | None = None) -> dict:
    """
    Retrieve memory usage statistics for a given process and its children.

    Parameters
    ----------
    pid : int, optional
        The process ID to retrieve memory stats for. If None, uses the current process ID.

    Returns
    -------
    dict
        Dictionary containing memory stats for the process and its children.

    Notes
    -----
    - The main process's stats are keyed by its PID.
    - Each child process's stats are keyed by its PID with a "c" suffix.
    """
    if pid is None:
        pid = os.getpid()
    process = psutil.Process(pid)
    child_list = process.children()
    info: dict[int | str, dict] = {}
    info[pid] = report_proc_mem(process)

    for child in child_list:
        info[f"{child.pid}c"] = report_proc_mem(child)
    return info


def convert_to_gbyte(value: float) -> float:
    """
    Convert a value from bytes to gigabytes.

    Parameters
    ----------
    value : float
        The value in bytes to be converted.

    Returns
    -------
    float
        The converted value in gigabytes.
    """
    return value / 1024.0 / 1024.0 / 1024.0


def convert_to_mbyte(value: float) -> float:
    """
    Convert a value from bytes to megabytes.

    Parameters
    ----------
    value : float
        The value in bytes to be converted.

    Returns
    -------
    float
        The converted value in megabytes.
    """
    return value / 1024.0 / 1024.0


def convert_to_kbyte(value: float) -> float:
    """
    Convert a value from bytes to kilobytes.

    Parameters
    ----------
    value : float
        The value in bytes to be converted.

    Returns
    -------
    float
        The converted value in kilobytes.
    """
    return value / 1024.0


###################### DOCKER REPORT ######################


def get_cpu_stat(stats: dict, key: str) -> float:
    """
    Retrieve CPU usage statistics from Docker stats.

    Parameters
    ----------
    stats : dict
        The Docker stats dictionary.
    key : str
        The key indicating the type of CPU statistic (e.g., "percentage").

    Returns
    -------
    float
        The CPU usage percentage, or -1 if the key is not recognized.

    Notes
    -----
    - Calculates the CPU usage percentage based on the difference between the current and previous CPU usage.
    """
    if key == "percentage":
        usage_delta = (
            stats["cpu_stats"]["cpu_usage"]["total_usage"]
            - stats["precpu_stats"]["cpu_usage"]["total_usage"]
        )
        system_delta = (
            stats["cpu_stats"]["system_cpu_usage"]
            - stats["precpu_stats"]["system_cpu_usage"]
        )
        len_cpu = stats["cpu_stats"]["online_cpus"]
        percentage = (usage_delta / system_delta) * len_cpu * 100
        return round(percentage, 2)
    return -1


def get_mem_stat(stats: dict, key: str) -> int:
    """
    Retrieve memory usage statistics from Docker stats.

    Parameters
    ----------
    stats : dict
        The Docker stats dictionary.
    key : str
        The key indicating the type of memory statistic (e.g., "used").

    Returns
    -------
    int
        The memory usage in bytes, or -1 if the key is not recognized.
    """
    if key == "used":
        return stats["memory_stats"]["usage"]
    return -1


def merge_report(f_report: Any, i_report: Any, prio: bool = True) -> Any:
    """Recursively merge two report values without mutating either input.

    Parameters
    ----------
    f_report :
        The first report.
    i_report :
        The second report.
    prio : bool, optional
        On scalar conflict, prefer ``f_report`` when ``True`` (default),
        otherwise prefer ``i_report``.

    Returns
    -------
    The merged value. When both inputs are dicts, returns a new dict and
    recurses into shared keys. When both are non-dict values, returns
    whichever side ``prio`` selects.

    Notes
    -----
    Earlier versions mutated both arguments and silently swallowed
    exceptions; both are now treated as bugs. Callers can rely on input
    immutability.
    """
    if isinstance(f_report, dict) and isinstance(i_report, dict):
        merged: dict = {}
        for key, f_value in f_report.items():
            if key in i_report:
                merged[key] = merge_report(f_value, i_report[key], prio)
            else:
                merged[key] = f_value
        for key, i_value in i_report.items():
            if key not in merged:
                merged[key] = i_value
        return merged
    if f_report == i_report:
        return f_report
    return f_report if prio else i_report


def get_dict_at(dictionary: dict, i: int = 0) -> tuple:
    """Retrieve the ``(key, value)`` pair at a specific insertion-order index.

    Parameters
    ----------
    dictionary : dict
        Source dictionary.
    i : int, optional
        Index of the pair to retrieve, default ``0``.

    Returns
    -------
    tuple
        ``(key, value)`` pair at position ``i``.

    Raises
    ------
    IndexError
        If ``i`` is out of range. Previously the error was logged and
        :data:`None` returned, which caused callers unpacking the result
        to hit :class:`TypeError` instead of a clear index error.
    """
    keys = list(dictionary.keys())
    return keys[i], dictionary[keys[i]]


def get_file_dir(file: str, to_string: bool = True):
    """
    Get the directory of a file.

    Parameters
    ----------
    file : str
        The file path.
    to_string : bool, optional
        Flag to return the directory as a string, default is True.

    Returns
    -------
    str or pathlib.Path
        The directory of the file as a string or Path object.
    """
    current_dir = pathlib.Path(file).parent.absolute()
    return str(current_dir) if to_string else current_dir


def get_parent_dir(file: str, parent_level: int = 1, to_string: bool = True):
    """
    Get the parent directory of a file by a specified number of levels.

    Parameters
    ----------
    file : str
        The file path.
    parent_level : int, optional
        The number of levels up to retrieve the parent directory, default is 1.
    to_string : bool, optional
        Flag to return the directory as a string, default is True.

    Returns
    -------
    str or pathlib.Path
        The parent directory of the file as a string or Path object.
    """
    current_dir = get_file_dir(file=file, to_string=False)
    for _ in range(parent_level):
        current_dir = current_dir.parent.absolute()
    return str(current_dir) if to_string else current_dir


def is_numpyarray(obj: Any) -> bool:
    """
    Check if an object is a NumPy array.

    Parameters
    ----------
    obj : Any
        The object to check.

    Returns
    -------
    bool
        True if the object is a NumPy array, False otherwise.
    """
    return isinstance(obj, np.ndarray)


def get_process_allowed_cpus() -> list[int]:
    """
    Retrieve the list of CPU cores available to the process.

    Returns
    -------
    list[int]
        A list of CPU core indices.

    Notes
    -----
    - Uses the call process's PID (0) to get the CPU affinity.
    """
    pid = 0
    affinity = os.sched_getaffinity(pid)  # type: ignore[attr-defined]  # Linux-only API
    return list(affinity)


def get_process_allowed_memory() -> float | None:
    """
    Retrieve the memory limit allowed to the process.

    Returns
    -------
    Optional[float]
        The memory limit in bytes, or None if unable to retrieve.

    Notes
    -----
    - Supports both cgroup v1 and v2 formats to get the memory limit.
    """
    if get_cgroup_version() == "v1":
        with open("/proc/self/cgroup") as file:
            for line in file:
                parts = line.strip().split(":")
                if len(parts) == 3 and parts[1] == "memory":
                    cgroup_path = parts[2]
                    memory_limit_file = re.sub(r"/task_\d+", "", cgroup_path)

                    number_of_tasks = len(
                        glob.glob(f"/sys/fs/cgroup/memory{memory_limit_file}/task_*")
                    )

                    with open(
                        f"/sys/fs/cgroup/memory{memory_limit_file}/memory.limit_in_bytes"
                    ) as limit_file:
                        memory_limit_str = limit_file.read().strip()
                        try:
                            memory_limit_int = int(memory_limit_str)
                            return memory_limit_int / number_of_tasks
                        except ValueError:
                            return None
            return None
    else:
        with open("/proc/self/cgroup") as file:
            for line in file:
                parts = line.strip().split(":")
                cgroup_path = parts[2]
                pattern = r"/task_\d+"
                cgroup_path = re.sub(pattern, "", cgroup_path)
                with open(f"/sys/fs/cgroup{cgroup_path}/memory.max") as limit_file:
                    number_of_tasks = len(
                        glob.glob(f"/sys/fs/cgroup{cgroup_path}/task_*")
                    )
                    memory_limit_str = limit_file.read().strip()
                    try:
                        memory_limit_int = int(memory_limit_str)
                        return memory_limit_int / number_of_tasks
                    except ValueError:
                        return None
            return None
