import glob
import json
import logging
import os
import pathlib
import re
import shlex
import subprocess
import sys
import time
import traceback
from threading import Thread
from typing import Any, Optional

import numpy as np
import psutil
import yaml

from qoa4ml.utils.logger import qoa_logger


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
    except Exception:
        return False


def get_cgroup_version() -> str:
    """
    Retrieve the current cgroup version.

    Returns
    -------
    str
        The cgroup version ("v1" or "v2").

    Notes
    -----
    Uses subprocess to execute the `mount` command and grep for cgroup version.
    """
    proc1 = subprocess.Popen("mount", stdout=subprocess.PIPE)
    proc2 = subprocess.Popen(
        shlex.split("grep cgroup"),
        stdin=proc1.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc1.stdout:
        proc1.stdout.close()
    out, _ = proc2.communicate()
    return "v2" if "cgroup2" in out.decode() else "v1"


if get_cgroup_version() == "v2":
    CGROUP_VERSION = "v2"
else:
    CGROUP_VERSION = "v1"


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


def load_config(file_path: str) -> Optional[dict]:
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
        with open(file_path) as f:
            if "json" in file_path:
                return json.load(f)
            elif "yaml" in file_path or "yml" in file_path:
                return yaml.safe_load(f)
            else:
                qoa_logger.warning("Unsupported format")
    except Exception:
        qoa_logger.error("Unable to load configuration")
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
    with open(file_path, "w") as f:
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
    with open(file_path, "w") as f:
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


def get_proc_cpu(pid: Optional[int] = None) -> dict:
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
    info = {}
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


def get_proc_mem(pid: Optional[int] = None) -> dict:
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
    info = {}
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


###################### SYSTEM REPORT ######################

sys_monitor_flag = False
process_monitor_flag = False
doc_monitor_flag = False


def system_report(
    client, interval: int, to_mb: bool = True, to_gb: bool = False, to_kb: bool = False
) -> None:
    """
    Generate a system report and send it to the client at regular intervals.

    Parameters
    ----------
    client : Any
        The client to send the report to.
    interval : int
        The time interval in seconds between reports.
    to_mb : bool, optional
        Convert network I/O values to megabytes, default is True.
    to_gb : bool, optional
        Convert network I/O values to gigabytes, default is False.
    to_kb : bool, optional
        Convert network I/O values to kilobytes, default is False.

    Notes
    -----
    - Collects CPU, memory, and network I/O statistics.
    - Logs errors if any occur during data collection or report sending.
    - Sleeps for the specified interval between reports.
    """
    report = {}
    last_net_value = {"sent": 0, "receive": 0}
    while sys_monitor_flag:
        try:
            report["sys_cpu_stats"] = get_sys_cpu()
        except Exception as e:
            qoa_logger.error(f"Error {type(e)} in report CPU stat: {e}")
            traceback.print_exception(*sys.exc_info())
        try:
            report["sys_mem_stats"] = get_sys_mem()
        except Exception as e:
            qoa_logger.error(f"Error {type(e)} in report memory stat: {e}")
            traceback.print_exception(*sys.exc_info())
        try:
            report["sys_net_stats"] = get_sys_net()
            sent = 0
            receive = 0
            if to_mb:
                sent = convert_to_mbyte(psutil.net_io_counters().bytes_sent)
                receive = convert_to_mbyte(psutil.net_io_counters().bytes_recv)
            elif to_gb:
                sent = convert_to_gbyte(psutil.net_io_counters().bytes_sent)
                receive = convert_to_gbyte(psutil.net_io_counters().bytes_recv)
            elif to_kb:
                sent = convert_to_kbyte(psutil.net_io_counters().bytes_sent)
                receive = convert_to_kbyte(psutil.net_io_counters().bytes_recv)
            else:
                sent = psutil.net_io_counters().bytes_sent
                receive = psutil.net_io_counters().bytes_recv

            curr_net_value = {"sent": sent, "receive": receive}
            report["sys_net_send"] = curr_net_value["sent"] - last_net_value["sent"]
            report["sys_net_receive"] = (
                curr_net_value["receive"] - last_net_value["receive"]
            )
            last_net_value = curr_net_value.copy()
        except Exception as e:
            qoa_logger.error(f"Error {type(e)} in report network stat: {e}")
            traceback.print_exception(*sys.exc_info())
        try:
            client.report(report=report)
        except Exception as e:
            qoa_logger.error(f"Error {type(e)} in sent system report: {e}")
            traceback.print_exception(*sys.exc_info())
        time.sleep(interval)


def sys_monitor(client, interval: int) -> None:
    """
    Start monitoring system reports.

    Parameters
    ----------
    client : Any
        The client to send the report to.
    interval : int
        The time interval in seconds between reports.

    Notes
    -----
    - Starts a new thread to generate system reports at regular intervals.
    """
    sub_thread = Thread(target=system_report, args=(client, interval))
    sub_thread.start()


###################### PROCESS REPORT ######################

# def process_report(client, interval:int, pid:int = None):
#     report = {}
#     while procMonitorFlag:
#         try:
#             report["proc_cpu_stats"] = get_proc_cpu()
#         except Exception as e:
#             qoaLogger.error("Error {} in report process cpu stat: {}".format(type(e),e.__traceback__))
#             traceback.print_exception(*sys.exc_info())
#         try:
#             report["proc_mem_stats"] = get_proc_mem()
#         except Exception as e:
#             qoaLogger.error("Error {} in report process memory stat: {}".format(type(e),e.__traceback__))
#             traceback.print_exception(*sys.exc_info())
#         try:
#             client.report(report=report)
#         except Exception as e:
#             qoaLogger.error("Error {} in sent process report: {}".format(type(e),e.__traceback__))
#             traceback.print_exception(*sys.exc_info())
#         time.sleep(interval)


# def process_monitor(client, interval:int, pid:int = None):
#     if (pid == None):
#         pid = os.getpid()
#     sub_thread = Thread(target=process_report, args=(client, interval, pid))
#     sub_thread.start()


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


def merge_report(f_report: dict, i_report: dict, prio: bool = True) -> dict:
    """
    Merge two report dictionaries.

    Parameters
    ----------
    f_report : dict
        The first report dictionary.
    i_report : dict
        The second report dictionary.
    prio : bool, optional
        Flag to determine which report takes priority in case of conflict, default is True.

    Returns
    -------
    dict
        The merged report dictionary.

    Notes
    -----
    - If both reports are dictionaries, merges them recursively.
    - If there is a conflict and prio is True, the value from f_report is used; otherwise, the value from i_report is used.
    """
    try:
        if isinstance(f_report, dict) and isinstance(i_report, dict):
            key_list = tuple(f_report.keys())
            for key in key_list:
                if key in i_report:
                    f_report[key] = merge_report(f_report[key], i_report[key], prio)
                    i_report.pop(key)
            f_report.update(i_report)
        elif f_report != i_report:
            return f_report if prio else i_report
    except Exception as e:
        qoa_logger.error(f"Error {type(e)} in merge_report: {e.__traceback__}")
        traceback.print_exception(*sys.exc_info())
    return f_report


def get_dict_at(dictionary: dict, i: int = 0):
    """
    Retrieve the key-value pair at a specific index in a dictionary.

    Parameters
    ----------
    dictionary : dict
        The dictionary from which to retrieve the key-value pair.
    i : int, optional
        The index of the key-value pair to retrieve, default is 0.

    Returns
    -------
    tuple
        A tuple containing the key and the value at the specified index.

    Raises
    ------
    IndexError
        If the index is out of range.

    Notes
    -----
    - Logs an error and prints the exception traceback if an error occurs.
    """
    try:
        keys = list(dictionary.keys())
        return keys[i], dictionary[keys[i]]
    except Exception as e:
        qoa_logger.error(f"Error {type(e)} in get_dict_at: {e.__traceback__}")
        traceback.print_exception(*sys.exc_info())


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
    affinity = os.sched_getaffinity(pid)
    return list(affinity)


def get_process_allowed_memory() -> Optional[float]:
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
    if CGROUP_VERSION == "v1":
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
                            return memory_limit_str
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
                        return memory_limit_str
            return None
