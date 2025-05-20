import os
import re

from qoa4ml.utils.logger import qoa_logger

DEFAULT_IGPU_PATH = "/sys/class/devfreq/"

MEMINFO_REG = re.compile(r"(?P<key>.+):\s+(?P<value>.+) (?P<unit>.)B")


def cat(path):
    with open(path) as f:
        return f.readline().rstrip("\x00")


def find_igpu():
    # Check if exist a integrated gpu
    # if not os.path.exists("/dev/nvhost-gpu") and not os.path.exists("/dev/nvhost-power-gpu"):
    #     return []
    igpu = {}
    if not os.path.isdir(DEFAULT_IGPU_PATH):
        qoa_logger.error(f"Folder {DEFAULT_IGPU_PATH} doesn't exist")
        return igpu
    for item in os.listdir(DEFAULT_IGPU_PATH):
        item_path = os.path.join(DEFAULT_IGPU_PATH, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            # Check name device
            name_path = f"{item_path}/device/of_node/name"
            if os.path.isfile(name_path):
                # Decode name
                name = cat(name_path)
                # Check if gpu
                if name in ["gv11b", "gp10b", "ga10b", "gpu"]:
                    # Extract real path GPU device
                    path = os.path.realpath(os.path.join(item_path, "device"))
                    frq_path = os.path.realpath(item_path)
                    igpu[name] = {
                        "type": "integrated",
                        "path": path,
                        "frq_path": frq_path,
                    }
                    qoa_logger.info(f'GPU "{name}" status in {path}')
                    qoa_logger.info(f'GPU "{name}" frq in {frq_path}')
                    path_railgate = os.path.join(path, "railgate_enable")
                    if os.path.isfile(path_railgate):
                        igpu[name]["railgate"] = path_railgate
                    path_3d_scaling = os.path.join(path, "enable_3d_scaling")
                    if os.path.isfile(path_3d_scaling):
                        igpu[name]["3d_scaling"] = path_3d_scaling
                else:
                    qoa_logger.debug(f"Skipped {name}")
    return igpu


def find_dgpu():
    pass


def get_gpu_load(gpu_list):
    gpu_load = {}
    # Read iGPU frequency
    for name, data in gpu_list.items():
        # Initialize GPU status
        gpu = {"type": data["type"]}
        if gpu["type"] == "integrated":
            if os.access(data["path"] + "/load", os.R_OK):
                with open(data["path"] + "/load") as f:
                    gpu["load"] = float(f.read()) / 10.0
        elif gpu["type"] == "discrete":
            qoa_logger.info("TODO discrete GPU")
        gpu_load[name] = gpu
    return gpu_load


def meminfo():
    # Read meminfo and decode
    # https://access.redhat.com/solutions/406773
    status_mem = {}
    with open("/proc/meminfo") as fp:
        for line in fp:
            # Search line
            match = re.search(MEMINFO_REG, line.strip())
            if match:
                parsed_line = match.groupdict()
                status_mem[parsed_line["key"]] = int(parsed_line["value"])
    return status_mem


def get_memory_status(mem_total):
    memory = {}
    status_mem = meminfo()
    # NOTE: Read memory use
    # NvMapMemUsed: Is the shared memory between CPU and GPU
    # This key is always available on Jetson (not really always)
    ram_shared = status_mem.get("NvMapMemUsed", 0)
    if mem_total:
        ram_shared = mem_total if ram_shared == 0 else ram_shared
    ram_total = status_mem.get("MemTotal", 0)
    ram_free = status_mem.get("MemFree", 0)
    ram_buffer = status_mem.get("Buffers", 0)
    ram_cached = status_mem.get("Cached", 0)
    ram_sreclaimable = status_mem.get("SReclaimable", 0)
    total_used_memory = ram_total - ram_free
    cached_memory = ram_cached + ram_sreclaimable  # + ram_Shmem
    memory["RAM"] = {
        "tot": ram_total,
        "used": total_used_memory - (ram_buffer + ram_cached),
        "free": ram_free,
        "buffers": ram_buffer,
        "cached": cached_memory,
        "shared": ram_shared,
    }
    return memory
