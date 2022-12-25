import os
import random

import numpy as np
import torch

from utils.logger import print_msg


def set_seed(random_seed):
    print_msg("Setting Seed....", "INFO")
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)


def print_config(conf):
    print_msg("Printing Configuration....", "INFO")
    print("=" * 60)
    print("CONFIGURATION")
    print(f"MODEL NAME    | {conf.model_name}")
    print(f"BATCH SIZE    | {conf.batch_size}")
    print(f"MAX EPOCH     | {conf.max_epoch}")
    print(f"SHUFFLE       | {conf.shuffle}")
    print(f"LEARNING RATE | {conf.lr}")
    print(f"SEED          | {conf.seed}")
    print("=" * 60)


def setdir(dirpath, dirname=None, reset=True):
    from shutil import rmtree

    filepath = os.path.join(dirpath, dirname) if dirname else dirpath
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    elif reset:
        print_msg(f"reset directory : {dirname}", "INFO")
        rmtree(filepath)
        os.mkdir(filepath)
    return filepath


def make_file_name(model_name, format, version="v0"):
    file_name = f"{model_name}_{version}.{format}"
    return file_name
