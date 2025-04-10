from dataclasses import fields, is_dataclass
import os
import random
from pathlib import Path
from typing import Any

import casadi as ca
import numpy as np
import torch


def SX_to_labels(SX: ca.SX) -> list[str]:
    return SX.str().strip("[]").split(", ")


def find_idx_for_labels(sub_vars: ca.SX, sub_label: str) -> list[int]:
    """Return a list of indices where sub_label is part of the variable label."""
    return [
        idx
        for idx, label in enumerate(sub_vars.str().strip("[]").split(", "))
        if sub_label in label
    ]


def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)



def tensor_to_numpy(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()


def add_prefix_extend(prefix: str, extended: dict, extending: dict) -> None:
    """
    Add a prefix to the keys of a dictionary and extend the with other dictionary with the result.
    Raises a ValueError if a key that has been extended with a prefix is already in the extended dict.
    """
    for k, v in extending.items():
        if extended.get(prefix + k, None) is not None:
            raise ValueError(f"Key {prefix + k} already exists in the dictionary.")
        extended[prefix + k] = v


def set_seed(seed: int):
    """Set the seed for all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def update_dataclass_from_dict(dataclass_instance, update_dict):
    """Recursively update a dataclass instance with values from a dictionary."""
    for field in fields(dataclass_instance):
        # Check if the field is present in the update dictionary
        if field.name in update_dict:
            # If the field is a dataclass itself, recursively update it
            if is_dataclass(getattr(dataclass_instance, field.name)):
                update_dataclass_from_dict(getattr(dataclass_instance, field.name), update_dict[field.name])
            else:
                # Otherwise, directly update the field
                setattr(dataclass_instance, field.name, update_dict[field.name])


def log_git_hash_and_diff(filename: Path):
    """Log the git hash and diff of the current commit to a file."""
    try:
        git_hash = (
            os.popen("git rev-parse HEAD").read().strip()
            if os.path.exists(".git")
            else "No git repository"
        )
        git_diff = (
            os.popen("git diff").read().strip()
            if os.path.exists(".git")
            else "No git repository"
        )

        with open(filename, "w") as f:
            f.write(f"Git hash: {git_hash}\n")
            f.write(f"Git diff:\n{git_diff}\n")
    except Exception as e:
        print(f"Error logging git hash and diff: {e}")

