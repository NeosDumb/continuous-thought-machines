import numpy as np
import random
import torch


import os
import zipfile
import glob
import tempfile


def zip_python_code(output_filename):
    """
    Zips all .py files in the current repository and saves it to the
    specified output filename.

    Args:
        output_filename: The name of the output zip file.
                         Defaults to "python_code_backup.zip".
    """
    current_dir = os.path.realpath(os.getcwd())
    tmp_dir = os.path.realpath(tempfile.gettempdir())

    # Resolve the final absolute path
    real_resolved = os.path.realpath(output_filename)

    # Check if the path safely resides within the current working directory or the temp directory
    is_in_current = (
        current_dir != "/"
        and os.path.commonpath([current_dir, real_resolved]) == current_dir
    )
    is_in_tmp = (
        tmp_dir != "/" and os.path.commonpath([tmp_dir, real_resolved]) == tmp_dir
    )

    if not (is_in_current or is_in_tmp):
        raise ValueError(
            "Output filename resolves outside of the allowed safe directories (current working directory or temp directory)."
        )

    with zipfile.ZipFile(output_filename, "w") as zipf:
        files = (
            glob.glob("models/**/*.py", recursive=True)
            + glob.glob("utils/**/*.py", recursive=True)
            + glob.glob("tasks/**/*.py", recursive=True)
            + glob.glob("*.py", recursive=True)
        )
        for file in files:
            root = "/".join(file.split("/")[:-1])
            nm = file.split("/")[-1]
            zipf.write(os.path.join(root, nm))


def set_seed(seed=42, deterministic=True):
    """
    ... and the answer is ...
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
