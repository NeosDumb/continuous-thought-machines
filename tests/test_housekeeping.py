import os
import zipfile
import random
import numpy as np
import torch
import pytest

from utils.housekeeping import zip_python_code, set_seed

def test_zip_python_code_traversal():
    """Test that zip_python_code prevents directory traversal characters."""
    error_msg = "Output filename resolves outside of the allowed safe directories"
    with pytest.raises(ValueError, match=error_msg):
        zip_python_code("../hacked_backup.zip")
    with pytest.raises(ValueError, match=error_msg):
        zip_python_code("../../hacked_backup.zip")
    with pytest.raises(ValueError, match=error_msg):
        zip_python_code("/etc/passwd.zip")

def test_zip_python_code(tmp_path):
    """Test that zip_python_code creates a zip file with expected .py files."""
    output_zip = tmp_path / "test_backup.zip"

    zip_python_code(str(output_zip))

    # Check if the file was created
    assert os.path.exists(str(output_zip))

    # Check the contents of the zip file
    with zipfile.ZipFile(str(output_zip), 'r') as zipf:
        zip_contents = zipf.namelist()

        assert len(zip_contents) > 0
        assert any(f.endswith('.py') for f in zip_contents)

        # utils/housekeeping.py must be in the zip file
        assert "utils/housekeeping.py" in zip_contents

def test_set_seed():
    """Test that set_seed makes random generation deterministic."""
    set_seed(42)

    rand_val1 = random.random()
    np_val1 = np.random.rand()
    torch_val1 = torch.rand(1).item()

    # Set a different seed to change the random state
    set_seed(99)

    rand_val2 = random.random()
    np_val2 = np.random.rand()
    torch_val2 = torch.rand(1).item()

    # Set the original seed again
    set_seed(42)

    rand_val3 = random.random()
    np_val3 = np.random.rand()
    torch_val3 = torch.rand(1).item()

    # Values should differ when seed is changed
    assert rand_val1 != rand_val2
    assert np_val1 != np_val2
    assert torch_val1 != torch_val2

    # Values should match when seed is restored
    assert rand_val1 == rand_val3
    assert np_val1 == np_val3
    assert torch_val1 == torch_val3

    # Check cudnn settings
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False

def test_set_seed_non_deterministic():
    """Test set_seed with deterministic=False."""
    set_seed(42, deterministic=False)
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is False
