import pytest
from tasks.image_classification.plotting import save_frames_to_mp4
import numpy as np
import os

def test_save_frames_to_mp4_validation():
    # Dummy frames to bypass the initial check
    dummy_frames = [np.zeros((10, 10, 3), dtype=np.uint8)]
    output_filename = "dummy_output.mp4"

    # Test invalid fps
    with pytest.raises(ValueError, match="fps must be a positive number"):
        save_frames_to_mp4(dummy_frames, output_filename, fps=-1)

    with pytest.raises(ValueError, match="fps must be a positive number"):
        save_frames_to_mp4(dummy_frames, output_filename, fps="15")

    # Test invalid gop_size
    with pytest.raises(ValueError, match="gop_size must be a positive integer"):
        save_frames_to_mp4(dummy_frames, output_filename, gop_size=-5)

    with pytest.raises(ValueError, match="gop_size must be a positive integer"):
        save_frames_to_mp4(dummy_frames, output_filename, gop_size="10")

    # Test invalid crf
    with pytest.raises(ValueError, match="crf must be an integer between 0 and 51"):
        save_frames_to_mp4(dummy_frames, output_filename, crf=-1)

    with pytest.raises(ValueError, match="crf must be an integer between 0 and 51"):
        save_frames_to_mp4(dummy_frames, output_filename, crf=52)

    # Test invalid preset
    with pytest.raises(ValueError, match="preset must be one of"):
        save_frames_to_mp4(dummy_frames, output_filename, preset="invalid_preset")

    # Test invalid pix_fmt
    with pytest.raises(ValueError, match="Invalid pix_fmt"):
        save_frames_to_mp4(dummy_frames, output_filename, pix_fmt="yuv420p; rm -rf /")

    # Test valid inputs shouldn't raise ValueError at this stage
    # (they might fail later in subprocess if ffmpeg is missing, but that's okay for this unit test)
    try:
        # Avoid running actual subprocess by providing invalid first frame shape for our purpose or catching general exceptions later
        pass
    except Exception:
        pass
