
import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Mocking dependencies that might be missing in some environments
for mod in ['numpy', 'imageio', 'matplotlib', 'matplotlib.pyplot', 'seaborn', 'torch', 'umap', 'matplotlib.patheffects', 'scipy', 'scipy.ndimage', 'tqdm', 'tqdm.auto']:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import numpy as np
# Ensure numpy.ndarray is available for isinstance checks if it was mocked
class RealMockArray:
    pass
np.ndarray = RealMockArray

from tasks.image_classification.plotting import save_frames_to_mp4

class TestSaveFramesSecurity(unittest.TestCase):
    def setUp(self):
        # Create a mock frame that passes validation
        self.frame = RealMockArray()
        self.frame.shape = (100, 100, 3)
        self.frame.dtype = 'uint8'
        self.frames = [self.frame]

    @patch('subprocess.Popen')
    def test_option_injection_prevented(self, mock_popen):
        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stderr = MagicMock()
        mock_process.wait.return_value = 0
        mock_process.communicate.return_value = (b'', b'')
        mock_popen.return_value = mock_process

        output_filename = "-version"
        save_frames_to_mp4(self.frames, output_filename)

        args, kwargs = mock_popen.call_args
        command = args[0]
        self.assertEqual(command[-1], "./-version")

    @patch('subprocess.Popen')
    def test_null_byte_raises_value_error(self, mock_popen):
        output_filename = "test.mp4\0.sh"
        with self.assertRaises(ValueError) as cm:
            save_frames_to_mp4(self.frames, output_filename)
        self.assertIn("null bytes", str(cm.exception))
        self.assertFalse(mock_popen.called)

    @patch('subprocess.Popen')
    def test_invalid_type_raises_type_error(self, mock_popen):
        output_filename = 123
        with self.assertRaises(TypeError) as cm:
            save_frames_to_mp4(self.frames, output_filename)
        self.assertIn("string or path-like", str(cm.exception))
        self.assertFalse(mock_popen.called)

    @patch('subprocess.Popen')
    def test_absolute_path_with_hyphen(self, mock_popen):
        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stderr = MagicMock()
        mock_process.wait.return_value = 0
        mock_process.communicate.return_value = (b'', b'')
        mock_popen.return_value = mock_process

        output_filename = "/tmp/-safe-name.mp4"
        save_frames_to_mp4(self.frames, output_filename)

        args, kwargs = mock_popen.call_args
        command = args[0]
        self.assertEqual(command[-1], "/tmp/-safe-name.mp4")

if __name__ == '__main__':
    unittest.main()
