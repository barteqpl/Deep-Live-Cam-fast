import os
import sys
import unittest
import numpy as np
from unittest.mock import MagicMock, patch

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import modules.globals
from modules.processors.frame.face_swapper import get_crop_mask, _CROP_MASK_CACHE, get_model_name
from modules.streamer import UDPStreamer

class TestNewFeatures(unittest.TestCase):
    def setUp(self):
        # Reset globals and cache
        modules.globals.swapper_model = "inswapper"
        modules.globals.execution_providers = ["CPUExecutionProvider"]
        _CROP_MASK_CACHE.clear()

    def test_crop_mask_cache(self):
        # 1. Test cache is initially empty
        self.assertEqual(len(_CROP_MASK_CACHE), 0)

        # 2. Test get_crop_mask returns correct shape and type
        mask_256 = get_crop_mask(256)
        self.assertIsInstance(mask_256, np.ndarray)
        self.assertEqual(mask_256.shape, (256, 256))
        self.assertEqual(mask_256.dtype, np.float32)

        # 3. Test caching mechanism
        self.assertIn(256, _CROP_MASK_CACHE)
        self.assertEqual(len(_CROP_MASK_CACHE), 1)

        # 4. Request same size, should return from cache
        mask_256_cached = get_crop_mask(256)
        self.assertIs(mask_256, mask_256_cached)

        # 5. Request different size, should increase cache size
        mask_128 = get_crop_mask(128)
        self.assertEqual(mask_128.shape, (128, 128))
        self.assertEqual(len(_CROP_MASK_CACHE), 2)

    def test_swapper_model_selection(self):
        # Test default
        modules.globals.swapper_model = "inswapper"
        self.assertEqual(get_model_name(), "inswapper_128_fp16.onnx")

        # Test simswap
        modules.globals.swapper_model = "simswap"
        self.assertEqual(get_model_name(), "simswap_256.onnx")

        # Test hififace
        modules.globals.swapper_model = "hififace"
        self.assertEqual(get_model_name(), "hififace_unofficial_256.onnx")

        # Test hyperswap
        modules.globals.swapper_model = "hyperswap"
        self.assertEqual(get_model_name(), "hyperswap_1a_256.onnx")

    @patch("subprocess.Popen")
    def test_udp_streamer(self, mock_popen):
        # Set up mock process
        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_popen.return_value = mock_process

        # Initialize streamer
        streamer = UDPStreamer(address="127.0.0.1:5000", width=640, height=480, fps=30.0)
        self.assertEqual(streamer.address, "127.0.0.1:5000")
        self.assertEqual(streamer.width, 640)
        self.assertEqual(streamer.height, 480)

        # Start streamer
        started = streamer.start()
        self.assertTrue(started)
        mock_popen.assert_called_once()

        # Write frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        streamer.write(dummy_frame)
        mock_process.stdin.write.assert_called()

        # Stop streamer
        streamer.stop()
        mock_process.stdin.close.assert_called_once()
        mock_process.terminate.assert_called_once()

    def test_argument_parsing(self):
        from modules.core import parse_args
        test_argv = [
            "run.py",
            "--disable-interpolation",
            "--interpolation-weight", "0.25",
            "--sharpness", "0.45",
            "--stream-udp", "6000"
        ]
        with patch("sys.argv", test_argv):
            with patch("modules.core.destroy"): # Avoid actually binding destroy signals
                parse_args()
                self.assertFalse(modules.globals.enable_interpolation)
                self.assertEqual(modules.globals.interpolation_weight, 0.25)
                self.assertEqual(modules.globals.sharpness, 0.45)
                self.assertEqual(modules.globals.stream_udp, "6000")

if __name__ == "__main__":
    unittest.main()
