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
from modules.processors.frame.face_swapper import get_crop_mask, _CROP_MASK_CACHE, get_model_name, get_margin_mask, _MARGIN_MASK_CACHE
from modules.streamer import UDPStreamer

class TestNewFeatures(unittest.TestCase):
    def setUp(self):
        # Reset globals and cache
        modules.globals.swapper_model = "inswapper"
        modules.globals.execution_providers = ["CPUExecutionProvider"]
        modules.globals.headless = True
        modules.globals.chin_blend_weight = 1.0
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
            "--chin-blend-weight", "0.75",
            "--stream-udp", "6000"
        ]
        with patch("sys.argv", test_argv):
            with patch("modules.core.destroy"): # Avoid actually binding destroy signals
                parse_args()
                self.assertFalse(modules.globals.enable_interpolation)
                self.assertEqual(modules.globals.interpolation_weight, 0.25)
                self.assertEqual(modules.globals.chin_blend_weight, 0.75)
                self.assertEqual(modules.globals.sharpness, 0.45)
                self.assertEqual(modules.globals.stream_udp, "6000")

    @patch("shutil.which")
    def test_pre_check(self, mock_which):
        from modules.core import pre_check
        
        # Test success (ffmpeg installed)
        mock_which.return_value = "/usr/bin/ffmpeg"
        with patch("sys.version_info", (3, 11, 0)):
            self.assertTrue(pre_check())
            
        # Test failure (ffmpeg missing)
        mock_which.return_value = None
        with patch("sys.version_info", (3, 11, 0)):
            self.assertFalse(pre_check())

    @patch("platform.system")
    def test_limit_resources_darwin(self, mock_system):
        from modules.core import limit_resources
        mock_system.return_value = "Darwin"
        
        modules.globals.max_memory = 4 # 4 GB
        
        # Patch resource module
        mock_resource = MagicMock()
        mock_resource.RLIM_INFINITY = -1
        mock_resource.RLIMIT_DATA = 2
        mock_resource.getrlimit.return_value = (1024, 2048)
        
        with patch.dict("sys.modules", {"resource": mock_resource}):
            limit_resources()
            mock_resource.getrlimit.assert_called_once_with(mock_resource.RLIMIT_DATA)
            # Should set to min(requested_memory, hard) -> min(4GB, 2048) -> 2048
            mock_resource.setrlimit.assert_called_once_with(mock_resource.RLIMIT_DATA, (2048, 2048))

    @patch("platform.system")
    def test_limit_resources_darwin_error_handling(self, mock_system):
        from modules.core import limit_resources
        mock_system.return_value = "Darwin"
        
        modules.globals.max_memory = 4 # 4 GB
        
        # Test that ValueError/OSError raised by setrlimit are caught and ignored
        mock_resource = MagicMock()
        mock_resource.RLIM_INFINITY = -1
        mock_resource.RLIMIT_DATA = 2
        mock_resource.getrlimit.return_value = (1024, 2048)
        mock_resource.setrlimit.side_effect = ValueError("current limit exceeds maximum limit")
        
        with patch.dict("sys.modules", {"resource": mock_resource}):
            # Should not raise ValueError
            try:
                limit_resources()
            except ValueError:
                self.fail("limit_resources() raised ValueError unexpectedly!")

    def test_margin_mask_cache(self):
        _MARGIN_MASK_CACHE.clear()
        self.assertEqual(len(_MARGIN_MASK_CACHE), 0)
        
        mask = get_margin_mask(256)
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(mask.shape, (256, 256))
        self.assertEqual(mask.dtype, np.float32)
        
        # Check boundary values are 0.0
        self.assertEqual(mask[0, 0], 0.0)
        self.assertEqual(mask[255, 255], 0.0)
        
        # Check center value is 1.0
        self.assertEqual(mask[128, 128], 1.0)
        
        self.assertIn(256, _MARGIN_MASK_CACHE)

if __name__ == "__main__":
    unittest.main()
