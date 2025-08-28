# First Try - Done

```python
import unittest
import paddle
from unittest.mock import patch, MagicMock

from fastdeploy.platforms.base import _Backend
from fastdeploy.platforms.cpu import CPUPlatform, Platform
from fastdeploy.platforms.cuda import CUDAPlatform


class TestCPUPlatform(unittest.TestCase):
    """Unit tests for CPUPlatform class."""

    def setUp(self):
        """Initialize a CPUPlatform instance before each test."""
        self.cpu_platform = CPUPlatform()

    def test_is_cpu_and_available(self):
        """Test that CPU platform correctly identifies as CPU and is always available."""
        self.assertTrue(self.cpu_platform.is_cpu())
        self.assertTrue(self.cpu_platform.available())

    def test_attention_backend(self):
        """Test that CPU platform returns an empty string for attention backend by default."""
        self.assertEqual(self.cpu_platform.get_attention_backend_cls(None), "")

    def test_verify_quant(self):
        """Test that CPU platform validates quantization types correctly.

        - Supported quantization should not raise an error.
        - Unsupported quantization should raise ValueError.
        """
        CPUPlatform.supported_quantization = ["int8"]

        try:
            self.cpu_platform.verify_quant("int8")
        except ValueError:
            self.fail("verify_quant raised ValueError unexpectedly!")

        with self.assertRaises(ValueError):
            self.cpu_platform.verify_quant("fp32")

    def test_supports_fp8(self):
        """Test that CPU platform does not support FP8 precision."""
        self.assertFalse(self.cpu_platform.supports_fp8())


class TestCUDAPlatform(unittest.TestCase):
    """Unit tests for CUDAPlatform class."""

    def setUp(self):
        """Initialize a CUDAPlatform instance before each test."""
        self.platform = CUDAPlatform()

    @patch("paddle.is_compiled_with_cuda", return_value=True)
    @patch("paddle.device.get_device", return_value="gpu")
    def test_is_cuda(self, mock_get_device, mock_custom, mock_xpu, mock_rocm, mock_cuda):
        """Test that CUDA platform correctly identifies CUDA availability
        when compiled with CUDA and the current device is GPU.
        """
        self.assertTrue(self.platform.is_cuda())

    @patch("paddle.static.cuda_places", return_value=[0])
    def test_available_true(self, mock_cuda_places):
        """Test that CUDA platform is available when GPU devices are detected."""
        self.assertTrue(self.platform.available())

    @patch("paddle.static.cuda_places", side_effect=Exception("No GPU"))
    def test_available_false(self, mock_cuda_places):
        """Test that CUDA platform is not available when no GPU devices exist."""
        self.assertFalse(self.platform.available())

    def test_get_attention_backend_cls_valid(self):
        """Test that CUDA platform returns correct backend class names for valid backends."""
        mapping = {
            _Backend.NATIVE_ATTN: "fastdeploy.model_executor.layers.attention.PaddleNativeAttnBackend",
            _Backend.APPEND_ATTN: "fastdeploy.model_executor.layers.attention.AppendAttentionBackend",
            _Backend.MLA_ATTN: "fastdeploy.model_executor.layers.attention.MLAAttentionBackend",
            _Backend.FLASH_ATTN: "fastdeploy.model_executor.layers.attention.FlashAttentionBackend",
        }
        for backend, expected in mapping.items():
            result = CUDAPlatform.get_attention_backend_cls(backend)
            self.assertEqual(result, expected)

    def test_get_attention_backend_cls_invalid(self):
        """Test that CUDA platform raises ValueError for an invalid backend identifier."""
        with self.assertRaises(ValueError):
            CUDAPlatform.get_attention_backend_cls("INVALID")

    @patch("paddle.is_compiled_with_cuda", return_value=True)
    @patch("paddle.is_compiled_with_rocm", return_value=False)
    def test_is_cuda_true(self, mock_rocm, mock_cuda):
        """Test that CUDA platform is recognized as CUDA when compiled with CUDA
        and not compiled with ROCm.
        """
        plat = CUDAPlatform()
        self.assertTrue(plat.is_cuda())

    @patch("paddle.is_compiled_with_cuda", return_value=False)
    def test_is_cuda_false(self, mock_cuda):
        """Test that CUDA platform is not recognized as CUDA when CUDA is not compiled."""
        plat = CUDAPlatform()
        self.assertFalse(plat.is_cuda())


if __name__ == "__main__":
    unittest.main()

```

