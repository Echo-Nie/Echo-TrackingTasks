## 64 test_draft_model_flags

```python
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import draft_model_set_value_by_flags


class TestDraftModelSetValueByFlags(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")
        np.random.seed(42)

    def test_basic_update(self):
        """
        Test normal update behavior:
        batch0 performs a decoder step, batch1 performs an encoder step
        """
        bs = 2
        pre_id_length = 5
        draft_tokens = paddle.to_tensor([[10, 11, 12], [20, 21, 22]], dtype="int64")
        pre_ids_all = paddle.zeros([bs, pre_id_length], dtype="int64")
        stop_flags = paddle.to_tensor([False, False], dtype="bool")
        seq_lens_this_time = paddle.to_tensor([3, 1], dtype="int32")
        seq_lens_encoder = paddle.to_tensor([0, 0], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([0, 0], dtype="int32")
        step_idx = paddle.to_tensor([3, 1], dtype="int64")  # batch0 decoder, batch1 encoder

        """ Call custom op """
        draft_model_set_value_by_flags(
            draft_tokens, pre_ids_all, stop_flags, seq_lens_this_time, seq_lens_encoder, seq_lens_decoder, step_idx
        )

        """
        batch0: 3 tokens updated at decoder step
        batch1: 1 token updated at encoder step
        """
        expected = np.array([[0, 10, 11, 12, 0], [0, 20, 0, 0, 0]], dtype=np.int64)

        np.testing.assert_array_equal(pre_ids_all.numpy(), expected)
        np.testing.assert_array_equal(seq_lens_this_time.numpy(), [1, 1])

    def test_stop_flags(self):
        """
        batch0 is skipped (stop_flags=True), batch1 updates normally
        """
        bs = 2
        pre_id_length = 4
        draft_tokens = paddle.to_tensor([[5, 6], [7, 8]], dtype="int64")
        pre_ids_all = paddle.zeros([bs, pre_id_length], dtype="int64")
        stop_flags = paddle.to_tensor([True, False], dtype="bool")
        seq_lens_this_time = paddle.to_tensor([2, 2], dtype="int32")
        seq_lens_encoder = paddle.to_tensor([0, 0], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([0, 0], dtype="int32")
        step_idx = paddle.to_tensor([1, 2], dtype="int64")

        draft_model_set_value_by_flags(
            draft_tokens, pre_ids_all, stop_flags, seq_lens_this_time, seq_lens_encoder, seq_lens_decoder, step_idx
        )

        """
        batch0: no update due to stop flag
        batch1: 2 tokens updated at decoder step
        """
        expected = np.array([[0, 0, 0, 0], [0, 7, 8, 0]], dtype=np.int64)

        np.testing.assert_array_equal(pre_ids_all.numpy(), expected)
        np.testing.assert_array_equal(seq_lens_this_time.numpy(), [2, 1])


if __name__ == "__main__":
    unittest.main()

```



## 70 cpu_gpu_platforms

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



## 77 get_filtered_metrics

### Source code

```python
def get_filtered_metrics(exclude_names: Set[str], extra_register_func=None) -> str:
    """
    获取合并后的 metrics 文本（去除指定指标）
    Get the merged metrics text (excluding specified metrics)
    :param exclude_names: 要排除的指标名集合
    Set of metric names to be excluded
    :param extra_register_func: 可选参数，自定义指标注册函数
    Optional parameter, custom metric registration function
    :return: 过滤后的指标文本 (str)
    The filtered metrics text (str)
    """

    # 1. 创建一个基础的注册表（registry）
    # Create a basic registry
    # Prometheus's CollectorRegistry is the container for all metrics
    base_registry = CollectorRegistry()

    # 2. 从多进程环境中收集指标到 base_registry
    # Collect metrics from the multi-process environment into base_registry
    # 如果应用使用 Gunicorn/Uvicorn 多进程模式，
    # If the application uses Gunicorn/Uvicorn multi-process mode,
    # prometheus_client 的 multiprocess 模块会把每个 worker 的指标写到磁盘文件，
    # The multiprocess module of prometheus_client will write each worker's metrics to disk files,
    # MultiProcessCollector 会读取这些文件并合并进 base_registry
    # MultiProcessCollector will read these files and merge them into base_registry
    multiprocess.MultiProcessCollector(base_registry)

    # 3. 创建一个新的 registry，用来存放过滤后的指标
    # Create a new registry to store the filtered metrics
    filtered_registry = CollectorRegistry()

    # 4. 注册一个 SimpleCollector 到 filtered_registry
    # Register a SimpleCollector to filtered_registry
    # SimpleCollector will read metrics from base_registry,
    # and exclude the metrics specified in the exclude_names set
    filtered_registry.register(SimpleCollector(base_registry, exclude_names))

    # 5. 如果用户传入了额外的注册函数，就把它的指标也注册到 filtered_registry
    # If the user provides an additional registration function, register its metrics to filtered_registry as well
    # 这样可以在原始指标的基础上增加自定义指标
    # This allows adding custom metrics on top of the original metrics
    if extra_register_func:
        extra_register_func(filtered_registry)

    # 6. 生成最终的指标文本（Prometheus exposition 格式）
    # Generate the final metrics text (Prometheus exposition format)
    # generate_latest 会把 registry 里的所有指标导出为纯文本
    # generate_latest will export all metrics in the registry as plain text
    return generate_latest(filtered_registry).decode("utf-8")
```



### Done

```python
"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import unittest
from unittest.mock import patch

from prometheus_client import Gauge

from fastdeploy.metrics.metrics import get_filtered_metrics


class TestGetFilteredMetrics(unittest.TestCase):
    def test_filtered_and_custom_metrics(self):
        """
        Test get_filtered_metrics function:
        1. Exclude specific metrics from base_registry
        2. Keep other metrics in base_registry
        3. Ensure metrics registered by extra_register_func are effective
        """

        exclude_names = {"metric_to_exclude"}

        # Simulated metrics in base_registry (Gauge instances)
        g_keep = Gauge("metric_to_keep", "Kept metric")
        g_keep.set(1.23)

        g_exclude = Gauge("metric_to_exclude", "Excluded metric")
        g_exclude.set(99)

        # Fake MultiProcessCollector: register our simulated metrics
        def fake_multiprocess_collector(registry):
            registry.register(g_keep)
            registry.register(g_exclude)

        # Custom metric via extra_register_func
        def extra_func(registry):
            g_custom = Gauge("custom_metric_total", "Custom metric")
            g_custom.set(42)
            registry.register(g_custom)

        with patch(
            "fastdeploy.metrics.metrics.multiprocess.MultiProcessCollector", side_effect=fake_multiprocess_collector
        ):
            result = get_filtered_metrics(exclude_names=exclude_names, extra_register_func=extra_func)

        print("==== result ====\n", result)

        # 1. Excluded metric should not appear
        self.assertNotIn("metric_to_exclude", result)

        # 2. Kept metric should appear
        self.assertIn("metric_to_keep", result)

        # 3. Custom metric should appear
        self.assertIn("custom_metric_total", result)


if __name__ == "__main__":
    unittest.main()
```

Done

## 84 reasoning

```python
"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import unittest

from fastdeploy.reasoning import ReasoningParser, ReasoningParserManager


class TestReasoningParser(ReasoningParser):
    def is_reasoning_end(self, input_ids):
        """
        Return True to simulate end of reasoning content.
        """
        return True

    def extract_content_ids(self, input_ids):
        """
        Return input_ids directly for testing.
        """
        return input_ids

    def extract_reasoning_content(self, model_output, request):
        """
        Used for testing non-streaming extraction.
        """
        return model_output, model_output

    def extract_reasoning_content_streaming(
        self, previous_text, current_text, delta_text, previous_token_ids, current_token_ids, delta_token_ids
    ):
        """
        Return None for streaming extraction; minimal implementation for testing.
        """
        return None


class TestReasoningParserManager(unittest.TestCase):
    """
    Unit tests for ReasoningParserManager functionality.
    """

    def setUp(self):
        """
        Save original registry to restore after each test.
        """
        self.original_parsers = ReasoningParserManager.reasoning_parsers.copy()

    def tearDown(self):
        """
        Restore original registry to avoid test pollution.
        """
        ReasoningParserManager.reasoning_parsers = self.original_parsers.copy()

    def test_register_and_get_parser(self):
        """
        Test that a parser can be registered and retrieved successfully.
        Verifies normal registration and retrieval functionality.
        """
        ReasoningParserManager.register_module(module=TestReasoningParser, name="test_parser", force=True)
        parser_cls = ReasoningParserManager.get_reasoning_parser("test_parser")
        self.assertIs(parser_cls, TestReasoningParser)

    def test_register_duplicate_without_force_raises(self):
        """
        Test that registering a parser with an existing name without force raises KeyError.
        Ensures duplicate registrations are handled correctly.
        """
        ReasoningParserManager.register_module(module=TestReasoningParser, name="test_parser2", force=True)
        with self.assertRaises(KeyError):
            ReasoningParserManager.register_module(module=TestReasoningParser, name="test_parser2", force=False)

    def test_register_non_subclass_raises(self):
        """
        Test that registering a class not inheriting from ReasoningParser raises TypeError.
        Ensures type safety for registered modules.
        """

        class NotParser:
            pass

        with self.assertRaises(TypeError):
            ReasoningParserManager.register_module(module=NotParser, name="not_parser")

    def test_get_unregistered_parser_raises(self):
        """
        Test that retrieving a parser that was not registered raises KeyError.
        Ensures get_reasoning_parser handles unknown names correctly.
        """
        with self.assertRaises(KeyError):
            ReasoningParserManager.get_reasoning_parser("nonexistent_parser")


if __name__ == "__main__":
    unittest.main()
```

