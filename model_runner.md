# Source Code

```python
# -*- coding: utf-8 -*-
"""
ModelRunnerBase 抽象基类

功能：
- 提供 Worker 调用模型的接口
- 抽象模型加载、执行和 profile 流程
- 支持不同设备（CPU/GPU）和配置

注意：
- 该类为抽象类，不能直接实例化
- 所有抽象方法必须由子类实现
"""

from abc import ABC, abstractmethod  # 导入抽象基类模块
from paddle import nn  # 导入 PaddlePaddle 的神经网络模块

from fastdeploy.config import FDConfig  # 导入 FastDeploy 配置类
from fastdeploy.utils import get_logger  # 导入日志工具
from fastdeploy.worker.output import ModelRunnerOutput  # 导入模型输出数据结构

# 初始化日志记录器，日志会保存到 model_runner_base.log 文件
logger = get_logger("model_runner_base", "model_runner_base.log")


class ModelRunnerBase(ABC):
    """
    ModelRunnerBase 抽象类

    功能说明：
    1. 抽象模型执行逻辑，包括输入准备、token 生成和处理
    2. 提供统一接口供 Worker 调用
    3. 子类需实现模型加载、获取和执行方法
    """

    def __init__(self, fd_config: FDConfig, device: str) -> None:
        """
        构造函数

        参数：
        - fd_config: FDConfig 实例，包含模型、设备和运行配置
        - device: 字符串，指定运行设备（如 'cpu', 'gpu:0'）

        功能：
        1. 保存 FDConfig
        2. 将 FDConfig 中的各类子配置保存到实例属性
        3. 保存指定设备
        """
        self.fd_config = fd_config  # 保存整个 FDConfig
        self.model_config = fd_config.model_config  # 模型相关配置
        self.load_config = fd_config.load_config  # 模型加载相关配置
        self.device_config = fd_config.device_config  # 设备相关配置
        self.speculative_config = fd_config.speculative_config  # 推理策略配置
        self.parallel_config = fd_config.parallel_config  # 并行策略配置
        self.graph_opt_config = fd_config.graph_opt_config  # 图优化配置
        self.quant_config = fd_config.quant_config  # 量化配置
        self.cache_config = fd_config.cache_config  # 缓存相关配置

        self.device = device  # 保存指定运行设备

    @abstractmethod
    def load_model(self) -> None:
        """
        抽象方法：加载模型

        功能：
        - 从本地或远程路径加载模型
        - 可根据不同后端实现不同加载逻辑

        注意：
        - 子类必须实现
        """
        raise NotImplementedError  # 抽象方法未实现，直接抛出异常

    @abstractmethod
    def get_model(self) -> nn.Layer:
        """
        抽象方法：获取模型实例

        返回：
        - nn.Layer 类型的 PaddlePaddle 模型

        功能：
        - 子类实现时需返回当前模型对象
        """
        raise NotImplementedError

    @abstractmethod
    def execute_model(self) -> ModelRunnerOutput:
        """
        抽象方法：执行模型

        返回：
        - ModelRunnerOutput 实例，包含模型输出结果

        功能：
        - 执行模型推理
        - 可以包含输入准备、token 生成、logits 计算等逻辑
        """
        raise NotImplementedError

    @abstractmethod
    def profile_run(self) -> None:
        """
        抽象方法：模型性能分析

        功能：
        - 使用虚拟输入执行一次前向推理
        - 用于分析显存/内存占用
        - 可用于自动调优或调试

        注意：
        - 子类必须实现
        """
        raise NotImplementedError

```

# 8.24 First Try

```python
# tests/worker/test_model_runner.py
import unittest
from types import SimpleNamespace

from fastdeploy.worker.model_runner_base import ModelRunnerBase
from fastdeploy.worker.output import ModelRunnerOutput

# ===========================
# Mock FDConfig for Unit Test
# ===========================
class MockFDConfig:
    """A fake FDConfig used for unit testing ModelRunner without real FDConfig dependencies."""
    def __init__(self):
        self.model_config = SimpleNamespace()
        self.load_config = SimpleNamespace()
        self.device_config = SimpleNamespace()
        self.speculative_config = SimpleNamespace()
        self.parallel_config = SimpleNamespace()
        self.graph_opt_config = SimpleNamespace(cudagraph_capture_sizes=[])
        self.quant_config = SimpleNamespace()
        self.cache_config = SimpleNamespace()


# ===========================
# Mock ModelRunner for Unit Test
# ===========================
class MockModelRunner(ModelRunnerBase):
    """A mock ModelRunner returning fake data for testing purposes."""

    def load_model(self):
        """Simulate loading a model."""
        self._model = "mock_model"
        return self._model

    def get_model(self):
        """Return the loaded mock model."""
        return getattr(self, "_model", None)

    def execute_model(self, batch_size=4, prompt_tokens=6, decode_tokens=8):
        """
        Simulate model execution and return fake ModelRunnerOutput.
        
        Args:
            batch_size (int): number of requests in this batch
            prompt_tokens (int): number of prompt tokens (unused, just for interface)
            decode_tokens (int): number of tokens to decode per request

        Returns:
            ModelRunnerOutput: fake output object
        """
        req_ids = [f"req_{i}" for i in range(batch_size)]
        req_id_to_index = {req_id: i for i, req_id in enumerate(req_ids)}
        sampled_token_ids = [[i for i in range(decode_tokens)] for _ in range(batch_size)]
        spec_token_ids = [[-1 for _ in range(decode_tokens)] for _ in range(batch_size)]

        output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
            sampled_token_ids=sampled_token_ids,
            spec_token_ids=spec_token_ids
        )

        # Add additional fields for testing
        output.generated_ids = sampled_token_ids
        output.logits = [[0.1 * i for i in range(decode_tokens)] for _ in range(batch_size)]

        return output

    def profile_run(self):
        """Return mock profile data."""
        return {"memory": "fake_memory_usage"}


# ===========================
# Unit Tests for ModelRunner
# ===========================
class TestMockModelRunner(unittest.TestCase):

    def setUp(self):
        """Set up a MockModelRunner with a fake FDConfig."""
        self.fd_config = MockFDConfig()
        self.runner = MockModelRunner(fd_config=self.fd_config, device="cpu")
        self.runner.load_model()

    def test_get_model_returns_model(self):
        """Test that get_model returns the loaded model."""
        model = self.runner.get_model()
        self.assertEqual(model, "mock_model")

    def test_execute_model_output_dimensions(self):
        """Test that execute_model returns output of correct batch and token dimensions."""
        batch_size = 4
        prompt_tokens = 6  # unused, just for API consistency
        decode_tokens = 8
        output = self.runner.execute_model(
            batch_size=batch_size,
            prompt_tokens=prompt_tokens,
            decode_tokens=decode_tokens
        )
        # Check batch size
        self.assertEqual(len(output.generated_ids), batch_size)
        self.assertEqual(len(output.logits), batch_size)
        # Check decode token size
        self.assertEqual(len(output.generated_ids[0]), decode_tokens)
        self.assertEqual(len(output.logits[0]), decode_tokens)

    def test_profile_run_returns_memory_info(self):
        """
        Test that profile_run returns memory info.
        """
        profile_info = self.runner.profile_run()
        self.assertIn("memory", profile_info)


if __name__ == "__main__":
    unittest.main()

```

```
/opt/conda/envs/python35-paddle120-env/lib/python3.10/site-packages/paddle/utils/cpp_extension/extension_utils.py:717: UserWarning: No ccache found. Please be aware that recompiling all source files may be required. You can download and install ccache from: https://github.com/ccache/ccache/blob/master/doc/INSTALL.md
  warnings.warn(warning_message)
/opt/conda/envs/python35-paddle120-env/lib/python3.10/site-packages/_distutils_hack/__init__.py:30: UserWarning: Setuptools is replacing distutils. Support for replacing an already imported distutils is deprecated. In the future, this condition will fail. Register concerns at https://github.com/pypa/setuptools/issues/new?template=distutils-deprecation.yml
  warnings.warn(
...
----------------------------------------------------------------------
Ran 3 tests in 0.005s

OK
```

# 8.27 Suggestion: dummy_run

It is recommended to refer to the `dummy run` method of `gpu_model_runner`, which constructs inputs of a specific length. The corresponding source code is as follows.

```python
	    def _dummy_run(
        self,
        num_tokens: paddle.Tensor,
        batch_size: paddle.Tensor,
        expected_decode_len: int = 1,
        in_capturing: bool = False,
    ) -> paddle.Tensor:
        """
        Use dummy inputs to run before formal execution.
        Args:
            num_tokens:
            expected_decode_len: Expected number of tokens generated
            in_capturing: Is cuda graph in capturing state
        """
        self._dummy_prefill_inputs(
            num_tokens=num_tokens,
            batch_size=batch_size,
            expected_decode_len=expected_decode_len,
        )
        if self.speculative_method in ["mtp"]:
            self.proposer.dummy_prefill_inputs(
                num_tokens=num_tokens,
                batch_size=batch_size,
                expected_decode_len=expected_decode_len,
            )
        while True:

            # 1. Initialize forward meta and attention meta data
            self._prepare_inputs()

            # 2. Padding inputs for cuda graph
            self.forward_meta.step_use_cudagraph = in_capturing and self.forward_meta.step_use_cudagraph
            self.padding_cudagraph_inputs()

            # 3. Run model
            if self.enable_mm:
                model_output = self.model(
                    self.share_inputs["ids_remove_padding"],
                    self.share_inputs["image_features"],
                    self.forward_meta,
                )
                hidden_states = model_output
            else:
                model_output = self.model(
                    ids_remove_padding=self.share_inputs["ids_remove_padding"],
                    forward_meta=self.forward_meta,
                )

                hidden_states = rebuild_padding(
                    model_output,
                    self.share_inputs["cu_seqlens_q"],
                    self.share_inputs["seq_lens_this_time"],
                    self.share_inputs["seq_lens_decoder"],
                    self.share_inputs["seq_lens_encoder"],
                    (
                        self.share_inputs["output_padding_offset"] if self.speculative_decoding else None
                    ),  # speculative decoding requires
                    self.parallel_config.max_model_len,
                )

            # 4. Execute spec decode
            logits = self.model.compute_logits(hidden_states)

            if not self.speculative_decoding:
                set_value_by_flags_and_idx(
                    self.share_inputs["pre_ids"],
                    self.share_inputs["input_ids"],
                    self.share_inputs["seq_lens_this_time"],
                    self.share_inputs["seq_lens_encoder"],
                    self.share_inputs["seq_lens_decoder"],
                    self.share_inputs["step_idx"],
                    self.share_inputs["stop_flags"],
                )
                sampler_output = self.sampler(logits, self.sampling_metadata)
                if self.parallel_config.tensor_parallel_size > 1:
                    paddle.distributed.broadcast(
                        sampler_output.sampled_token_ids,
                        self.parallel_config.data_parallel_rank * self.parallel_config.tensor_parallel_size,
                        group=self.parallel_config.tp_group,
                    )
            else:
                self.sampler(
                    logits,
                    self.sampling_metadata,
                    self.parallel_config.max_model_len,
                    self.share_inputs,
                )
                sampler_output = None
                if self.parallel_config.tensor_parallel_size > 1:
                    paddle.distributed.broadcast(
                        self.share_inputs["accept_tokens"],
                        self.parallel_config.data_parallel_rank * self.parallel_config.tensor_parallel_size,
                        group=self.parallel_config.tp_group,
                    )
                    paddle.distributed.broadcast(
                        self.share_inputs["accept_num"],
                        self.parallel_config.data_parallel_rank * self.parallel_config.tensor_parallel_size,
                        group=self.parallel_config.tp_group,
                    )
                    paddle.distributed.broadcast(
                        self.share_inputs["step_idx"],
                        self.parallel_config.data_parallel_rank * self.parallel_config.tensor_parallel_size,
                        group=self.parallel_config.tp_group,
                    )
                    paddle.distributed.broadcast(
                        self.share_inputs["stop_flags"],
                        self.parallel_config.data_parallel_rank * self.parallel_config.tensor_parallel_size,
                        group=self.parallel_config.tp_group,
                    )

            # 5. post process
            model_output_data = ModelOutputData(
                next_tokens=self.share_inputs["next_tokens"],
                stop_flags=self.share_inputs["stop_flags"],
                step_idx=self.share_inputs["step_idx"],
                max_dec_len=self.share_inputs["max_dec_len"],
                pre_ids=self.share_inputs["pre_ids"],
                seq_lens_this_time=self.share_inputs["seq_lens_this_time"],
                eos_token_id=self.share_inputs["eos_token_id"],
                not_need_stop=self.share_inputs["not_need_stop"],
                input_ids=self.share_inputs["input_ids"],
                stop_nums=self.share_inputs["stop_nums"],
                seq_lens_encoder=self.share_inputs["seq_lens_encoder"],
                seq_lens_decoder=self.share_inputs["seq_lens_decoder"],
                is_block_step=self.share_inputs["is_block_step"],
                full_hidden_states=model_output,
                msg_queue_id=self.parallel_config.msg_queue_id,
                mp_rank=self.parallel_config.tensor_parallel_rank,
                use_ep=self.parallel_config.use_ep,
                draft_tokens=(self.share_inputs["draft_tokens"] if self.speculative_decoding else None),
                actual_draft_token_num=(
                    self.share_inputs["actual_draft_token_num"] if self.speculative_decoding else None
                ),
                accept_tokens=(self.share_inputs["accept_tokens"] if self.speculative_decoding else None),
                accept_num=(self.share_inputs["accept_num"] if self.speculative_decoding else None),
                enable_thinking=(self.share_inputs["enable_thinking"] if self.enable_mm else None),
                think_end_id=(self.model_config.think_end_id if self.enable_mm else -1),
                need_think_end=(self.share_inputs["need_think_end"] if self.enable_mm else None),
                reasoning_index=(self.share_inputs["reasoning_index"] if self.enable_mm else None),
                stop_token_ids=self.share_inputs["stop_seqs"],
                stop_seqs_len=self.share_inputs["stop_seqs_len"],
            )

            post_process(
                sampler_output=sampler_output,
                model_output=model_output_data,
                share_inputs=self.share_inputs,
                block_size=self.cache_config.block_size,
                speculative_decoding=self.speculative_decoding,
                skip_save_output=True,
            )

            if self.speculative_decoding:
                if self.speculative_method == "mtp":
                    self.proposer.run(full_hidden_states=model_output)
                else:
                    self.proposer.run(share_inputs=self.share_inputs)

            # 7. Updata 'infer_seed' and step_cuda()
            self.share_inputs["infer_seed"].add_(self.infer_seed_increment)
            self.share_inputs["infer_seed"][:] %= self.MAX_INFER_SEED
            step_cuda(
                self.share_inputs,
                self.cache_config.block_size,
                self.cache_config.enc_dec_block_num,
                self.speculative_config,
                self.cache_config.enable_prefix_caching,
            )

            if int((self.share_inputs["seq_lens_this_time"] > 0).sum()) == 0:
                break

```

# 8.27 Try fix

```python
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unittest
from types import SimpleNamespace

from fastdeploy.worker.model_runner_base import ModelRunnerBase
from fastdeploy.worker.output import ModelRunnerOutput


class DummyFDConfig:
    """
    Dummy FDConfig for unit testing without real dependencies.
    """

    def __init__(self):
        self.model_config = SimpleNamespace()
        self.load_config = SimpleNamespace()
        self.device_config = SimpleNamespace()
        self.speculative_config = SimpleNamespace()
        self.parallel_config = SimpleNamespace()
        self.graph_opt_config = SimpleNamespace(cudagraph_capture_sizes=[])
        self.quant_config = SimpleNamespace()
        self.cache_config = SimpleNamespace()


class DummyModelRunner(ModelRunnerBase):
    """
    Dummy ModelRunner that generates fake outputs for testing.
    """

    def load_model(self):
        # Simulate loading a model by storing a dummy object
        self._dummy_model = "dummy_model"
        return self._dummy_model

    def get_model(self):
        # Return the dummy model
        return getattr(self, "_dummy_model", None)

    def _dummy_run(self, dummy_batch_size: int, dummy_prompt_tokens: int, dummy_decode_tokens: int):
        """
        Simulate the execution of a model using fake data.
        Args:
            dummy_batch_size: number of requests in the batch
            dummy_prompt_tokens: number of tokens in the prompt
            dummy_decode_tokens: number of tokens to decode
        """

        # Step 1. Build dummy prompt inputs (not used, only for shape consistency)
        dummy_prompt_ids = [[j for j in range(dummy_prompt_tokens)] for _ in range(dummy_batch_size)]

        # Step 2. Initialize structures for generated outputs
        dummy_generated_ids = [[] for _ in range(dummy_batch_size)]  # generated tokens
        dummy_logits = [[] for _ in range(dummy_batch_size)]         # logits per step

        # Step 3. Simulate decoding loop
        for step in range(dummy_decode_tokens):
            dummy_next_token = step
            dummy_step_logits = [0.1 * (i + step) for i in range(dummy_prompt_tokens)]
            for b in range(dummy_batch_size):
                dummy_generated_ids[b].append(dummy_next_token)
                dummy_logits[b].append(dummy_step_logits)

        # Step 4. Construct ModelRunnerOutput
        dummy_req_ids = [f"req_{i}" for i in range(dummy_batch_size)]
        dummy_req_id_to_index = {req_id: i for i, req_id in enumerate(dummy_req_ids)}
        dummy_spec_token_ids = [[-1 for _ in range(dummy_decode_tokens)] for _ in range(dummy_batch_size)]

        output = ModelRunnerOutput(
            req_ids=dummy_req_ids,
            req_id_to_index=dummy_req_id_to_index,
            sampled_token_ids=dummy_generated_ids,
            spec_token_ids=dummy_spec_token_ids,
        )
        # Attach additional fields for testing
        output.generated_ids = dummy_generated_ids
        output.logits = dummy_logits

        return output

    def execute_model(self, batch_size=4, prompt_tokens=6, decode_tokens=8):
        """
        Public entry point: run the dummy execution.
        """
        return self._dummy_run(batch_size, prompt_tokens, decode_tokens)

    def profile_run(self):
        # Return fake profiling info
        return {"memory": "dummy_memory_usage"}


class TestDummyModelRunner(unittest.TestCase):

    def setUp(self):
        # Initialize DummyModelRunner with a dummy FDConfig
        self.fd_config = DummyFDConfig()
        self.runner = DummyModelRunner(fd_config=self.fd_config, device="cpu")
        self.runner.load_model()

    def test_get_model_returns_model(self):
        # Ensure get_model returns the dummy model
        model = self.runner.get_model()
        self.assertEqual(model, "dummy_model")

    def test_execute_model_output_dimensions(self):
        # Ensure execute_model returns correct batch and token dimensions
        dummy_batch_size = 3
        dummy_prompt_tokens = 5
        dummy_decode_tokens = 7
        output = self.runner.execute_model(
            batch_size=dummy_batch_size,
            prompt_tokens=dummy_prompt_tokens,
            decode_tokens=dummy_decode_tokens,
        )
        self.assertEqual(len(output.generated_ids), dummy_batch_size)
        self.assertEqual(len(output.logits), dummy_batch_size)
        self.assertEqual(len(output.generated_ids[0]), dummy_decode_tokens)
        self.assertEqual(len(output.logits[0]), dummy_decode_tokens)

    def test_profile_run_returns_memory_info(self):
        # Ensure profile_run returns profiling info
        profile_info = self.runner.profile_run()
        self.assertIn("memory", profile_info)


if __name__ == "__main__":
    unittest.main()

```

