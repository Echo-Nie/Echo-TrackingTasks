## 25 fused_get_rotary_embedding

```python
import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import fused_get_rotary_embedding


def numpy_rope(position_ids, head_dim, prompt_num=0, seq_len=None):
    """Numpy reference implementation"""
    batch_size, max_seq_len = position_ids.shape
    if seq_len is None:
        seq_len = max_seq_len - prompt_num

    inv_head_dim = 1.0 / float(head_dim)
    rope_embedding = np.empty((2, batch_size, 1, seq_len, head_dim), dtype=np.float32)

    for b in range(batch_size):
        for s in range(seq_len):
            pos = position_ids[b, s + prompt_num]
            for h in range(0, head_dim, 2):
                exponent_factor = -float(h) * inv_head_dim
                inv_freq = np.power(10000.0, exponent_factor)
                val = pos * inv_freq
                cos_val, sin_val = np.cos(val), np.sin(val)
                rope_embedding[0, b, 0, s, h : h + 2] = cos_val
                rope_embedding[1, b, 0, s, h : h + 2] = sin_val
    return rope_embedding


class TestFusedGetRotaryEmbedding(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")
        np.random.seed(42)
        self.batch_size = 2
        self.seq_len = 4
        self.head_dim = 8

    def _run_and_check(self, batch_size, seq_len, head_dim, prompt_num=0):
        input_ids = paddle.randint(0, 100, [batch_size, seq_len], dtype="int32")
        position_ids = paddle.arange(seq_len + 2 * prompt_num).tile([batch_size, 1]).astype("float32")

        head_dim_tensor = paddle.arange(head_dim, dtype="int32")

        out = fused_get_rotary_embedding(input_ids, position_ids, head_dim_tensor, prompt_num)
        out_np = out.numpy()
        ref = numpy_rope(position_ids.numpy(), head_dim, prompt_num, seq_len=seq_len)

        # check shape
        expect_shape = (2, batch_size, 1, seq_len, head_dim)
        self.assertEqual(tuple(out.shape), expect_shape)

        # check values
        np.testing.assert_allclose(out_np, ref, rtol=1e-5, atol=1e-6)

    def test_basic_case(self):
        self._run_and_check(self.batch_size, self.seq_len, self.head_dim)

    def test_minimal_head_dim(self):
        self._run_and_check(batch_size=1, seq_len=2, head_dim=2)

    def test_with_prompt_num(self):
        self._run_and_check(self.batch_size, self.seq_len, self.head_dim, prompt_num=3)


if __name__ == "__main__":
    unittest.main()
```



## 34 get_position_ids_and_mask_encoder_batch

### Code

```python
#include "helper.h"
#include "paddle/extension.h"

// CUDA kernel: 为每个 batch 生成 position_ids 和 mask_encoder_batch
__global__ void GetPositionIdsAndMaskEncoderBatchKernel(
    const int* seq_lens_encoder,  // [bsz] 每个 batch 的 encoder 长度
    const int* seq_lens_decoder,  // [bsz] 每个 batch 的 decoder 长度
    const int* seq_lens_this_time, // [bsz] decoder 当前 step 的长度（可能小于 decoder 长度）
    int* position_ids,  // 输出的一维 position_ids
    int* mask_encoder_batch, // 输出 mask，encoder 部分为 1，decoder 部分为 0
    const int bsz) {  // batch size
  // 当前线程索引，每个线程处理一个 batch
  int tid = threadIdx.x;
  if (tid >= bsz) return;  // 防止线程越界

  // 动态计算当前 batch 的偏移量(offset)：
  // 因为输出是一维数组，每个 batch 的 position_ids 紧跟前一个 batch
  int offset = 0;
  for (int i = 0; i < tid; i++) {
    offset += seq_lens_encoder[i];         // 累加 encoder 长度
    if (seq_lens_decoder[i] > 0) {
      offset += seq_lens_this_time[i];     // 累加 decoder 当前 step 长度
    }
  }

  // 当前 batch 的 encoder/decoder 长度
  int encoder_len = seq_lens_encoder[tid];
  int decoder_len = seq_lens_decoder[tid];
  int seq_len_this_time = seq_lens_this_time[tid];

  // 写入 encoder 的 position_ids
  for (int i = 0; i < encoder_len; i++) {
    position_ids[offset + i] = i;          // encoder position 从 0 开始
    mask_encoder_batch[offset + i] = 1;    // encoder mask=1
  }
  offset += encoder_len;  // 偏移量增加 encoder 长度

  // 写入 decoder 的 position_ids（如果 decoder 长度 > 0）
  if (decoder_len > 0) {
    for (int i = 0; i < seq_len_this_time; i++) {
      position_ids[offset + i] = decoder_len + i;  // decoder position 从 decoder_len 开始
      mask_encoder_batch[offset + i] = 0;          // decoder mask=0
    }
  }
}

// CPU 调用 CUDA kernel 的包装函数
void GetPositionIdsAndMaskEncoderBatch(
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& position_ids,
    const paddle::Tensor& mask_encoder_batch) {

  const int bsz = seq_lens_this_time.shape()[0]; // batch size

  // Launch kernel，每个 batch 一个线程
  GetPositionIdsAndMaskEncoderBatchKernel<<<1, bsz, 0, position_ids.stream()>>>(
      seq_lens_encoder.data<int>(),
      seq_lens_decoder.data<int>(),
      seq_lens_this_time.data<int>(),
      const_cast<int*>(position_ids.data<int>()),
      const_cast<int*>(mask_encoder_batch.data<int>()),
      bsz);
}

// PaddlePaddle 静态注册算子
PD_BUILD_STATIC_OP(get_position_ids_and_mask_encoder_batch)
    .Inputs({"seq_lens_encoder",
             "seq_lens_decoder",
             "seq_lens_this_time",
             "position_ids",
             "mask_encoder_batch"})
    .Outputs({"position_ids_out", "mask_encoder_batch_out"})
    .SetInplaceMap({{"position_ids", "position_ids_out"},
                    {"mask_encoder_batch", "mask_encoder_batch_out"}})
    .SetKernelFn(PD_KERNEL(GetPositionIdsAndMaskEncoderBatch));

```



### Test

```python
import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import get_position_ids_and_mask_encoder_batch


class TestGetPositionIdsAndMaskEncoderBatch(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        paddle.set_device("gpu")

    def test_basic_functionality(self):
        # Test normal case with batch size 2
        seq_lens_encoder = paddle.to_tensor([3, 2], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([1, 2], dtype="int32")
        seq_lens_this_time = paddle.to_tensor([1, 2], dtype="int32")

        total_len = int(seq_lens_encoder.numpy().sum() + seq_lens_this_time.numpy().sum())
        position_ids = paddle.zeros([total_len], dtype="int32")
        mask_encoder_batch = paddle.zeros([total_len], dtype="int32")

        # Call the custom operator
        get_position_ids_and_mask_encoder_batch(
            seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, position_ids, mask_encoder_batch
        )

        expected_position_ids = np.array([0, 1, 2, 1, 0, 1, 2, 3], dtype=np.int32)

        expected_mask = np.array([1, 1, 1, 0, 1, 1, 0, 0], dtype=np.int32)

        # Convert to numpy for comparison
        position_ids_np = position_ids.numpy()
        mask_encoder_batch_np = mask_encoder_batch.numpy()

        # Assert equality
        np.testing.assert_array_equal(position_ids_np, expected_position_ids)
        np.testing.assert_array_equal(mask_encoder_batch_np, expected_mask)

    def test_empty_decoder(self):
        # Test case where decoder length is 0
        seq_lens_encoder = paddle.to_tensor([2], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([0], dtype="int32")
        seq_lens_this_time = paddle.to_tensor([0], dtype="int32")

        position_ids = paddle.zeros([2], dtype="int32")
        mask_encoder_batch = paddle.zeros([2], dtype="int32")

        get_position_ids_and_mask_encoder_batch(
            seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, position_ids, mask_encoder_batch
        )

        expected_position_ids = np.array([0, 1], dtype=np.int32)
        expected_mask = np.array([1, 1], dtype=np.int32)

        np.testing.assert_array_equal(position_ids.numpy(), expected_position_ids)
        np.testing.assert_array_equal(mask_encoder_batch.numpy(), expected_mask)


if __name__ == "__main__":
    unittest.main()
```



## 35 moe_redundant_topk_select

```python
import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import moe_redundant_topk_select


class TestMoERedundantTopKSelect(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")
        np.random.seed(42)

    def _run_and_check(
        self,
        gating_shape,
        expert_num,
        moe_topk,
        apply_norm_weight=False,
        enable_softmax_top_k_fused=False,
        use_bias=False,
    ):
        """Helper function to run the operator and check."""
        gating_logits = paddle.to_tensor(np.random.rand(*gating_shape).astype("float32"))
        expert_id_to_ep_rank_array = paddle.to_tensor(
            np.random.randint(0, expert_num, size=(expert_num,)).astype("int32")
        )
        expert_in_rank_num_list = paddle.to_tensor(np.random.randint(1, 4, size=(expert_num,)).astype("int32"))
        tokens_per_expert_stats_list = paddle.zeros([expert_num], dtype="int32")
        bias = None
        if use_bias:
            bias = paddle.to_tensor(np.random.rand(*gating_shape[:-1], expert_num).astype("float32"))

        outputs = moe_redundant_topk_select(
            gating_logits=gating_logits,
            expert_id_to_ep_rank_array=expert_id_to_ep_rank_array,
            expert_in_rank_num_list=expert_in_rank_num_list,
            tokens_per_expert_stats_list=tokens_per_expert_stats_list,
            bias=bias,
            moe_topk=moe_topk,
            apply_norm_weight=apply_norm_weight,
            enable_softmax_top_k_fused=enable_softmax_top_k_fused,
            redundant_ep_rank_num_plus_one=2,
        )

        topk_ids, topk_weights = outputs

        # Check shapes are correct
        expected_shape = [int(np.prod(gating_shape[:-1])), moe_topk]
        self.assertEqual(topk_ids.shape, expected_shape)
        self.assertEqual(topk_weights.shape, expected_shape)

        # Check topk_ids are non-negative
        self.assertTrue(np.all(topk_ids.numpy() >= 0))

        # Check topk weights are non-negative
        self.assertTrue(np.all(topk_weights.numpy() >= -1e-6))

        # Check tokens_per_expert_stats_list has valid values
        self.assertEqual(tokens_per_expert_stats_list.shape[0], expert_num)
        self.assertTrue(np.all(tokens_per_expert_stats_list.numpy() >= 0))

    def test_basic_case(self):
        self._run_and_check(gating_shape=(4, 16), expert_num=8, moe_topk=2)

    def test_3d_input_case(self):
        self._run_and_check(gating_shape=(2, 3, 8), expert_num=8, moe_topk=2)

    def test_with_bias(self):
        self._run_and_check(gating_shape=(3, 12), expert_num=4, moe_topk=2, use_bias=True)

    def test_with_norm_weight(self):
        self._run_and_check(gating_shape=(5, 10), expert_num=4, moe_topk=2, apply_norm_weight=True)

    def test_softmax_topk_fused(self):
        self._run_and_check(gating_shape=(6, 8), expert_num=8, moe_topk=2, enable_softmax_top_k_fused=True)


if __name__ == "__main__":
    unittest.main()
```



## 36 extract_text_token_output

```python
import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import extract_text_token_output


class TestExtractTextTokenOutput(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")
        np.random.seed(42)

    def _run_and_check(
        self,
        bsz,
        hidden_size,
        max_seq_len_v,
        max_seq_len_index_v,
        mm_token_num_len_v,
        seq_lens_this_time_v,
        cu_seqlens_q_v,
        hidden_states_v,
    ):

        max_seq_len = paddle.to_tensor([max_seq_len_v], dtype="int32")
        max_seq_len_index = paddle.to_tensor([max_seq_len_index_v], dtype="int32")
        mm_token_num_len = paddle.to_tensor([mm_token_num_len_v], dtype="int32")
        seq_lens_this_time = paddle.to_tensor(seq_lens_this_time_v, dtype="int32")
        cu_seqlens_q = paddle.to_tensor(cu_seqlens_q_v, dtype="int32")
        hidden_states = paddle.to_tensor(hidden_states_v, dtype="float32")

        out = extract_text_token_output(
            max_seq_len, max_seq_len_index, mm_token_num_len, seq_lens_this_time, cu_seqlens_q, hidden_states
        )[0]
        out_np = out.numpy()

        expect = np.ones((bsz, hidden_size), dtype="float32")
        for i in range(bsz):
            true_bsz = cu_seqlens_q_v[i + 1] - 1
            if (max_seq_len_v == mm_token_num_len_v) and (i == max_seq_len_index_v):
                expect[i, :] = 0.0
            else:
                if seq_lens_this_time_v[i] != 0:
                    expect[i, :] = hidden_states_v[true_bsz, :]

        if out_np.ndim == 1:
            np.testing.assert_allclose(out_np, expect[0], rtol=1e-5, atol=1e-5)
        else:
            np.testing.assert_allclose(out_np, expect, rtol=1e-5, atol=1e-5)

    def test_basic_case(self):
        bsz, hidden_size = 2, 4
        max_seq_len_v = 3
        max_seq_len_index_v = 0
        mm_token_num_len_v = 2
        seq_lens_this_time_v = [2, 1]
        cu_seqlens_q_v = [0, 2, 3]
        hidden_states_v = np.arange(12).reshape(3, 4).astype("float32")

        self._run_and_check(
            bsz,
            hidden_size,
            max_seq_len_v,
            max_seq_len_index_v,
            mm_token_num_len_v,
            seq_lens_this_time_v,
            cu_seqlens_q_v,
            hidden_states_v,
        )

    def test_zero_case(self):
        bsz, hidden_size = 2, 4
        max_seq_len_v = 5
        max_seq_len_index_v = 1
        mm_token_num_len_v = 5
        seq_lens_this_time_v = [1, 1]
        cu_seqlens_q_v = [0, 1, 2]
        hidden_states_v = np.random.randn(2, hidden_size).astype("float32")

        self._run_and_check(
            bsz,
            hidden_size,
            max_seq_len_v,
            max_seq_len_index_v,
            mm_token_num_len_v,
            seq_lens_this_time_v,
            cu_seqlens_q_v,
            hidden_states_v,
        )


if __name__ == "__main__":
    unittest.main()
```

## 37 top_k

### First Try size=1

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

from fastdeploy.model_executor.ops.gpu import top_k_renorm_probs


class TestTopKRenormProbs(unittest.TestCase):

    def setUp(self):
        paddle.set_device("gpu")
        np.random.seed(42)

    def test_basic_functionality(self):
        """Test the operator"""
        batch_size = 1
        vocab_size = 5
        # Construct a valid probability distribution
        probs = np.random.rand(batch_size, vocab_size).astype("float32")
        probs /= probs.sum(axis=1, keepdims=True)
        top_k = np.array([2], dtype="int64")

        probs_tensor = paddle.to_tensor(probs)
        top_k_tensor = paddle.to_tensor(top_k)

        # Call the operator
        renorm_probs = top_k_renorm_probs(probs_tensor, top_k_tensor)[0].numpy()
        renorm_probs = renorm_probs.reshape(batch_size, vocab_size)

        self.assertEqual(renorm_probs.shape, probs.shape)

        # Non top-k positions must be zero
        top_indices = np.argsort(probs[0])[::-1][: top_k[0]]
        for j in range(vocab_size):
            if j not in top_indices:
                self.assertAlmostEqual(renorm_probs[0, j], 0.0, places=6)

        self.assertAlmostEqual(renorm_probs[0].sum(), 1.0, places=6)

    def test_edge_cases(self):
        """Test the operator with edge-case top_k values"""
        probs = np.array([[0.1, 0.3, 0.4, 0.2]], dtype="float32")

        # Case 1: top_k = 1, only the max element remains
        top_k_tensor = paddle.to_tensor(np.array([1], dtype="int64"))
        renorm_probs = top_k_renorm_probs(paddle.to_tensor(probs), top_k_tensor)[0].numpy()
        renorm_probs = renorm_probs.reshape(1, -1)
        self.assertEqual((renorm_probs > 0).sum(), 1)  # Only one non-zero element
        self.assertAlmostEqual(renorm_probs.sum(), 1.0, places=6)

        # Case 2: top_k = vocab_size, output should match input
        top_k_tensor = paddle.to_tensor(np.array([4], dtype="int64"))
        renorm_probs = top_k_renorm_probs(paddle.to_tensor(probs), top_k_tensor)[0].numpy()
        renorm_probs = renorm_probs.reshape(1, -1)
        np.testing.assert_allclose(renorm_probs, probs, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

```

### add size=2,3

```python
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

import unittest
import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import top_k_renorm_probs


class TestTopKRenormProbs(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")
        np.random.seed(42)

    def _check_output(self, probs, top_k):
        probs_tensor = paddle.to_tensor(probs)
        top_k_tensor = paddle.to_tensor(top_k)
        renorm_probs = top_k_renorm_probs(probs_tensor, top_k_tensor).numpy()

        self.assertEqual(renorm_probs.shape, probs.shape)

        batch_size, vocab_size = probs.shape
        for b in range(batch_size):
            self.assertAlmostEqual(renorm_probs[b].sum(), 1.0, places=6)
            top_indices = np.argsort(probs[b])[::-1][: top_k[b]]
            for j in range(vocab_size):
                if j not in top_indices:
                    self.assertAlmostEqual(renorm_probs[b, j], 0.0, places=6)

    def test_single_batch_basic(self):
        """Test with batch_size = 1"""
        probs = np.random.rand(1, 5).astype("float32")
        probs /= probs.sum(axis=1, keepdims=True)
        top_k = np.array([2], dtype="int64")
        self._check_output(probs, top_k)

    def test_single_batch_edge_cases(self):
        """Test edge cases with batch_size = 1"""
        probs = np.array([[0.1, 0.3, 0.4, 0.2]], dtype="float32")

        # top_k = 1
        self._check_output(probs, np.array([1], dtype="int64"))

        # top_k = vocab_size
        renorm_probs = top_k_renorm_probs(
            paddle.to_tensor(probs),
            paddle.to_tensor(np.array([4], dtype="int64"))
        ).numpy()
        np.testing.assert_allclose(renorm_probs, probs, rtol=1e-6, atol=1e-6)

    def test_batch_size_two(self):
        """Test with batch_size = 2"""
        probs = np.random.rand(2, 5).astype("float32")
        probs /= probs.sum(axis=1, keepdims=True)
        top_k = np.array([2, 3], dtype="int64")
        self._check_output(probs, top_k)

    def test_batch_size_three(self):
        """Test with batch_size = 3"""
        probs = np.random.rand(3, 6).astype("float32")
        probs /= probs.sum(axis=1, keepdims=True)
        top_k = np.array([1, 2, 4], dtype="int64")
        self._check_output(probs, top_k)


if __name__ == "__main__":
    unittest.main()

```

### Print

```python
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

import unittest
import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import top_k_renorm_probs


class TestTopKRenormProbs(unittest.TestCase):
    """Unit tests for top_k_renorm_probs operator"""

    def setUp(self):
        paddle.set_device("gpu")
        np.random.seed(42)

    def _check_output(self, probs, top_k):
        """Helper to validate shape, normalization and masking"""
        print("\n=== Running check ===")
        print("Input probs shape:", probs.shape)
        print("Input top_k:", top_k)

        probs_tensor = paddle.to_tensor(probs)
        top_k_tensor = paddle.to_tensor(top_k)
        renorm_probs = top_k_renorm_probs(probs_tensor, top_k_tensor).numpy()

        print("Output shape:", renorm_probs.shape)
        print("Output probs:\n", renorm_probs)

        self.assertEqual(renorm_probs.shape, probs.shape)

        batch_size, vocab_size = probs.shape
        for b in range(batch_size):
            sum_val = renorm_probs[b].sum()
            print(f"Batch {b}: sum={sum_val:.6f}")
            self.assertAlmostEqual(sum_val, 1.0, places=6)

            top_indices = np.argsort(probs[b])[::-1][: top_k[b]]
            mask_non_topk = [j for j in range(vocab_size) if j not in top_indices]
            if mask_non_topk:
                print(f"Batch {b}: non-top-k indices {mask_non_topk} should be zero")

            for j in mask_non_topk:
                self.assertAlmostEqual(renorm_probs[b, j], 0.0, places=6)

    def test_single_batch_basic(self):
        """Test with batch_size = 1"""
        probs = np.random.rand(1, 5).astype("float32")
        probs /= probs.sum(axis=1, keepdims=True)
        top_k = np.array([2], dtype="int64")
        self._check_output(probs, top_k)

    def test_single_batch_edge_cases(self):
        """Test edge cases with batch_size = 1"""
        probs = np.array([[0.1, 0.3, 0.4, 0.2]], dtype="float32")

        # top_k = 1
        self._check_output(probs, np.array([1], dtype="int64"))

        # top_k = vocab_size
        renorm_probs = top_k_renorm_probs(
            paddle.to_tensor(probs),
            paddle.to_tensor(np.array([4], dtype="int64"))
        ).numpy()
        print("\nEdge case top_k=vocab_size")
        print("Input probs:\n", probs)
        print("Output probs:\n", renorm_probs)
        np.testing.assert_allclose(renorm_probs, probs, rtol=1e-6, atol=1e-6)

    def test_batch_size_two(self):
        """Test with batch_size = 2"""
        probs = np.random.rand(2, 5).astype("float32")
        probs /= probs.sum(axis=1, keepdims=True)
        top_k = np.array([2, 3], dtype="int64")
        self._check_output(probs, top_k)

    def test_batch_size_three(self):
        """Test with batch_size = 3"""
        probs = np.random.rand(3, 6).astype("float32")
        probs /= probs.sum(axis=1, keepdims=True)
        top_k = np.array([1, 2, 4], dtype="int64")
        self._check_output(probs, top_k)


if __name__ == "__main__":
    unittest.main()

```



### Output

```bash
=== Running check ===
Input probs shape: (3, 6)
Input top_k: [1 2 4]
W0914 10:01:09.158907  2341 gpu_resources.cc:114] Please NOTE: device: 0, GPU Compute Capability: 8.0, Driver API Version: 13.0, Runtime API Version: 12.8
Output shape: (3, 6)
Output probs:
 [[0.         1.         0.         0.         0.         0.        ]
 [0.         0.47175145 0.         0.         0.         0.5282486 ]
 [0.44425836 0.11332123 0.         0.         0.16236813 0.28005224]]
Batch 0: sum=1.000000
Batch 0: non-top-k indices [0, 2, 3, 4, 5] should be zero
Batch 1: sum=1.000000
Batch 1: non-top-k indices [0, 2, 3, 4] should be zero
Batch 2: sum=1.000000
Batch 2: non-top-k indices [2, 3] should be zero
.
=== Running check ===
Input probs shape: (2, 5)
Input top_k: [2 3]
Output shape: (2, 5)
Output probs:
 [[0.         0.5649906  0.4350094  0.         0.        ]
 [0.         0.         0.3981753  0.2763285  0.32549617]]
Batch 0: sum=1.000000
Batch 0: non-top-k indices [0, 3, 4] should be zero
Batch 1: sum=1.000000
Batch 1: non-top-k indices [0, 1] should be zero
.
=== Running check ===
Input probs shape: (1, 5)
Input top_k: [2]
Output shape: (1, 5)
Output probs:
 [[0.        0.5649906 0.4350094 0.        0.       ]]
Batch 0: sum=1.000000
Batch 0: non-top-k indices [0, 3, 4] should be zero
.
=== Running check ===
Input probs shape: (1, 4)
Input top_k: [1]
Output shape: (1, 4)
Output probs:
 [[0. 0. 1. 0.]]
Batch 0: sum=1.000000
Batch 0: non-top-k indices [0, 1, 3] should be zero

Edge case top_k=vocab_size
Input probs:
 [[0.1 0.3 0.4 0.2]]
Output probs:
 [[0.1 0.3 0.4 0.2]]
.
----------------------------------------------------------------------
Ran 4 tests in 0.197s

OK
```



## 49 pre_cache_len_concat

```python
import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import pre_cache_len_concat


def ref_pre_cache_len_concat(seq_lens_decoder, seq_lens_this_time, block_size):
    """
    Reference implementation.
    """
    bsz = len(seq_lens_this_time)
    cu_seqlens_k = np.zeros(bsz + 1, dtype=np.int32)
    batch_ids = []
    tile_ids_per_batch = []
    total_tokens = 0
    gridx = 0

    for bid in range(bsz):
        cache_len = int(seq_lens_decoder[bid])
        q_len = int(seq_lens_this_time[bid])
        if q_len <= 0:
            cache_len = 0
        loop_times = (cache_len + block_size - 1) // block_size  # div_up
        for tile_id in range(loop_times):
            batch_ids.append(bid)
            tile_ids_per_batch.append(tile_id)
        gridx += loop_times
        total_tokens += cache_len + q_len
        cu_seqlens_k[bid + 1] = total_tokens

    return (
        cu_seqlens_k,
        np.array(batch_ids, dtype=np.int32),
        np.array(tile_ids_per_batch, dtype=np.int32),
        np.array([gridx], dtype=np.int32),
        np.array([total_tokens], dtype=np.int32),
    )


class TestPreCacheLenConcat(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")

    def test_smoke_shapes(self):
        bsz = 3
        max_dec_len, block_size = 16, 4

        seq_lens_decoder = np.array([8, 4, 2], dtype=np.int32)
        seq_lens_this_time = np.array([2, 3, 1], dtype=np.int32)

        seq_lens_decoder_t = paddle.to_tensor(seq_lens_decoder, dtype="int32")
        seq_lens_this_time_t = paddle.to_tensor(seq_lens_this_time, dtype="int32")

        outputs = pre_cache_len_concat(seq_lens_decoder_t, seq_lens_this_time_t, max_dec_len, block_size)
        cu_seqlens_k, batch_ids, tile_ids, num_blocks, kv_token_num = [out.numpy() for out in outputs]

        # Shape checks
        self.assertEqual(cu_seqlens_k.shape[0], bsz + 1)
        self.assertEqual(batch_ids.shape, tile_ids.shape)
        self.assertEqual(num_blocks.shape, (1,))
        self.assertEqual(kv_token_num.shape, (1,))

        # Basic value sanity checks
        self.assertTrue(np.all(np.diff(cu_seqlens_k) >= 0))  # monotonic
        self.assertGreaterEqual(num_blocks[0], 0)
        self.assertGreaterEqual(kv_token_num[0], 0)

    def test_strict_values_with_ref(self):
        max_dec_len, block_size = 16, 4

        seq_lens_decoder = np.array([8, 4, 2], dtype=np.int32)
        seq_lens_this_time = np.array([2, 3, 1], dtype=np.int32)

        seq_lens_decoder_t = paddle.to_tensor(seq_lens_decoder, dtype="int32")
        seq_lens_this_time_t = paddle.to_tensor(seq_lens_this_time, dtype="int32")

        outputs = pre_cache_len_concat(seq_lens_decoder_t, seq_lens_this_time_t, max_dec_len, block_size)
        cu_seqlens_k, batch_ids, tile_ids, num_blocks, kv_token_num = [out.numpy() for out in outputs]

        # Reference implementation
        ref_outputs = ref_pre_cache_len_concat(seq_lens_decoder, seq_lens_this_time, block_size)
        ref_cu, ref_batch_ids, ref_tile_ids, ref_num_blocks, ref_kv_token_num = ref_outputs

        # Compare all outputs against reference
        np.testing.assert_array_equal(cu_seqlens_k, ref_cu)
        np.testing.assert_array_equal(batch_ids[: len(ref_batch_ids)], ref_batch_ids)
        np.testing.assert_array_equal(tile_ids[: len(ref_tile_ids)], ref_tile_ids)
        self.assertEqual(num_blocks[0], ref_num_blocks[0])
        self.assertEqual(kv_token_num[0], ref_kv_token_num[0])


if __name__ == "__main__":
    unittest.main()
```



## 64 test_draft_model_flags

### Code

```c++
// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "helper.h"

// ====================== CUDA Kernel ======================
// 将 draft_tokens 中的新生成 token 更新到 pre_ids_all 中，处理推理过程中序列的累积。
// draft_tokens: 每个 batch 的草稿 token，shape=[bs, max_draft_token]
// pre_ids_all: 保存每个 batch 所有生成 token 的数组，shape=[bs, pre_id_length]
// stop_flags: 每个 batch 是否已经停止生成，shape=[bs]
// seq_lens_this_time: 本次生成的 token 数量，shape=[bs]
// step_idx: 当前解码步数，shape=[bs]
// bs: batch size
// pre_id_length: pre_ids_all 每行长度
// max_draft_token: draft_tokens 每行长度
__global__ void update_pre_ids_kernel(const int64_t* draft_tokens,
                                      int64_t* pre_ids_all,
                                      const bool* stop_flags,
                                      int* seq_lens_this_time,
                                      const int64_t* step_idx,
                                      int bs,
                                      int pre_id_length,
                                      int max_draft_token) {
    // 每个线程处理一个 batch
    int tid = threadIdx.x;

    // 线程 tid 有效且该 batch 本次生成 token 不为 0 且未停止
    if (tid < bs && seq_lens_this_time[tid] != 0 && !stop_flags[tid]) {
        // 指向当前 batch 的 pre_ids_all
        int64_t* pre_ids_all_now = pre_ids_all + tid * pre_id_length;
        // 指向当前 batch 的 draft_tokens
        const int64_t* draft_token_now = draft_tokens + tid * max_draft_token;
        const int seq_len_this_time = seq_lens_this_time[tid];

        // 如果是 Decoder 步（step_idx > 1）
        if (step_idx[tid] - 1 > 0 /*Decoder Step*/) {
            // 从后向前，将本次生成的 token 放入 pre_ids_all 的对应位置
            for (int i = 0; i < seq_len_this_time; ++i) {
                // 将 draft_token 的最后几个 token 按顺序填入 pre_ids_all
                pre_ids_all_now[step_idx[tid] - i] =
                    draft_token_now[seq_len_this_time - 1 - i];
            }
        } 
        // 如果是 Encoder 第一步（step_idx == 1）
        else if (step_idx[tid] == 1 /*Encoder Step*/) {
            pre_ids_all_now[1] = draft_token_now[0]; // 只更新第一个 token
        }

        // 本次生成 token 数量置为 1（方便下一步统计或处理）
        seq_lens_this_time[tid] = 1;
    }
}

// ====================== Host 函数 ======================
// 负责调用 CUDA kernel，将 draft_tokens 更新到 pre_ids_all
// 传入的都是 Paddle Tensor
void SpeculateDraftModelUpdate(const paddle::Tensor& draft_tokens,
                               const paddle::Tensor& pre_ids_all,
                               const paddle::Tensor& stop_flags,
                               const paddle::Tensor& seq_lens_this_time,
                               const paddle::Tensor& seq_lens_encoder,
                               const paddle::Tensor& seq_lens_decoder,
                               const paddle::Tensor& step_idx) {
    int64_t real_bs = seq_lens_this_time.shape()[0];  // batch size
    int64_t pre_id_length = pre_ids_all.shape()[1];   // 每行 pre_ids_all 长度
    auto cu_stream = seq_lens_this_time.stream();     // CUDA stream
    int64_t max_draft_token = draft_tokens.shape()[1]; // 每行 draft token 数量

    // CUDA block 大小，保证覆盖所有 batch
    int block_size = (real_bs + 32 - 1) / 32 * 32;

    // 调用 kernel
    update_pre_ids_kernel<<<1, block_size, 0, cu_stream>>>(
        draft_tokens.data<int64_t>(),                        // 输入草稿 token
        const_cast<int64_t*>(pre_ids_all.data<int64_t>()),  // 输出 pre_ids_all
        stop_flags.data<bool>(),                             // 停止标志
        const_cast<int*>(seq_lens_this_time.data<int>()),   // 本次生成长度
        step_idx.data<int64_t>(),                            // 步数
        real_bs,                                            // batch size
        pre_id_length,                                      // pre_ids_all 长度
        max_draft_token);                                   // draft token 长度
}

// ====================== Paddle Static Op ======================
// 将上面的函数注册为 PaddlePaddle 静态算子
PD_BUILD_STATIC_OP(draft_model_set_value_by_flags)
    .Inputs({"draft_tokens",
             "pre_ids_all",
             "stop_flags",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "step_idx"})
    .Outputs({"pre_ids_all_out"})
    .SetInplaceMap({{"pre_ids_all", "pre_ids_all_out"}})  // inplace 更新 pre_ids_all
    .SetKernelFn(PD_KERNEL(SpeculateDraftModelUpdate));    // 调用的 kernel 函数

```

**CUDA kernel**

- 每个线程处理一个 batch
- 根据 step_idx 判断 Encoder/Decoder
- 将本次生成 token 填入 pre_ids_all，保证序列连续

**Host 函数**

- 提取 Paddle Tensor 的 shape、CUDA stream
- 调用 kernel 并传入 raw 指针

**Static Op 注册**

- PaddlePaddle 端注册静态算子
- `SetInplaceMap` 表示 pre_ids_all 会被原地更新

### Test

```python
import unittest
import numpy as np
import paddle
from fastdeploy.model_executor.ops.gpu import draft_model_set_value_by_flags

class TestDraftModelSetValueByFlags(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")
        np.random.seed(42)

    def test_basic_update(self):
        bs = 2
        pre_id_length = 5
        draft_tokens = paddle.to_tensor([[10,11,12],[20,21,22]], dtype='int64')
        pre_ids_all = paddle.zeros([bs, pre_id_length], dtype='int64')
        stop_flags = paddle.to_tensor([False, False], dtype='bool')
        seq_lens_this_time = paddle.to_tensor([3,1], dtype='int32')
        seq_lens_encoder = paddle.to_tensor([0,0], dtype='int32')
        seq_lens_decoder = paddle.to_tensor([0,0], dtype='int32')
        step_idx = paddle.to_tensor([3,1], dtype='int64')  # batch0 decoder, batch1 encoder

        draft_model_set_value_by_flags(
            draft_tokens,
            pre_ids_all,
            stop_flags,
            seq_lens_this_time,
            seq_lens_encoder,
            seq_lens_decoder,
            step_idx
        )

        expected = np.array([
            [0,10,11,12,0],  # batch0 decoder step
            [0,20,0,0,0]     # batch1 encoder step
        ], dtype=np.int64)

        np.testing.assert_array_equal(pre_ids_all.numpy(), expected)
        np.testing.assert_array_equal(seq_lens_this_time.numpy(), [1,1])

    def test_stop_flags(self):
        bs = 2
        pre_id_length = 4
        draft_tokens = paddle.to_tensor([[5,6],[7,8]], dtype='int64')
        pre_ids_all = paddle.zeros([bs, pre_id_length], dtype='int64')
        stop_flags = paddle.to_tensor([True, False], dtype='bool')
        seq_lens_this_time = paddle.to_tensor([2,2], dtype='int32')
        seq_lens_encoder = paddle.to_tensor([0,0], dtype='int32')
        seq_lens_decoder = paddle.to_tensor([0,0], dtype='int32')
        step_idx = paddle.to_tensor([1,2], dtype='int64')

        draft_model_set_value_by_flags(
            draft_tokens,
            pre_ids_all,
            stop_flags,
            seq_lens_this_time,
            seq_lens_encoder,
            seq_lens_decoder,
            step_idx
        )

        expected = np.array([
            [0,0,0,0],      # batch0 stop_flags=True, 不更新
            [0,7,8,0]       # batch1 decoder step
        ], dtype=np.int64)

        np.testing.assert_array_equal(pre_ids_all.numpy(), expected)
        np.testing.assert_array_equal(seq_lens_this_time.numpy(), [2,1])

if __name__ == "__main__":
    unittest.main()

```

```
..
----------------------------------------------------------------------
Ran 2 tests in 0.193s

OK
```

## 68 ngram_match

### First Try

```python
import unittest

import paddle

from fastdeploy.model_executor.ops.gpu import ngram_match


class TestNgramMatchOp(unittest.TestCase):

    def setUp(self):
        paddle.set_device("cpu")

    def test_basic_match(self):
        """
        Case 1: input_ids overlaps with pre_ids, and can extract draft tokens.
        """
        batch_size = 1
        seq_len = 6

        # Input IDs
        input_ids = paddle.to_tensor([[10, 20, 30, 40, 50, 60]], dtype="int64")
        # Length of input IDs
        input_ids_len = paddle.to_tensor([6], dtype="int64")
        # Previous IDs
        pre_ids = paddle.to_tensor([[10, 20, 30, 40, 0, 0]], dtype="int64")
        # Current step index
        step_idx = paddle.to_tensor([3], dtype="int64")
        # Number of draft tokens
        draft_token_num = paddle.to_tensor([3], dtype="int32")
        # Placeholder for draft tokens
        draft_tokens = paddle.zeros([batch_size, seq_len], dtype="int64")

        # Sequence lengths for this time step
        seq_lens_this_time = paddle.zeros([batch_size], dtype="int32")
        # Sequence lengths for encoder
        seq_lens_encoder = paddle.zeros([batch_size], dtype="int32")
        # Sequence lengths for decoder
        seq_lens_decoder = paddle.ones([batch_size], dtype="int32")
        # Maximum decoding length
        max_dec_len = paddle.to_tensor([10], dtype="int64")

        # Call the OP (in-place modification)
        ngram_match(
            input_ids,
            input_ids_len,
            pre_ids,
            step_idx,
            draft_token_num,
            draft_tokens,
            seq_lens_this_time,
            seq_lens_encoder,
            seq_lens_decoder,
            max_dec_len,
            3,  # Maximum n-gram size
            4,  # Maximum draft tokens
        )

        print("draft_tokens_out:", draft_tokens.numpy())
        print("seq_lens_this_time_out:", seq_lens_this_time.numpy())

        # Check if draft tokens are correctly extracted
        self.assertIn(50, draft_tokens.numpy()[0])
        self.assertIn(60, draft_tokens.numpy()[0])
        self.assertEqual(seq_lens_this_time.numpy()[0], 3)

    def test_no_match(self):
        """
        Case 2: pre_ids does not match input_ids, should only keep the current token.
        """
        batch_size = 1
        input_ids = paddle.to_tensor([[100, 200, 300, 400]], dtype="int64")
        input_ids_len = paddle.to_tensor([4], dtype="int64")
        pre_ids = paddle.to_tensor([[1, 2, 3, 4]], dtype="int64")
        step_idx = paddle.to_tensor([3], dtype="int64")
        draft_token_num = paddle.to_tensor([2], dtype="int32")
        draft_tokens = paddle.zeros([batch_size, 4], dtype="int64")

        seq_lens_this_time = paddle.zeros([batch_size], dtype="int32")
        seq_lens_encoder = paddle.zeros([batch_size], dtype="int32")
        seq_lens_decoder = paddle.ones([batch_size], dtype="int32")
        max_dec_len = paddle.to_tensor([6], dtype="int64")

        ngram_match(
            input_ids,
            input_ids_len,
            pre_ids,
            step_idx,
            draft_token_num,
            draft_tokens,
            seq_lens_this_time,
            seq_lens_encoder,
            seq_lens_decoder,
            max_dec_len,
            3,
            3,
        )

        print("draft_tokens_out:", draft_tokens.numpy())
        print("seq_lens_this_time_out:", seq_lens_this_time.numpy())

        # No match → should only keep 1 token
        self.assertEqual(seq_lens_this_time.numpy()[0], 1)


if __name__ == "__main__":
    unittest.main()
```



### Fix 9.3

```python
import unittest

import paddle

from fastdeploy.model_executor.ops.gpu import ngram_match


class TestNgramMatchOp(unittest.TestCase):

    def setUp(self):
        paddle.set_device("cpu")

    def test_basic_match(self):
        """
        Case 1: input_ids overlaps with pre_ids, and can extract draft tokens.
        """
        batch_size = 1
        seq_len = 6

        # Input IDs
        input_ids = paddle.to_tensor([[10, 20, 30, 40, 50, 60]], dtype="int64")
        # Length of input IDs
        input_ids_len = paddle.to_tensor([6], dtype="int64")
        # Previous IDs
        pre_ids = paddle.to_tensor([[10, 20, 30, 40, 0, 0]], dtype="int64")
        # Current step index
        step_idx = paddle.to_tensor([3], dtype="int64")
        # Number of draft tokens
        draft_token_num = paddle.to_tensor([3], dtype="int32")
        # Placeholder for draft tokens
        draft_tokens = paddle.zeros([batch_size, seq_len], dtype="int64")
   
   		# Sequence lengths for this time step
        seq_lens_this_time = paddle.zeros([batch_size], dtype="int32")
        # Sequence lengths for encoder
        seq_lens_encoder = paddle.zeros([batch_size], dtype="int32")
        # Sequence lengths for decoder
        seq_lens_decoder = paddle.ones([batch_size], dtype="int32")
        # Maximum decoding length
        max_dec_len = paddle.to_tensor([10], dtype="int64")
    
        ngram_match(
            input_ids,
            input_ids_len,
            pre_ids,
            step_idx,
            draft_token_num,
            draft_tokens,
            seq_lens_this_time,
            seq_lens_encoder,
            seq_lens_decoder,
            max_dec_len,
            3,
            4,
        )
    	
        # print("step_idx:", step_idx.numpy())
        # print("draft_tokens_out:", draft_tokens.numpy())
        # print("seq_lens_this_time_out:", seq_lens_this_time.numpy())
    
        # Extract non-zero tokens and assert the results.
        nonzero_tokens = draft_tokens.numpy()[0][draft_tokens.numpy()[0] != 0]
        expected_tokens = [50, 60]
        self.assertTrue((nonzero_tokens == expected_tokens).all())
    
        # Check length
        self.assertEqual(seq_lens_this_time.numpy()[0], 3)



    def test_no_match(self):
        """
        Case 2: pre_ids does not match input_ids, should only keep the current token.
        """
        batch_size = 1
        input_ids = paddle.to_tensor([[100, 200, 300, 400]], dtype="int64")
        input_ids_len = paddle.to_tensor([4], dtype="int64")
        pre_ids = paddle.to_tensor([[1, 2, 3, 4]], dtype="int64")
        step_idx = paddle.to_tensor([3], dtype="int64")
        draft_token_num = paddle.to_tensor([2], dtype="int32")
        draft_tokens = paddle.zeros([batch_size, 4], dtype="int64")

        seq_lens_this_time = paddle.zeros([batch_size], dtype="int32")
        seq_lens_encoder = paddle.zeros([batch_size], dtype="int32")
        seq_lens_decoder = paddle.ones([batch_size], dtype="int32")
        max_dec_len = paddle.to_tensor([6], dtype="int64")

        ngram_match(
            input_ids,
            input_ids_len,
            pre_ids,
            step_idx,
            draft_token_num,
            draft_tokens,
            seq_lens_this_time,
            seq_lens_encoder,
            seq_lens_decoder,
            max_dec_len,
            3,
            3,
        )

        print("draft_tokens_out:", draft_tokens.numpy())
        print("seq_lens_this_time_out:", seq_lens_this_time.numpy())

        # No match → should only keep 1 token
        self.assertEqual(seq_lens_this_time.numpy()[0], 1)


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



## 76 XGrammarChecker

```python
import json
import unittest

from fastdeploy.engine.request import Request
from fastdeploy.model_executor.guided_decoding.xgrammar_backend import XGrammarChecker


def make_request(**kwargs) -> Request:
    """
    Construct a Request object with default fields and override with any provided keyword arguments.
    This helper function simplifies creating Request instances for testing by
    pre-filling common fields and allowing selective overrides.
    """
    base = dict(
        request_id="req-1",
        prompt="",
        prompt_token_ids=[],
        prompt_token_ids_len=0,
        messages=[],
        history=[],
        tools=[],
        system="",
        sampling_params={},
        eos_token_ids=[],
        arrival_time=0.0,
        guided_json=None,
        guided_grammar=None,
        guided_json_object=None,
        guided_choice=None,
        structural_tag=None,
    )
    base.update(kwargs)
    return Request(**base)


class TestXGrammarChecker(unittest.TestCase):
    def setUp(self):
        self.checker = XGrammarChecker()

    def test_guided_json_valid(self):
        """
        Test that a valid guided_json passes the schema check.
        """

        request = make_request(guided_json={"type": "string"})
        request, err = self.checker.schema_format(request)
        self.assertIsNone(err)
        self.assertIsInstance(request.guided_json, str)

    def test_guided_json_invalid(self):
        """
        Test that an invalid guided_json returns an error.
        """

        request = make_request(guided_json={"type": "unknown_type"})
        request, err = self.checker.schema_format(request)
        self.assertIsNotNone(err)

    def test_guided_json_object(self):
        """
        Test that guided_json_object generates a JSON object type.
        """

        request = make_request(guided_json_object=True)
        request, err = self.checker.schema_format(request)
        self.assertIsNone(err)
        self.assertEqual(request.guided_json, '{"type": "object"}')

    def test_guided_grammar_valid(self):
        """
        Test that a valid guided_grammar passes the schema check.
        """

        request = make_request(guided_grammar='root ::= "yes" | "no"')
        request, err = self.checker.schema_format(request)
        self.assertIsNone(err)
        self.assertIn("root", request.guided_grammar)

    def test_guided_grammar_invalid(self):
        """
        Test that an invalid guided_grammar returns an error.
        """

        request = make_request(guided_grammar="root := ")
        request, err = self.checker.schema_format(request)
        self.assertIsNotNone(err)

    def test_guided_choice_valid(self):
        """
        Test that a valid guided_choice is correctly converted to EBNF.
        """

        request = make_request(guided_choice=["yes", "no"])
        request, err = self.checker.schema_format(request)
        self.assertIsNone(err)
        self.assertIn("yes", request.guided_grammar)
        self.assertIn("no", request.guided_grammar)

    def test_guided_choice_invalid(self):
        """
        Test that an invalid guided_choice (containing None) raises TypeError.
        """

        request = make_request(guided_choice=[None])
        with self.assertRaises(TypeError):
            self.checker.schema_format(request)

    def test_structural_tag_valid(self):
        """
        Test that a valid structural_tag passes the schema check.
        """

        structural_tag = {
            "structures": [{"begin": "<a>", "schema": {"type": "string"}, "end": "</a>"}],
            "triggers": ["<a>"],
        }
        request = make_request(structural_tag=json.dumps(structural_tag))
        request, err = self.checker.schema_format(request)
        self.assertIsNone(err)

    def test_structural_tag_invalid(self):
        """
        Test that a structural_tag missing 'triggers' raises KeyError.
        """

        structural_tag = {"structures": [{"begin": "<a>", "schema": {"type": "string"}, "end": "</a>"}]}
        request = make_request(structural_tag=json.dumps(structural_tag))
        with self.assertRaises(KeyError):
            self.checker.schema_format(request)

    def test_regex_passthrough(self):
        """
        Test that regex is not modified by schema_format and passes through as-is.
        """

        request = make_request()
        request.regex = "^[a-z]+$"
        request, err = self.checker.schema_format(request)
        self.assertIsNone(err)
        self.assertEqual(request.regex, "^[a-z]+$")


if __name__ == "__main__":
    unittest.main(verbosity=2)
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



