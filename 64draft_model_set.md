# Code

draft_model_set_value_by_flags 

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

# Test

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