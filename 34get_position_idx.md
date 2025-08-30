# Code

get_position_ids_and_mask_encoder_batch

```c++
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

# Test

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
        # 假设批次大小为 2
        seq_lens_encoder = paddle.to_tensor([3, 2], dtype='int32')
        seq_lens_decoder = paddle.to_tensor([1, 2], dtype='int32')
        seq_lens_this_time = paddle.to_tensor([1, 2], dtype='int32')

        # 输出长度 = sum(encoder + decoder_this_time)
        total_len = int(seq_lens_encoder.numpy().sum() + seq_lens_this_time.numpy().sum())
        position_ids = paddle.zeros([total_len], dtype='int32')
        mask_encoder_batch = paddle.zeros([total_len], dtype='int32')

        # 调用自定义算子
        get_position_ids_and_mask_encoder_batch(
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            position_ids,
            mask_encoder_batch
        )

        # 构造预期输出
        expected_position_ids = np.array([
            0, 1, 2, 
            1,    
            0, 1,  
            2, 3 
        ], dtype=np.int32)

        expected_mask = np.array([
            1, 1, 1, 
            0,      
            1, 1,   
            0, 0    
        ], dtype=np.int32)

        # 转 numpy，便于比较
        position_ids_np = position_ids.numpy()
        mask_encoder_batch_np = mask_encoder_batch.numpy()

        np.testing.assert_array_equal(position_ids_np, expected_position_ids)
        np.testing.assert_array_equal(mask_encoder_batch_np, expected_mask)

    def test_empty_decoder(self):
        # 测试 decoder 长度为 0 的情况
        seq_lens_encoder = paddle.to_tensor([2], dtype='int32')
        seq_lens_decoder = paddle.to_tensor([0], dtype='int32')
        seq_lens_this_time = paddle.to_tensor([0], dtype='int32')

        position_ids = paddle.zeros([2], dtype='int32')
        mask_encoder_batch = paddle.zeros([2], dtype='int32')

        get_position_ids_and_mask_encoder_batch(
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            position_ids,
            mask_encoder_batch
        )

        expected_position_ids = np.array([0, 1], dtype=np.int32)
        expected_mask = np.array([1, 1], dtype=np.int32)

        np.testing.assert_array_equal(position_ids.numpy(), expected_position_ids)
        np.testing.assert_array_equal(mask_encoder_batch.numpy(), expected_mask)


if __name__ == "__main__":
    unittest.main()

```

```
..
----------------------------------------------------------------------
Ran 2 tests in 0.194s

OK
```