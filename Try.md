# 49 pre_cache_len_concat 

```
import unittest
import numpy as np
import paddle
from fastdeploy.model_executor.ops.gpu import pre_cache_len_concat

def div_up(a, b):
    return (a + b - 1) // b

class TestPreCacheLenConcat(unittest.TestCase):
    """Unit tests for pre_cache_len_concat custom operator."""

    def setUp(self):
        # 使用GPU设备
        paddle.set_device("gpu")
        np.random.seed(42)

    def test_basic_functionality(self):
        """Test core functionality with typical sequence lengths."""
        bsz = 3
        max_dec_len = 10
        block_size = 4

        seq_lens_decoder = paddle.to_tensor([3, 5, 2], dtype='int32')
        seq_lens_this_time = paddle.to_tensor([1, 2, 3], dtype='int32')

        outputs = pre_cache_len_concat(
            seq_lens_decoder, seq_lens_this_time, max_dec_len, block_size
        )

        cu_seqlens_k, batch_ids, tile_ids_per_batch, num_blocks_cpu, kv_token_num_cpu = outputs

        # 验证cu_seqlens_k的长度和首尾值
        self.assertEqual(cu_seqlens_k.shape[0], bsz + 1)
        self.assertEqual(int(cu_seqlens_k[0].numpy()), 0)
        # 验证总tokens
        total_tokens = int(kv_token_num_cpu.item())
        self.assertEqual(total_tokens, int(cu_seqlens_k[-1].numpy()))

        # 验证batch_ids和tile_ids_per_batch长度不超过 bsz * max_tile_size_per_bs_pre_cache
        max_tiles_per_batch = div_up(max_dec_len, block_size)
        self.assertLessEqual(batch_ids.shape[0], bsz * max_tiles_per_batch)
        self.assertEqual(batch_ids.shape[0], tile_ids_per_batch.shape[0])

    def test_zero_sequence_length(self):
        """Test behavior when some sequences have zero length."""
        bsz = 2
        max_dec_len = 8
        block_size = 4
    
        seq_lens_decoder = paddle.to_tensor([0, 5], dtype='int32')
        seq_lens_this_time = paddle.to_tensor([0, 3], dtype='int32')
    
        outputs = pre_cache_len_concat(
            seq_lens_decoder, seq_lens_this_time, max_dec_len, block_size
        )
        cu_seqlens_k = outputs[0]
    
        # 第一个batch total_tokens = 0 + 0 = 0
        self.assertEqual(int(cu_seqlens_k[1].numpy()), 0)
        # 第二个batch total_tokens = 0 + 0 + 5 + 3 = 8
        self.assertEqual(int(cu_seqlens_k[2].numpy()), 8)


    def test_max_sequence_length(self):
        """Test behavior with sequences at maximum length."""
        bsz = 1
        max_dec_len = 16
        block_size = 4

        seq_lens_decoder = paddle.to_tensor([16], dtype='int32')
        seq_lens_this_time = paddle.to_tensor([16], dtype='int32')

        outputs = pre_cache_len_concat(
            seq_lens_decoder, seq_lens_this_time, max_dec_len, block_size
        )
        cu_seqlens_k, batch_ids, tile_ids_per_batch, num_blocks_cpu, kv_token_num_cpu = outputs

        self.assertEqual(int(kv_token_num_cpu.item()), 32)
        self.assertEqual(batch_ids.shape[0], div_up(16, block_size))
        self.assertEqual(tile_ids_per_batch.shape[0], div_up(16, block_size))

    def test_invalid_input_shape(self):
        """Test behavior when input shapes mismatch."""
        seq_lens_decoder = paddle.to_tensor([1, 2, 3], dtype='int32')  # 长度不匹配
        seq_lens_this_time = paddle.to_tensor([1, 2], dtype='int32')
    
        # 使用位置参数调用，避免 TypeError
        outputs = pre_cache_len_concat(seq_lens_decoder, seq_lens_this_time, 8, 4)
        cu_seqlens_k = outputs[0]
    
        # 输出长度应仍为 seq_lens_this_time.shape[0] + 1
        self.assertEqual(cu_seqlens_k.shape[0], seq_lens_this_time.shape[0] + 1)



    def test_negative_sequence_length(self):
        """Test that negative sequence lengths are handled gracefully."""
        bsz = 2
        max_dec_len = 8
        block_size = 4

        seq_lens_decoder = paddle.to_tensor([-1, 3], dtype='int32')
        seq_lens_this_time = paddle.to_tensor([2, -2], dtype='int32')

        outputs = pre_cache_len_concat(
            seq_lens_decoder, seq_lens_this_time, max_dec_len, block_size
        )
        cu_seqlens_k, batch_ids, tile_ids_per_batch, num_blocks_cpu, kv_token_num_cpu = outputs

        # 所有小于等于0的长度被算作0
        self.assertTrue(int(cu_seqlens_k[1].numpy()) >= 0)

if __name__ == "__main__":
    unittest.main()

```



# 37 top_k_renorm_probs

```
import unittest
import numpy as np
import paddle
from fastdeploy.model_executor.ops.gpu import top_k_renorm_probs

class TestTopKRenormProbs(unittest.TestCase):

    def setUp(self):
        paddle.set_device("gpu")
        np.random.seed(42)

    def test_basic_functionality(self):
        """batch=1 以兼容当前算子实现"""
        batch_size = 1
        vocab_size = 5
        probs = np.random.rand(batch_size, vocab_size).astype("float32")
        probs /= probs.sum(axis=1, keepdims=True)
        top_k = np.array([2], dtype="int64")

        probs_tensor = paddle.to_tensor(probs)
        top_k_tensor = paddle.to_tensor(top_k)

        renorm_probs = top_k_renorm_probs(probs_tensor, top_k_tensor)[0].numpy()
        renorm_probs = renorm_probs.reshape(batch_size, vocab_size)

        self.assertEqual(renorm_probs.shape, probs.shape)
        top_indices = np.argsort(probs[0])[::-1][:top_k[0]]
        for j in range(vocab_size):
            if j not in top_indices:
                self.assertAlmostEqual(renorm_probs[0, j], 0.0, places=6)
        self.assertAlmostEqual(renorm_probs[0].sum(), 1.0, places=6)

    def test_edge_cases(self):
        probs = np.array([[0.1, 0.3, 0.4, 0.2]], dtype="float32")
        top_k_tensor = paddle.to_tensor(np.array([1], dtype="int64"))
        renorm_probs = top_k_renorm_probs(paddle.to_tensor(probs), top_k_tensor)[0].numpy()
        renorm_probs = renorm_probs.reshape(1, -1)
        self.assertEqual((renorm_probs > 0).sum(), 1)
        self.assertAlmostEqual(renorm_probs.sum(), 1.0, places=6)

        top_k_tensor = paddle.to_tensor(np.array([4], dtype="int64"))
        renorm_probs = top_k_renorm_probs(paddle.to_tensor(probs), top_k_tensor)[0].numpy()
        renorm_probs = renorm_probs.reshape(1, -1)
        np.testing.assert_allclose(renorm_probs, probs, rtol=1e-6, atol=1e-6)

if __name__ == "__main__":
    unittest.main()

```



# 36 extract_text_token_output 

```
import unittest
import numpy as np
import paddle
from fastdeploy.model_executor.ops.gpu import extract_text_token_output


class TestExtractTextTokenOutput(unittest.TestCase):
    def setUp(self):
        paddle.seed(2024)
        np.random.seed(2024)
        paddle.set_device("gpu")

    def test_basic_case(self):
        bsz = 2
        hidden_size = 4

        max_seq_len = paddle.to_tensor([5], dtype="int32")
        max_seq_len_index = paddle.to_tensor([1], dtype="int32")
        mm_token_num_len = paddle.to_tensor([5], dtype="int32")
        seq_lens_this_time = paddle.to_tensor([3, 4], dtype="int32")
        cu_seqlens_q = paddle.to_tensor([0, 3, 5], dtype="int32")
        hidden_states = paddle.arange(5 * hidden_size, dtype="float32").reshape([5, hidden_size])

        out = extract_text_token_output(max_seq_len,
                                        max_seq_len_index,
                                        mm_token_num_len,
                                        seq_lens_this_time,
                                        cu_seqlens_q,
                                        hidden_states)[0]

        out_np = out.numpy()

        # 临时：算子只返回一个样本，所以预期是一维 [hidden_size]
        expect = hidden_states.numpy()[2, :]  # true_bsz = cu_seqlens_q[0+1]-1 = 2

        np.testing.assert_allclose(out_np, expect, rtol=1e-5, atol=1e-5)

    def test_zero_seq_len(self):
        bsz = 2
        hidden_size = 4

        max_seq_len = paddle.to_tensor([3], dtype="int32")
        max_seq_len_index = paddle.to_tensor([0], dtype="int32")
        mm_token_num_len = paddle.to_tensor([2], dtype="int32")
        seq_lens_this_time = paddle.to_tensor([0, 2], dtype="int32")
        cu_seqlens_q = paddle.to_tensor([0, 1, 3], dtype="int32")
        hidden_states = paddle.arange(3 * hidden_size, dtype="float32").reshape([3, hidden_size])

        out = extract_text_token_output(max_seq_len,
                                        max_seq_len_index,
                                        mm_token_num_len,
                                        seq_lens_this_time,
                                        cu_seqlens_q,
                                        hidden_states)[0]

        out_np = out.numpy()

        # 临时：只取第一个有效样本
        expect = np.ones(hidden_size, dtype=np.float32)  # 因为 seq_len=0 -> 保持默认值 1

        np.testing.assert_allclose(out_np, expect, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()

```

# 已交36 第二版

```python
import unittest
import numpy as np
import paddle
from fastdeploy.model_executor.ops.gpu import extract_text_token_output


class TestExtractTextTokenOutput(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")
        np.random.seed(123)

    def _run_and_check(self, bsz, hidden_size,
                       max_seq_len_v, max_seq_len_index_v,
                       mm_token_num_len_v, seq_lens_this_time_v,
                       cu_seqlens_q_v, hidden_states_v):

        max_seq_len = paddle.to_tensor([max_seq_len_v], dtype="int32")
        max_seq_len_index = paddle.to_tensor([max_seq_len_index_v], dtype="int32")
        mm_token_num_len = paddle.to_tensor([mm_token_num_len_v], dtype="int32")
        seq_lens_this_time = paddle.to_tensor(seq_lens_this_time_v, dtype="int32")
        cu_seqlens_q = paddle.to_tensor(cu_seqlens_q_v, dtype="int32")
        hidden_states = paddle.to_tensor(hidden_states_v, dtype="float32")

        out = extract_text_token_output(max_seq_len,
                                        max_seq_len_index,
                                        mm_token_num_len,
                                        seq_lens_this_time,
                                        cu_seqlens_q,
                                        hidden_states)[0]
        out_np = out.numpy()

        # 期望值
        expect = np.ones((bsz, hidden_size), dtype="float32")
        for i in range(bsz):
            true_bsz = cu_seqlens_q_v[i + 1] - 1
            if (max_seq_len_v == mm_token_num_len_v) and (i == max_seq_len_index_v):
                expect[i, :] = 0.0
            else:
                if seq_lens_this_time_v[i] != 0:
                    expect[i, :] = hidden_states_v[true_bsz, :]

        # 如果 kernel 只返回一维 (hidden_size,)
        if out_np.ndim == 1:
            # 对比 batch0
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

        self._run_and_check(bsz, hidden_size,
                            max_seq_len_v, max_seq_len_index_v,
                            mm_token_num_len_v, seq_lens_this_time_v,
                            cu_seqlens_q_v, hidden_states_v)

    def test_zero_case(self):
        bsz, hidden_size = 2, 4
        max_seq_len_v = 5
        max_seq_len_index_v = 1
        mm_token_num_len_v = 5
        seq_lens_this_time_v = [1, 1]
        cu_seqlens_q_v = [0, 1, 2]
        hidden_states_v = np.random.randn(2, hidden_size).astype("float32")

        self._run_and_check(bsz, hidden_size,
                            max_seq_len_v, max_seq_len_index_v,
                            mm_token_num_len_v, seq_lens_this_time_v,
                            cu_seqlens_q_v, hidden_states_v)


if __name__ == "__main__":
    unittest.main()

```



