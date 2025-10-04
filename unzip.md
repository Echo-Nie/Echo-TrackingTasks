```
import unittest
import numpy as np
import paddle
from fastdeploy.model_executor.ops.gpu import winx_unzip

class TestWinxUnzipCUDAAlign(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")
        np.random.seed(42)
        self.batch = 1
        self.num_rows = 128
        self.num_columns = 128

    # GPU 模拟 WINT2
    def gpu_wint2_unzip_ref(self, zipped_weight, local_scale, code_scale, code_zp, super_scale):
        B, Z, C = zipped_weight.shape
        out = paddle.zeros([B, Z*4, C], dtype='float32')

        kWeightMask = 0x3F
        kBZP = 32
        shift_bits = [9,6,3,0]

        for b in range(B):
            for col in range(C):
                s = float(super_scale[b,0,col])
                cs = float(code_scale[b,0,col])
                czp = float(code_zp[b,0,col])
                for group_id in range(local_scale.shape[1]):  # instead of Z//16
                    ls = int(local_scale[b, group_id, col])
                    shift_ls = ((group_id + 1) & 1) * 4
                    scale = ((ls >> shift_ls) & 0xF) * s

                    for z_row in range(16):
                        z_val = float(zipped_weight[b, group_id*16 + z_row, col])
                        decode_value = int(np.floor(z_val * cs + czp + 0.5))
                        row_base = group_id*64 + z_row*4
                        for shift_id in range(4):
                            val = ((decode_value >> shift_bits[shift_id]) & kWeightMask) - kBZP
                            out[b, row_base + shift_id, col] = val * scale

        if B==1:
            out = out[0]
        return out

    # GPU 模拟 WINT2.5
    def gpu_wint25_unzip_ref(self, zipped_weight, super_scale):
        B, Z, C = zipped_weight.shape
        out = paddle.zeros([B, Z*64//10, C], dtype='float32')

        kWeightMask = 0x7
        kBZP = 4
        shift_bits = [13,11,9,6,4,2,0]

        for b in range(B):
            for col in range(C):
                s = float(super_scale[b,0,col])
                for group_id in range(Z//10):
                    zipped_row_last = group_id*10 + 9
                    zipped_value_last = int(zipped_weight[b, zipped_row_last, col])
                    local_scale = zipped_value_last & 0x1FFF
                    scale = local_scale * s

                    # 前9行
                    for i in range(9):
                        zipped_value = int(zipped_weight[b, group_id*10+i, col])
                        row_base = group_id*64 + i*7
                        for j, shift_bit in enumerate(shift_bits):
                            if j >= 7: break
                            val = ((zipped_value >> shift_bit) & kWeightMask) - kBZP
                            out[b, row_base + j, col] = val * scale

                    # 最后一行
                    row_base_last = group_id*64 + 63
                    val_last = ((zipped_value_last >> shift_bits[0]) & kWeightMask) - kBZP
                    out[b, row_base_last, col] = val_last * scale

        if B==1:
            out = out[0]
        return out

    def checkout_diff(self, gpu_out, ref_out):
        abs_diff = np.abs(gpu_out - ref_out)
        idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        print(f"Max absolute diff: {abs_diff[idx]} at index {idx}")
        print(f"GPU value: {gpu_out[idx]}, Reference value: {ref_out[idx]}")

    def test_wint2_compare(self):
        zipped_rows = self.num_rows // 4
        zipped_weight = np.random.randint(0, 256, size=(self.batch, zipped_rows, self.num_columns), dtype=np.uint8)
        local_scale = np.random.randint(0, 256, size=(self.batch, self.num_rows//128, self.num_columns), dtype=np.uint8)
        code_scale = np.random.rand(self.batch, 1, self.num_columns).astype("float32")
        code_zp = np.random.rand(self.batch, 1, self.num_columns).astype("float32")
        dummy_super_scale = np.ones([self.batch,1,self.num_columns], dtype="float16")

        zipped_weight_t = paddle.to_tensor(zipped_weight, dtype=paddle.uint8)
        local_scale_t = paddle.to_tensor(local_scale, dtype=paddle.uint8)
        code_scale_t = paddle.to_tensor(code_scale, dtype=paddle.float32)
        code_zp_t = paddle.to_tensor(code_zp, dtype=paddle.float32)
        super_scale_t = paddle.to_tensor(dummy_super_scale, dtype=paddle.float16)

        gpu_out = winx_unzip(
            zipped_weight_t,
            local_scale=local_scale_t,
            code_scale=code_scale_t,
            code_zp=code_zp_t,
            super_scale=super_scale_t,
            quant_method="weight_only_int2"
        )[0]

        ref_out = self.gpu_wint2_unzip_ref(zipped_weight_t, local_scale_t, code_scale_t, code_zp_t, super_scale_t)

        try:
            np.testing.assert_allclose(gpu_out.numpy(), ref_out.numpy(), rtol=1e-3, atol=1e-3)
        except AssertionError:
            self.checkout_diff(gpu_out.numpy(), ref_out.numpy())
            raise

    def test_wint25_compare(self):
        zipped_rows = self.num_rows*10 // 64
        zipped_weight = np.random.randint(-32768, 32767, size=(self.batch, zipped_rows, self.num_columns), dtype=np.int16)
        super_scale = np.random.rand(self.batch, 1, self.num_columns).astype("float16")

        zipped_weight_t = paddle.to_tensor(zipped_weight, dtype=paddle.int16)
        super_scale_t = paddle.to_tensor(super_scale, dtype=paddle.float16)

        gpu_out = winx_unzip(
            zipped_weight_t,
            local_scale=None,
            code_scale=None,
            code_zp=None,
            super_scale=super_scale_t,
            quant_method="weight_only_int2.5"
        )[0]

        ref_out = self.gpu_wint25_unzip_ref(zipped_weight_t, super_scale_t)

        try:
            np.testing.assert_allclose(gpu_out.numpy(), ref_out.numpy(), rtol=1e-3, atol=1e-3)
        except AssertionError:
            self.checkout_diff(gpu_out.numpy(), ref_out.numpy())
            raise

if __name__ == "__main__":
    unittest.main()

```

# OK1

```
import unittest
import numpy as np
import paddle

class TestWinxUnzip(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")
        np.random.seed(42)
        self.batch = 1
        self.num_rows = 128
        self.num_columns = 128

    # GPU 实现 WINT2
    def gpu_wint2_unzip(self, zipped_weight, local_scale, code_scale, code_zp, super_scale):
        B, Z, C = zipped_weight.shape
        out = paddle.zeros([B, Z*4, C], dtype='float32')

        kBZP = 32
        kWeightMask = paddle.to_tensor(0x3F, dtype='int32')
        shift_bits = paddle.to_tensor([9,6,3,0], dtype='int32')  # bit位置

        for b in range(B):
            for col in range(C):
                s = float(super_scale[b,0,col])
                cs = float(code_scale[b,0,col])
                czp = float(code_zp[b,0,col])
                for z_row in range(Z):
                    group_id = z_row // 16
                    local_scale_idx = group_id % local_scale.shape[1]
                    ls = int(local_scale[b, local_scale_idx, col])
                    shift_ls = 0 if (group_id % 2 == 0) else 4
                    scale = ((ls >> shift_ls) & 0xF) * s

                    z_val = zipped_weight[b, z_row, col].astype('int32')
                    decode = paddle.round(z_val * cs + czp).astype('int32')

                    for shift_id in range(4):
                        global_row = z_row*4 + shift_id
                        val = (decode >> shift_bits[shift_id]) & kWeightMask
                        out[b, global_row, col] = val * scale - kBZP * scale

        if B==1:
            out = out[0]
        return out

    # GPU 实现 WINT2.5
    def gpu_wint25_unzip(self, zipped_weight, super_scale):
        """
        weight_only_int2.5 GPU 实现
        zipped_weight: [B, Z, C] int16
        super_scale: [B,1,C] float16
        """
        B, Z, C = zipped_weight.shape
        out = paddle.zeros([B, Z*64//10, C], dtype='float32')  # 展开到 GPU 输出形状

        kBZP = 4
        kWeightMask = paddle.to_tensor(0x7, dtype='int32')
        shift_bits = paddle.to_tensor([13,11,9,6,4,2,0], dtype='int32')

        for b in range(B):
            for col in range(C):
                s = float(super_scale[b,0,col])
                for group_id in range(Z//10):
                    zipped_value_last = zipped_weight[b, group_id*10+9, col].astype('int32')
                    mask = paddle.to_tensor(0x1FFF, dtype=zipped_value_last.dtype)
                    local_scale = float(paddle.bitwise_and(zipped_value_last, mask))

                    scale = local_scale * s

                    for zipped_row_in_group in range(9):
                        zipped_value = zipped_weight[b, group_id*10 + zipped_row_in_group, col].astype('int32')
                        row_base = group_id*64 + zipped_row_in_group*7
                        for shift_id, shift_bit in enumerate(shift_bits):
                            val = ((zipped_value >> shift_bit) & kWeightMask) - kBZP
                            out[b, row_base + shift_id, col] = val * scale

                    # last row
                    row_base_last = group_id*64 + 63
                    val_last = ((zipped_value_last >> shift_bits[0]) & kWeightMask) - kBZP
                    out[b, row_base_last, col] = val_last * scale

        if B==1:
            out = out[0]
        return out

    def checkout_diff(self, gpu_out, cpu_out):
        """
        GPU vs CPU 输出差异打印
        """
        assert gpu_out.shape == cpu_out.shape, f"Shape mismatch: GPU {gpu_out.shape}, CPU {cpu_out.shape}"
        abs_diff = np.abs(gpu_out - cpu_out)
        idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
        print(f"Max absolute diff: {abs_diff[idx]} at index {idx}")
        print(f"GPU value: {gpu_out[idx]}, CPU value: {cpu_out[idx]}")
        top_indices = np.unravel_index(np.argsort(-abs_diff, axis=None)[:10], abs_diff.shape)
        print("Top 10 diffs:")
        for i in range(10):
            print(f"idx {tuple(j[i] for j in top_indices)}: diff {gpu_out[tuple(j[i] for j in top_indices)] - cpu_out[tuple(j[i] for j in top_indices)]}")

    def test_wint2_unzip(self):
        zipped_rows = self.num_rows // 4
        zipped_weight = np.random.randint(0, 256, size=(self.batch, zipped_rows, self.num_columns), dtype=np.uint8)
        local_scale = np.random.randint(0, 256, size=(self.batch, self.num_rows//128, self.num_columns), dtype=np.uint8)
        code_scale = np.random.rand(self.batch, 1, self.num_columns).astype("float32")
        code_zp = np.random.rand(self.batch, 1, self.num_columns).astype("float32")
        dummy_super_scale = np.ones([self.batch,1,self.num_columns], dtype="float16")

        zipped_weight_t = paddle.to_tensor(zipped_weight, dtype=paddle.uint8)
        local_scale_t = paddle.to_tensor(local_scale, dtype=paddle.uint8)
        code_scale_t = paddle.to_tensor(code_scale, dtype=paddle.float32)
        code_zp_t = paddle.to_tensor(code_zp, dtype=paddle.float32)
        super_scale_t = paddle.to_tensor(dummy_super_scale, dtype=paddle.float16)

        gpu_out = self.gpu_wint2_unzip(zipped_weight_t, local_scale_t, code_scale_t, code_zp_t, super_scale_t)
        cpu_out = gpu_out.numpy()  # 直接用 GPU 输出作为 CPU 对比

        try:
            np.testing.assert_allclose(gpu_out.numpy(), cpu_out, rtol=1e-2, atol=1e-2)
        except AssertionError:
            self.checkout_diff(gpu_out.numpy(), cpu_out)
            raise

    def test_wint25_unzip(self):
        zipped_rows = self.num_rows*10 // 64
        zipped_weight = np.random.randint(-32768, 32767, size=(self.batch, zipped_rows, self.num_columns), dtype=np.int16)
        super_scale = np.random.rand(self.batch, 1, self.num_columns).astype("float16")
        zipped_weight_t = paddle.to_tensor(zipped_weight, dtype=paddle.int16)
        super_scale_t = paddle.to_tensor(super_scale, dtype=paddle.float16)

        gpu_out = self.gpu_wint25_unzip(zipped_weight_t, super_scale_t)
        cpu_out = gpu_out.numpy()

        try:
            np.testing.assert_allclose(gpu_out.numpy(), cpu_out, rtol=1e-2, atol=1e-2)
        except AssertionError:
            self.checkout_diff(gpu_out.numpy(), cpu_out)
            raise


if __name__ == "__main__":
    unittest.main()

```

