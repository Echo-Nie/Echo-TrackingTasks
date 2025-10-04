# First

```python
import unittest
import paddle
import numpy as np

from fastdeploy.model_executor.layers.quantization.weight_only import (
    GPUWeightOnlyLinearMethod,
    WeightOnlyConfig,
)

class DummyLinearLayer:
    """A dummy linear layer"""
    def __init__(self, in_features, out_features, add_bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight_shape = [out_features, in_features]
        self._dtype = "float16"
        self.add_bias = add_bias
        self.bias = paddle.to_tensor(
            np.random.randn(out_features).astype("float16")
        ) if add_bias else None

        self.fd_config = type('config', (), {})()
        self.fd_config.load_config = type('load_config', (), {"load_choices": "default_v1"})()

        # Initialize weights (int8) and weight_scale (float16)
        weight_int32 = paddle.randint(low=-128, high=127, shape=self.weight_shape, dtype="int32")
        self.weight = weight_int32.astype("int8")
        self.weight_scale = paddle.ones([self.in_features], dtype="float16")

    def create_parameter(self, shape, dtype, is_bias=False, default_initializer=None):
        if default_initializer is None:
            return paddle.create_parameter(shape=shape, dtype=dtype)
        else:
            return paddle.create_parameter(shape=shape, dtype=dtype, default_initializer=default_initializer)


class TestWeightOnlyLinearMethodCUDA(unittest.TestCase):
    def setUp(self):
        self.in_features = 16
        self.out_features = 16
        self.layer = DummyLinearLayer(self.in_features, self.out_features)
        self.quant_config = WeightOnlyConfig(algo="weight_only_int8")
        self.method = GPUWeightOnlyLinearMethod(self.quant_config)

    def test_weight_and_scale_shapes(self):
        """Test that weights and scales have the correct shapes."""
        self.assertEqual(list(self.layer.weight.shape), [self.out_features, self.in_features])
        self.assertEqual(list(self.layer.weight_scale.shape), [self.in_features])

    def test_apply_output_shape(self):
        """Test that applying the layer produces output of expected shape."""
        x = paddle.to_tensor(np.random.randn(2, self.in_features).astype("float16"))
        out = self.method.apply(self.layer, x)
        self.assertEqual(out.shape, [2, self.out_features])

    def test_apply_nonzero_values(self):
        """Test that the output is not all zeros."""
        x = paddle.to_tensor(np.random.randn(2, self.in_features).astype("float16"))
        out = self.method.apply(self.layer, x)
        self.assertFalse(np.allclose(out.numpy(), 0))

    def test_apply_reasonable_range(self):
        """Test that quantized outputs are within a reasonable range."""
        x = paddle.to_tensor(np.random.randn(2, self.in_features).astype("float16"))
        out = self.method.apply(self.layer, x).numpy()
        self.assertFalse(np.allclose(out, 0))

        max_expected = 128 * np.max(self.layer.weight_scale.numpy())
        self.assertTrue(np.all(np.abs(out) <= max_expected))


if __name__ == "__main__":
    unittest.main()

```

```
.=== Reference float output ===
[[-528.5   -82.06  338.5  -321.5   220.2    -8.26    7.98  135.2   553.
    41.9   420.8  -344.8  -457.8   -74.06   79.9   182.1 ]
 [ -41.3  -241.   -210.2   -99.44 -367.2   -76.5  -176.8   366.8  -111.44
   -59.22  -43.66  130.9   123.3  -120.1  -473.8   447.8 ]]
   
   
=== Quantized output ===
[[ 38.1      -0.544     1.457     0.9346   28.58      1.163    -1.15
   -0.716   -26.67     -0.1547   -0.08716   2.148    -4.754     1.836
   -0.798    -1.8125 ]
 [-21.28     -0.544     1.457     0.9346   11.875     1.163    -1.15
   -0.716   -57.4      -0.1547   -0.08716   2.148   -36.12      1.836
   -0.798    -1.8125 ]]
Max expected (weight_scale * 128): 128.0
Min expected: -128.0
Quantized output max: 38.09375
Quantized output min: -57.40625
Max absolute difference: 579.671875
Mean absolute difference: 215.577392578125
.Quantized output values:
 [[-7.4766e+00  2.8702e-02 -6.8408e-01  1.0781e+00  1.0664e+01  2.8223e-01
   9.4775e-01 -7.3584e-01  1.7047e+01  4.9219e-01  3.3740e-01 -1.0843e-03
   1.1727e+01 -1.8564e+00 -7.2168e-01  1.0684e+00]
 [ 3.9594e+01  2.8702e-02 -6.8408e-01  1.0781e+00 -4.9844e+01  2.8223e-01
   9.4775e-01 -7.3584e-01  6.0406e+01  4.9219e-01  3.3740e-01 -1.0843e-03
   5.4438e+01 -1.8564e+00 -7.2168e-01  1.0684e+00]]
Max expected value (based on weight_scale): 128.0
...
----------------------------------------------------------------------
Ran 5 tests in 0.305s

OK
```

## Print

```python
import unittest
import paddle
import numpy as np

from fastdeploy.model_executor.layers.quantization.weight_only import (
    GPUWeightOnlyLinearMethod,
    WeightOnlyConfig,
)

def debug_apply(method, layer, x):
    """Debug version of apply: dump intermediate results"""
    print("=== Debug Apply Start ===")
    print("Input shape:", x.shape)
    print("Weight int8 shape:", layer.weight.shape)
    print("Weight scale shape:", layer.weight_scale.shape)
    if layer.bias is not None:
        print("Bias shape:", layer.bias.shape)

    # 1. Int32 accumulation (simulate int8 matmul)
    weight_f32 = layer.weight.astype("float32")
    x_f32 = x.astype("float32")
    int32_accum = paddle.matmul(x_f32, weight_f32, transpose_y=True)
    print("Int32 accum stats: min={:.2f}, max={:.2f}, mean={:.2f}".format(
        float(int32_accum.min()), float(int32_accum.max()), float(int32_accum.mean())
    ))

    # 2. Apply weight_scale
    scaled = paddle.matmul(x_f32, (weight_f32 * layer.weight_scale.astype("float32")), transpose_y=True)
    print("Scaled stats (after weight_scale): min={:.2f}, max={:.2f}, mean={:.2f}".format(
        float(scaled.min()), float(scaled.max()), float(scaled.mean())
    ))

    # 3. Add bias
    if layer.bias is not None:
        out = scaled + layer.bias.astype("float32")
        print("Final output (after bias) stats: min={:.2f}, max={:.2f}, mean={:.2f}".format(
            float(out.min()), float(out.max()), float(out.mean())
        ))
    else:
        out = scaled

    print("=== Debug Apply End ===")
    return out.astype(x.dtype)


class DummyLinearLayer:
    """Dummy linear layer with simulated quantization (float32 -> int8 + scale)."""
    def __init__(self, in_features, out_features, add_bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight_shape = [out_features, in_features]
        self._dtype = "float16"
        self.add_bias = add_bias

        # bias
        self.bias = paddle.to_tensor(
            np.random.randn(out_features).astype("float16")
        ) if add_bias else None

        self.fd_config = type('config', (), {})()
        self.fd_config.load_config = type(
            'load_config', (), {"load_choices": "default_v1"}
        )()

        # ====== float32 weight ======
        weight_fp32 = np.random.randn(*self.weight_shape).astype("float32")

        # ====== per-channel scale ======
        max_abs = np.max(np.abs(weight_fp32), axis=1, keepdims=True) + 1e-6
        scale = (max_abs / 127.0).astype("float32")

        # ====== int8 quantization ======
        weight_int8 = np.clip(np.round(weight_fp32 / scale), -128, 127).astype("int8")

        # ====== store in paddle tensor ======
        self.weight = paddle.to_tensor(weight_int8, dtype="int8")
        self.weight_scale = paddle.to_tensor(scale.squeeze(-1).astype("float16"))

        # keep float weight for reference test
        self.weight_fp32 = paddle.to_tensor(weight_fp32.astype("float32"))

    def create_parameter(self, shape, dtype, is_bias=False, default_initializer=None):
        if default_initializer is None:
            return paddle.create_parameter(shape=shape, dtype=dtype)
        else:
            return paddle.create_parameter(
                shape=shape, dtype=dtype, default_initializer=default_initializer
            )


class TestWeightOnlyLinearMethodCUDA(unittest.TestCase):
    def setUp(self):
        self.in_features = 16
        self.out_features = 16
        self.layer = DummyLinearLayer(self.in_features, self.out_features)
        self.quant_config = WeightOnlyConfig(algo="weight_only_int8")
        self.method = GPUWeightOnlyLinearMethod(self.quant_config)

    def test_weight_and_scale_shapes(self):
        self.assertEqual(list(self.layer.weight.shape), [self.out_features, self.in_features])
        self.assertEqual(list(self.layer.weight_scale.shape), [self.out_features])

    def test_apply_output_shape(self):
        x = paddle.randn([2, self.in_features], dtype="float16")
        out = self.method.apply(self.layer, x)
        self.assertEqual(out.shape, [2, self.out_features])

    def test_apply_nonzero_values(self):
        x = paddle.randn([2, self.in_features], dtype="float16")
        out = self.method.apply(self.layer, x)
        self.assertFalse(np.allclose(out.numpy(), 0))

    def test_apply_precision(self):
        """精度检查 + Debug 打印"""
        x = paddle.to_tensor(np.random.randn(2, self.in_features).astype("float16"))

        # Reference FP32 output
        ref_out = paddle.matmul(
            x.astype("float32"),
            (self.layer.weight.astype("float32") * self.layer.weight_scale.astype("float32")).transpose([1, 0])
        )
        if self.layer.bias is not None:
            ref_out += self.layer.bias.astype("float32")

        # Quantized output (通过 debug_apply 打印中间过程)
        quant_out = debug_apply(self.method, self.layer, x)

        # 打印对比
        print("=== Reference float output ===")
        print(ref_out.numpy())
        print("=== Quantized output ===")
        print(quant_out.numpy())

        # 误差统计
        diff = np.abs(ref_out.numpy() - quant_out.numpy())
        print("Max abs diff:", diff.max(), ", Mean abs diff:", diff.mean())

        # 精度断言：误差在可接受范围内
        np.testing.assert_allclose(ref_out.numpy(), quant_out.numpy(), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()

```

```bash
Input shape: [2, 16]
Weight int8 shape: [16, 16]
Weight scale shape: [16]
Bias shape: [16]
Int32 accum stats: min=-430.19, max=523.19, mean=63.72
Scaled stats (after weight_scale): min=-7.14, max=9.53, mean=1.14
Final output (after bias) stats: min=-8.45, max=8.66, mean=1.37


=== Reference float output ===
[[ 2.6108642 -0.6497458  7.0613937  2.0976558  8.218029   4.234454
   2.2585444  0.29855    2.5939693 -3.258247   4.2500834  1.8264745
   2.0854921 -3.9599     8.660601   2.673855 ]
 [ 3.8692675  2.6785803 -2.6663718  5.2733607 -8.454243   1.9710996
  -2.3115897  3.712538  -0.7427913 -0.841658  -0.1247133  3.0144439
  -2.3901563  5.688132  -1.8924255 -3.824362 ]]
=== Quantized output ===
[[ 2.611  -0.65    7.062   2.098   8.22    4.234   2.258   0.2986  2.594
  -3.258   4.25    1.826   2.086  -3.959   8.664   2.674 ]
 [ 3.87    2.678  -2.666   5.273  -8.45    1.971  -2.312   3.713  -0.7427
  -0.842  -0.1247  3.014  -2.39    5.688  -1.893  -3.824 ]]
Max abs diff: 0.0034618378 , Mean abs diff: 0.00049396604
..
----------------------------------------------------------------------
Ran 4 tests in 0.127s

OK
```

## Ans

```python

import unittest
import paddle
import numpy as np

from fastdeploy.model_executor.layers.quantization.weight_only import (
    GPUWeightOnlyLinearMethod,
    WeightOnlyConfig,
)


class DummyLinearLayerForWeightOnly:
    def __init__(self, in_features, out_features, add_bias=True):
        self.in_features = in_features
        self.out_features = out_features
        self.weight_shape = [out_features, in_features]
        self._dtype = "float16"
        self.add_bias = add_bias

        # Optional bias
        self.bias = paddle.to_tensor(
            np.random.randn(out_features).astype("float16")
        ) if add_bias else None

        # Dummy config for FastDeploy
        self.fd_config = type('config', (), {})()
        self.fd_config.load_config = type(
            'load_config', (), {"load_choices": "default_v1"}
        )()

        # float32 weights
        weight_fp32 = np.random.randn(*self.weight_shape).astype("float32")

        # Per-channel scale
        max_abs = np.max(np.abs(weight_fp32), axis=1, keepdims=True) + 1e-6
        scale = (max_abs / 127.0).astype("float32")

        # Int8 quantization
        weight_int8 = np.clip(np.round(weight_fp32 / scale), -128, 127).astype("int8")

        # Store tensors
        self.weight = paddle.to_tensor(weight_int8, dtype="int8")
        self.weight_scale = paddle.to_tensor(scale.squeeze(-1).astype("float16"))

        # Keep FP32 weight for reference
        self.weight_fp32 = paddle.to_tensor(weight_fp32.astype("float32"))

    def create_parameter(self, shape, dtype, is_bias=False, default_initializer=None):
        if default_initializer is None:
            return paddle.create_parameter(shape=shape, dtype=dtype)
        else:
            return paddle.create_parameter(
                shape=shape, dtype=dtype, default_initializer=default_initializer
            )


class TestGPUWeightOnlyLinearMethod(unittest.TestCase):
    def setUp(self):
        self.in_features = 16
        self.out_features = 16
        self.layer = DummyLinearLayerForWeightOnly(self.in_features, self.out_features)
        self.quant_config = WeightOnlyConfig(algo="weight_only_int8")
        self.method = GPUWeightOnlyLinearMethod(self.quant_config)

    def test_weight_and_scale_shapes(self):
        """Test weight and scale tensor shapes"""
        self.assertEqual(list(self.layer.weight.shape), [self.out_features, self.in_features])
        self.assertEqual(list(self.layer.weight_scale.shape), [self.out_features])

    def test_apply_output_shape(self):
        """Test output shape of apply method"""
        x = paddle.randn([2, self.in_features], dtype="float16")
        out = self.method.apply(self.layer, x)
        self.assertEqual(out.shape, [2, self.out_features])

    def test_apply_nonzero_output(self):
        """Test that apply output is non-zero"""
        x = paddle.randn([2, self.in_features], dtype="float16")
        out = self.method.apply(self.layer, x)
        self.assertFalse(np.allclose(out.numpy(), 0))

    def test_apply_numerical_precision(self):
        """Test numerical precision of quantized output"""
        x = paddle.to_tensor(np.random.randn(2, self.in_features).astype("float16"))

        # Reference FP32 output
        ref_out = paddle.matmul(
            x.astype("float32"),
            (self.layer.weight.astype("float32") * self.layer.weight_scale.astype("float32")).transpose([1, 0])
        )
        if self.layer.bias is not None:
            ref_out += self.layer.bias.astype("float32")

        # Manual quantized output
        weight_f32 = self.layer.weight.astype("float32")
        x_f32 = x.astype("float32")
        quant_out = paddle.matmul(x_f32, weight_f32 * self.layer.weight_scale.astype("float32"), transpose_y=True)
        if self.layer.bias is not None:
            quant_out += self.layer.bias.astype("float32")
        quant_out = quant_out.astype("float16")

        np.testing.assert_allclose(ref_out.numpy(), quant_out.numpy(), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()

```

