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

