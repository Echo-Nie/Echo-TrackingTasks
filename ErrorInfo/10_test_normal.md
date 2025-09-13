## Error Info

```bash
======================================================================
ERROR: test_api (__main__.TestNormalAPI)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/aistudio/test.py", line 143, in test_api
    ret_static = self.static_api()
  File "/home/aistudio/test.py", line 115, in static_api
    ret = exe.run(fetch_list=[out])
  File "/opt/conda/envs/python35-paddle120-env/lib/python3.10/site-packages/paddle/base/executor.py", line 1906, in run
    res = self._run_impl(
  File "/opt/conda/envs/python35-paddle120-env/lib/python3.10/site-packages/paddle/base/executor.py", line 2047, in _run_impl
    program, new_exe = self._executor_cache.get_program_and_executor(
  File "/opt/conda/envs/python35-paddle120-env/lib/python3.10/site-packages/paddle/base/executor.py", line 930, in get_program_and_executor
    self._CachedData(
  File "/opt/conda/envs/python35-paddle120-env/lib/python3.10/site-packages/paddle/base/executor.py", line 891, in __init__
    _get_strong_program_cache_key_for_new_exe(
  File "/opt/conda/envs/python35-paddle120-env/lib/python3.10/site-packages/paddle/base/executor.py", line 656, in _get_strong_program_cache_key_for_new_exe
    + _get_program_cache_key(feed, fetch_list)
  File "/opt/conda/envs/python35-paddle120-env/lib/python3.10/site-packages/paddle/base/executor.py", line 694, in _get_program_cache_key
    return str(_get_feed_fetch_var_names(feed, fetch_list))
  File "/opt/conda/envs/python35-paddle120-env/lib/python3.10/site-packages/paddle/base/executor.py", line 689, in _get_feed_fetch_var_names
    fetch_var_names = list(map(_to_name_str, fetch_list))
  File "/opt/conda/envs/python35-paddle120-env/lib/python3.10/site-packages/paddle/base/executor.py", line 642, in _to_name_str
    return _to_str(var)
  File "/opt/conda/envs/python35-paddle120-env/lib/python3.10/site-packages/paddle/base/executor.py", line 624, in _to_str
    return var.desc.name()
AttributeError: 'Tensor' object has no attribute 'desc'

======================================================================
FAIL: test_api (__main__.TestNormalAPIComplex)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/aistudio/test.py", line 350, in test_api
    ret_static = self.static_api()
  File "/home/aistudio/test.py", line 314, in static_api
    mean = paddle.static.data('Mean', (), 'complex128')
  File "/opt/conda/envs/python35-paddle120-env/lib/python3.10/site-packages/paddle/base/wrapped_decorator.py", line 50, in wrapper
    return decorated(*args, **kwargs)
  File "/opt/conda/envs/python35-paddle120-env/lib/python3.10/site-packages/paddle/base/framework.py", line 743, in __impl__
    assert not in_dygraph_mode(), (
AssertionError: In PaddlePaddle 2.x, we turn on dynamic graph mode by default, and 'data()' is only supported in static graph mode. So if you want to use this api, please call 'paddle.enable_static()' before this api to enter static graph mode.

======================================================================
FAIL: test_api (__main__.TestNormalAPIComplex_mean_is_tensor)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/aistudio/test.py", line 350, in test_api
    ret_static = self.static_api()
  File "/home/aistudio/test.py", line 283, in static_api
    mean = paddle.static.data(
  File "/opt/conda/envs/python35-paddle120-env/lib/python3.10/site-packages/paddle/base/wrapped_decorator.py", line 50, in wrapper
    return decorated(*args, **kwargs)
  File "/opt/conda/envs/python35-paddle120-env/lib/python3.10/site-packages/paddle/base/framework.py", line 743, in __impl__
    assert not in_dygraph_mode(), (
AssertionError: In PaddlePaddle 2.x, we turn on dynamic graph mode by default, and 'data()' is only supported in static graph mode. So if you want to use this api, please call 'paddle.enable_static()' before this api to enter static graph mode.

======================================================================
FAIL: test_api (__main__.TestNormalAPIComplex_mean_std_are_tensor)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/aistudio/test.py", line 350, in test_api
    ret_static = self.static_api()
  File "/home/aistudio/test.py", line 264, in static_api
    mean = paddle.static.data(
  File "/opt/conda/envs/python35-paddle120-env/lib/python3.10/site-packages/paddle/base/wrapped_decorator.py", line 50, in wrapper
    return decorated(*args, **kwargs)
  File "/opt/conda/envs/python35-paddle120-env/lib/python3.10/site-packages/paddle/base/framework.py", line 743, in __impl__
    assert not in_dygraph_mode(), (
AssertionError: In PaddlePaddle 2.x, we turn on dynamic graph mode by default, and 'data()' is only supported in static graph mode. So if you want to use this api, please call 'paddle.enable_static()' before this api to enter static graph mode.

======================================================================
FAIL: test_api (__main__.TestNormalAPIComplex_mean_std_are_tensor_with_different_dtype)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/aistudio/test.py", line 350, in test_api
    ret_static = self.static_api()
  File "/home/aistudio/test.py", line 264, in static_api
    mean = paddle.static.data(
  File "/opt/conda/envs/python35-paddle120-env/lib/python3.10/site-packages/paddle/base/wrapped_decorator.py", line 50, in wrapper
    return decorated(*args, **kwargs)
  File "/opt/conda/envs/python35-paddle120-env/lib/python3.10/site-packages/paddle/base/framework.py", line 743, in __impl__
    assert not in_dygraph_mode(), (
AssertionError: In PaddlePaddle 2.x, we turn on dynamic graph mode by default, and 'data()' is only supported in static graph mode. So if you want to use this api, please call 'paddle.enable_static()' before this api to enter static graph mode.

======================================================================
FAIL: test_api (__main__.TestNormalAPIComplex_std_is_tensor)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/aistudio/test.py", line 350, in test_api
    ret_static = self.static_api()
  File "/home/aistudio/test.py", line 295, in static_api
    mean = paddle.static.data('Mean', self.std.shape, 'complex128')
  File "/opt/conda/envs/python35-paddle120-env/lib/python3.10/site-packages/paddle/base/wrapped_decorator.py", line 50, in wrapper
    return decorated(*args, **kwargs)
  File "/opt/conda/envs/python35-paddle120-env/lib/python3.10/site-packages/paddle/base/framework.py", line 743, in __impl__
    assert not in_dygraph_mode(), (
AssertionError: In PaddlePaddle 2.x, we turn on dynamic graph mode by default, and 'data()' is only supported in static graph mode. So if you want to use this api, please call 'paddle.enable_static()' before this api to enter static graph mode.

======================================================================
FAIL: test_api (__main__.TestNormalAPI_mean_is_tensor)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/aistudio/test.py", line 143, in test_api
    ret_static = self.static_api()
  File "/home/aistudio/test.py", line 89, in static_api
    mean = paddle.static.data(
  File "/opt/conda/envs/python35-paddle120-env/lib/python3.10/site-packages/paddle/base/wrapped_decorator.py", line 50, in wrapper
    return decorated(*args, **kwargs)
  File "/opt/conda/envs/python35-paddle120-env/lib/python3.10/site-packages/paddle/base/framework.py", line 743, in __impl__
    assert not in_dygraph_mode(), (
AssertionError: In PaddlePaddle 2.x, we turn on dynamic graph mode by default, and 'data()' is only supported in static graph mode. So if you want to use this api, please call 'paddle.enable_static()' before this api to enter static graph mode.

======================================================================
FAIL: test_api (__main__.TestNormalAPI_mean_std_are_tensor)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/aistudio/test.py", line 143, in test_api
    ret_static = self.static_api()
  File "/home/aistudio/test.py", line 70, in static_api
    mean = paddle.static.data(
  File "/opt/conda/envs/python35-paddle120-env/lib/python3.10/site-packages/paddle/base/wrapped_decorator.py", line 50, in wrapper
    return decorated(*args, **kwargs)
  File "/opt/conda/envs/python35-paddle120-env/lib/python3.10/site-packages/paddle/base/framework.py", line 743, in __impl__
    assert not in_dygraph_mode(), (
AssertionError: In PaddlePaddle 2.x, we turn on dynamic graph mode by default, and 'data()' is only supported in static graph mode. So if you want to use this api, please call 'paddle.enable_static()' before this api to enter static graph mode.

======================================================================
FAIL: test_api (__main__.TestNormalAPI_mean_std_are_tensor_with_different_dtype)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/aistudio/test.py", line 143, in test_api
    ret_static = self.static_api()
  File "/home/aistudio/test.py", line 70, in static_api
    mean = paddle.static.data(
  File "/opt/conda/envs/python35-paddle120-env/lib/python3.10/site-packages/paddle/base/wrapped_decorator.py", line 50, in wrapper
    return decorated(*args, **kwargs)
  File "/opt/conda/envs/python35-paddle120-env/lib/python3.10/site-packages/paddle/base/framework.py", line 743, in __impl__
    assert not in_dygraph_mode(), (
AssertionError: In PaddlePaddle 2.x, we turn on dynamic graph mode by default, and 'data()' is only supported in static graph mode. So if you want to use this api, please call 'paddle.enable_static()' before this api to enter static graph mode.

======================================================================
FAIL: test_api (__main__.TestNormalAPI_std_is_tensor)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/aistudio/test.py", line 143, in test_api
    ret_static = self.static_api()
  File "/home/aistudio/test.py", line 101, in static_api
    std = paddle.static.data('Std', self.std.shape, self.std.dtype)
  File "/opt/conda/envs/python35-paddle120-env/lib/python3.10/site-packages/paddle/base/wrapped_decorator.py", line 50, in wrapper
    return decorated(*args, **kwargs)
  File "/opt/conda/envs/python35-paddle120-env/lib/python3.10/site-packages/paddle/base/framework.py", line 743, in __impl__
    assert not in_dygraph_mode(), (
AssertionError: In PaddlePaddle 2.x, we turn on dynamic graph mode by default, and 'data()' is only supported in static graph mode. So if you want to use this api, please call 'paddle.enable_static()' before this api to enter static graph mode.

----------------------------------------------------------------------
Ran 13 tests in 0.201s

FAILED (failures=9, errors=1)
```

