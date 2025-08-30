# Source code

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



# First Try

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
