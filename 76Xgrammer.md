## 1 源码

 **根据不同输入类型（schema）判断格式是否正确，并返回合法结果或错误信息。**

支持的 6 种输入类型（schema）：

1. **guided_json** → JSON Schema 格式（字符串或字典）
2. **guided_json_object** → 特殊 JSON Schema，对象类型
3. **guided_grammar** → 文法规则（EBNF 格式）
4. **guided_choice** → 一组选项（自动转成文法规则）
5. **structural_tag** → 结构化标签
6. **regex** → 正则表达式（不做校验）

**JSON Schema**：数据结构规范

**EBNF**：扩展巴科斯-诺尔范式

**Trigger**：解析触发器

**Regex**：正则表达式

## 2 分析

**源码分支逻辑**：

- guided_json → JSON Schema 校验
- guided_json_object → 固定对象
- guided_grammar → 文法校验
- guided_choice → 选项转文法
- structural_tag → 标签校验
- regex → 透传

**单测思路**：

- 每个分支提供合法与非法数据
- 合法时 `err=None`，非法时 `err!=None`
- 特殊情况（unsupported JSON）只检查 string 化



```python
def schema_format(self, request: Request):
```

根据 `Request` 对象中的字段进入不同分支。



### 2.1 guided_json

```python
if request.guided_json:
    if not isinstance(request.guided_json, str):
        guided_json = json.dumps(request.guided_json)
    else:
        guided_json = request.guided_json
    try:
        Grammar.from_json_schema(guided_json, any_whitespace=self.any_whitespace)
    except RuntimeError as e:
        return request, f"Invalid JSON format: {guided_json}, error message: {e!s}"
    if self._unsupported_json_schema(guided_json):
        return request, f"unsupported JSON schema: {guided_json}"
    request.guided_json = guided_json
    return request, None
```

确认 JSON 格式是否正确、判断是否包含不支持的特性、转成字符串存储。

JSON：键值对结构的通用数据格式、Schema规定 JSON 数据结构的规范、_unsupported_json_schema：内部检查器，检测是否用了不支持的特性（如 `uniqueItems`、`format`）



### 2.2 guided_json_object

```python
elif request.guided_json_object:
    request.guided_json = '{"type": "object"}'
    return request, None
```

快速生成 JSON Schema：始终为对象类型，不做复杂校验。



### 2.3 guided_grammar

```python
elif request.guided_grammar:
    try:
        Grammar.from_ebnf(request.guided_grammar)
    except RuntimeError as e:
        return request, f"Invalid grammar format: {request.guided_grammar}, error message: {e!s}"
    return request, None
```

检查文法规则（EBNF 格式）是否合法

Grammar：文法，用来描述语言结构

EBNF：扩展巴科斯-诺尔范式，一种描述语法的正式语言



### 2.4 guided_choice

```python
elif request.guided_choice:
    try:
        escaped_choices = (re.sub(r'(["\\])', r"\\\1", c) for c in request.guided_choice)
        guided_choice = "root ::= " + " | ".join(f'"{c}"' for c in escaped_choices)
        Grammar.from_ebnf(guided_choice)
    except RuntimeError as e:
        return request, f"Invalid choice format: {guided_choice}, error message: {e!s}"
    request.guided_grammar = guided_choice
    return request, None
```

把一组选项转成 EBNF 文法，校验文法合法性

输入：`["yes", "no"]`

转换：`root ::= "yes" | "no"`



### 2.5 structural_tag

```python
elif request.structural_tag:
    try:
        structural_tag = json.loads(request.structural_tag)
        tags = [
            StructuralTagItem(
                begin=s["begin"],
                schema=json.dumps(s["schema"]),
                end=s["end"],
            )
            for s in structural_tag["structures"]
        ]
        Grammar.from_structural_tag(tags, structural_tag["triggers"])
    except RuntimeError as e:
        return request, f"Invalid structural_tag format: {structural_tag}, error message: {e!s}"
    return request, None
```

解析结构化标签 JSON，校验结构是否正确

StructuralTagItem：结构化标签对象，定义起始符、结束符和内容格式

Trigger：触发器，用于标记文本解析的起点



### 2.6 regex

```python
else:
    # regex is not format
```

regex（正则表达式）直接透传，不做任何校验



## 3 设计

| 输入类型           | 合法示例                                  | 非法示例                               | 预期结果                          |
| ------------------ | ----------------------------------------- | -------------------------------------- | --------------------------------- |
| guided_json        | {"type": "string"}                        | {"type": "unknown_type"}               | 合法 → None，非法 → err           |
| guided_json_object | True                                      | -                                      | 始终合法                          |
| guided_grammar     | `'root ::= "yes" | "no"'`                 | `"root :="`                            | 合法 → None，非法 → err           |
| guided_choice      | `["yes","no"]`                            | `[None]`                               | 合法 → grammar，非法 → err        |
| structural_tag     | `{"structures":[...],"triggers":["<a>"]}` | `{"structures":[{"begin":"<a>",...}]}` | triggers 缺失 → err               |
| regex              | `"^[a-z]+$"`                              | -                                      | 始终合法                          |
| unsupported JSON   | `{"type":"array","uniqueItems":true}`     | -                                      | string 化，不报错（源码逻辑如此） |



## 4 Code

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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the
# specific language governing permissions and limitations under the License.


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



# CI不通过_8.29

```
___
ERROR
collecting
model_executor/guided_decoding/test_xgrammar_checker.py
model_executor/guided_decoding/test_xgrammar_checker.py
Error: Process completed with exit code 8.
```

log :

```
model_executor/guided_decoding/test_xgrammar_checker.py:20: in <module>
    from fastdeploy.model_executor.guided_decoding.xgrammar_backend import XGrammarChecker
/usr/local/lib/python3.10/site-packages/fastdeploy/model_executor/guided_decoding/xgrammar_backend.py:23: in <module>
    import torch
E   ModuleNotFoundError: No module named 'torch'
```

