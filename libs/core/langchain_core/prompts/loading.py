"""提示模板加载模块。

本模块提供从文件加载和保存提示模板的功能。
支持 JSON 和 YAML 格式的提示配置文件。

核心功能:
---------
1. **load_prompt()**: 从文件加载提示模板
2. **load_prompt_from_config()**: 从配置字典加载提示模板

支持的提示类型:
---------
- "prompt": PromptTemplate（普通字符串模板）
- "few_shot": FewShotPromptTemplate（少样本学习模板）
- "chat": ChatPromptTemplate（聊天消息模板）

安全警告:
---------
**不再支持 jinja2 格式的模板加载**，因为它可能导致任意代码执行。
请将 jinja2 模板迁移到 f-string 格式。

**不再支持 lc:// Hub 路径**，旧的 GitHub 基础的 Hub 已废弃。
请使用新的 LangChain Hub: https://smith.langchain.com/hub

使用示例:
---------
>>> from langchain_core.prompts import load_prompt
>>>
>>> # 从 YAML 文件加载
>>> prompt = load_prompt("prompts/greeting.yaml")
>>>
>>> # 从 JSON 文件加载
>>> prompt = load_prompt("prompts/greeting.json")

配置文件格式示例 (YAML):
---------
```yaml
# greeting.yaml
_type: prompt
template: "你好，{name}！欢迎来到{place}。"
template_format: f-string
```

Few-shot 配置示例:
```yaml
_type: few_shot
prefix: "请完成以下任务:"
example_separator: "\\n\\n"
examples:
  - input: "2+2"
    output: "4"
example_prompt:
  _type: prompt
  template: "输入: {input}\\n输出: {output}"
suffix: "输入: {input}\\n输出:"
input_variables:
  - input
```
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path

import yaml

from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate

# 旧的 LangChain Hub URL（已废弃）
URL_BASE = "https://raw.githubusercontent.com/hwchase17/langchain-hub/master/prompts/"

# 日志记录器
logger = logging.getLogger(__name__)


def load_prompt_from_config(config: dict) -> BasePromptTemplate:
    """从配置字典加载提示模板。

    这是加载提示的核心函数，根据配置中的 '_type' 字段
    选择合适的加载器创建对应类型的提示模板。

    Args:
        config: 包含提示配置的字典。
            必须包含以下字段之一:
            - _type: 提示类型，可选 "prompt"、"few_shot"、"chat"
            - template: 模板字符串（对于 "prompt" 类型）

    Returns:
        加载的 PromptTemplate 对象。

    Raises:
        ValueError: 如果提示类型不支持。

    示例:
        >>> config = {
        ...     "_type": "prompt",
        ...     "template": "你好，{name}！"
        ... }
        >>> prompt = load_prompt_from_config(config)
    """
    # 检查是否有 _type 字段
    if "_type" not in config:
        logger.warning("未找到 `_type` 字段，默认使用 `prompt`。")

    # 获取并移除类型字段
    config_type = config.pop("_type", "prompt")

    # 检查是否支持该类型
    if config_type not in type_to_loader_dict:
        msg = f"不支持加载 {config_type} 类型的提示"
        raise ValueError(msg)

    # 获取对应的加载器并加载
    prompt_loader = type_to_loader_dict[config_type]
    return prompt_loader(config)


def _load_template(var_name: str, config: dict) -> dict:
    """如果适用，从路径加载模板。

    这是一个内部辅助函数，处理模板的路径加载逻辑。
    如果配置中存在 `{var_name}_path`，则从该文件加载内容。

    Args:
        var_name: 变量名（如 "template"、"prefix"、"suffix"）。
        config: 配置字典。

    Returns:
        更新后的配置字典。

    Raises:
        ValueError: 如果同时提供了 {var_name} 和 {var_name}_path。
        ValueError: 如果文件格式不是 .txt。
    """
    # 检查是否存在路径配置
    if f"{var_name}_path" in config:
        # 确保不能同时提供值和路径
        if var_name in config:
            msg = f"不能同时提供 `{var_name}_path` 和 `{var_name}`。"
            raise ValueError(msg)

        # 获取模板路径
        template_path = Path(config.pop(f"{var_name}_path"))

        # 加载模板（只支持 .txt 格式）
        if template_path.suffix == ".txt":
            template = template_path.read_text(encoding="utf-8")
        else:
            raise ValueError

        # 设置模板值
        config[var_name] = template

    return config


def _load_examples(config: dict) -> dict:
    """如果需要，加载示例。

    处理 few-shot 提示中的示例加载。
    示例可以是列表，也可以是指向 JSON/YAML 文件的路径。

    Args:
        config: 配置字典。

    Returns:
        更新后的配置字典。

    Raises:
        ValueError: 如果示例文件格式不是 json 或 yaml。
        ValueError: 如果示例格式不是列表或字符串。
    """
    if isinstance(config["examples"], list):
        # 已经是列表，无需加载
        pass
    elif isinstance(config["examples"], str):
        # 从文件加载
        path = Path(config["examples"])
        with path.open(encoding="utf-8") as f:
            if path.suffix == ".json":
                examples = json.load(f)
            elif path.suffix in {".yaml", ".yml"}:
                examples = yaml.safe_load(f)
            else:
                msg = "无效的文件格式。只支持 json 或 yaml 格式。"
                raise ValueError(msg)
        config["examples"] = examples
    else:
        msg = "无效的示例格式。只支持列表或字符串。"
        raise ValueError(msg)  # noqa:TRY004

    return config


def _load_output_parser(config: dict) -> dict:
    """加载输出解析器。

    如果配置中包含 output_parser 配置，则创建对应的解析器。
    目前只支持 "default" 类型（StrOutputParser）。

    Args:
        config: 配置字典。

    Returns:
        更新后的配置字典。

    Raises:
        ValueError: 如果输出解析器类型不支持。
    """
    if _config := config.get("output_parser"):
        if output_parser_type := _config.get("_type") != "default":
            msg = f"不支持的输出解析器类型: {output_parser_type}"
            raise ValueError(msg)
        config["output_parser"] = StrOutputParser(**_config)
    return config


def _load_few_shot_prompt(config: dict) -> FewShotPromptTemplate:
    """从配置加载 Few-shot 提示模板。

    加载包含示例的 few-shot 学习提示模板。

    配置结构:
        - prefix: 示例前的说明文本（可选）
        - suffix: 示例后的模板（必需）
        - example_prompt: 单个示例的模板
        - examples: 示例列表或文件路径
        - example_separator: 示例分隔符

    Args:
        config: 配置字典。

    Returns:
        加载的 FewShotPromptTemplate。
    """
    # 加载 suffix 和 prefix 模板
    config = _load_template("suffix", config)
    config = _load_template("prefix", config)

    # 加载示例提示模板
    if "example_prompt_path" in config:
        # 从文件路径加载
        if "example_prompt" in config:
            msg = (
                "只能指定 example_prompt 和 example_prompt_path 中的一个。"
            )
            raise ValueError(msg)
        config["example_prompt"] = load_prompt(config.pop("example_prompt_path"))
    else:
        # 从内联配置加载
        config["example_prompt"] = load_prompt_from_config(config["example_prompt"])

    # 加载示例
    config = _load_examples(config)
    config = _load_output_parser(config)

    return FewShotPromptTemplate(**config)


def _load_prompt(config: dict) -> PromptTemplate:
    """从配置加载普通提示模板。

    Args:
        config: 配置字典。

    Returns:
        加载的 PromptTemplate。

    Raises:
        ValueError: 如果模板格式是 jinja2（出于安全原因不再支持）。
    """
    # 如果需要，从文件加载模板
    config = _load_template("template", config)
    config = _load_output_parser(config)

    # 安全检查：禁止 jinja2 格式
    template_format = config.get("template_format", "f-string")
    if template_format == "jinja2":
        # 由于安全问题已禁用
        # 参考: https://github.com/langchain-ai/langchain/issues/4394
        msg = (
            f"不再支持加载 '{template_format}' 格式的模板，"
            f"因为它可能导致任意代码执行。请迁移到使用 'f-string' 格式，"
            f"该格式不存在此问题。"
        )
        raise ValueError(msg)

    return PromptTemplate(**config)


def load_prompt(path: str | Path, encoding: str | None = None) -> BasePromptTemplate:
    """从本地文件加载提示模板的统一方法。

    这是加载提示文件的主要入口函数。支持 JSON 和 YAML 格式。

    Args:
        path: 提示文件的路径。
        encoding: 文件编码。如果未指定，使用系统默认编码。

    Returns:
        加载的 PromptTemplate 对象。

    Raises:
        RuntimeError: 如果路径是旧的 LangChain Hub 路径（lc://）。
        ValueError: 如果文件格式不支持。

    使用示例:
        >>> # 从 YAML 文件加载
        >>> prompt = load_prompt("prompts/greeting.yaml")
        >>>
        >>> # 从 JSON 文件加载，指定编码
        >>> prompt = load_prompt("prompts/greeting.json", encoding="utf-8")
        >>>
        >>> # 在加载后使用
        >>> result = prompt.format(name="世界")
    """
    # 检查是否是旧的 Hub 路径
    if isinstance(path, str) and path.startswith("lc://"):
        msg = (
            "不再支持从已废弃的 GitHub 基础 Hub 加载。"
            "请使用新的 LangChain Hub: https://smith.langchain.com/hub"
        )
        raise RuntimeError(msg)

    return _load_prompt_from_file(path, encoding)


def _load_prompt_from_file(
    file: str | Path, encoding: str | None = None
) -> BasePromptTemplate:
    """从文件加载提示模板。

    这是 load_prompt() 的内部实现。

    Args:
        file: 文件路径。
        encoding: 文件编码。

    Returns:
        加载的 BasePromptTemplate。

    Raises:
        ValueError: 如果文件类型不支持。
    """
    # 转换为 Path 对象
    file_path = Path(file)

    # 根据文件后缀加载
    if file_path.suffix == ".json":
        with file_path.open(encoding=encoding) as f:
            config = json.load(f)
    elif file_path.suffix.endswith((".yaml", ".yml")):
        with file_path.open(encoding=encoding) as f:
            config = yaml.safe_load(f)
    else:
        msg = f"不支持的文件类型: {file_path.suffix}"
        raise ValueError(msg)

    # 从配置创建提示
    return load_prompt_from_config(config)


def _load_chat_prompt(config: dict) -> ChatPromptTemplate:
    """从配置加载聊天提示模板。

    这是一个简化的加载器，从配置中提取消息模板
    并创建 ChatPromptTemplate。

    Args:
        config: 配置字典。

    Returns:
        加载的 ChatPromptTemplate。

    Raises:
        ValueError: 如果无法加载聊天提示（缺少模板）。
    """
    messages = config.pop("messages")
    template = messages[0]["prompt"].pop("template") if messages else None
    config.pop("input_variables")

    if not template:
        msg = "无法加载没有模板的聊天提示"
        raise ValueError(msg)

    return ChatPromptTemplate.from_template(template=template, **config)


# 类型到加载器的映射字典
# 根据配置中的 _type 字段选择对应的加载函数
type_to_loader_dict: dict[str, Callable[[dict], BasePromptTemplate]] = {
    "prompt": _load_prompt,      # 普通字符串模板
    "few_shot": _load_few_shot_prompt,  # Few-shot 学习模板
    "chat": _load_chat_prompt,   # 聊天消息模板
}
