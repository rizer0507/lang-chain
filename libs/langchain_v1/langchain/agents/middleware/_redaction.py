"""脱敏工具模块（内部使用）。

本模块提供中间件组件共享的脱敏功能。

核心功能:
---------
- `RedactionStrategy`: 脱敏策略类型
- `PIIMatch`: 敏感数据匹配结果
- `PIIDetectionError`: PII 检测异常
- 内置检测器: email, credit_card, ip, mac_address, url
- `RedactionRule`: 脱敏规则配置
"""

from __future__ import annotations

import hashlib
import ipaddress
import operator
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

from typing_extensions import TypedDict

RedactionStrategy = Literal["block", "redact", "mask", "hash"]
"""Supported strategies for handling detected sensitive values.

中文翻译:
支持处理检测到的敏感值的策略。"""


class PIIMatch(TypedDict):
    """Represents an individual match of sensitive data.

    中文翻译:
    代表敏感数据的单独匹配。"""

    type: str
    value: str
    start: int
    end: int


class PIIDetectionError(Exception):
    """Raised when configured to block on detected sensitive values.

    中文翻译:
    当配置为阻止检测到的敏感值时引发。"""

    def __init__(self, pii_type: str, matches: Sequence[PIIMatch]) -> None:
        """Initialize the exception with match context.

        Args:
            pii_type: Name of the detected sensitive type.
            matches: All matches that were detected for that type.
        

        中文翻译:
        使用匹配上下文初始化异常。
        参数：
            pii_type：检测到的敏感类型的名称。
            匹配：针对该类型检测到的所有匹配。"""
        self.pii_type = pii_type
        self.matches = list(matches)
        count = len(matches)
        msg = f"Detected {count} instance(s) of {pii_type} in text content"
        super().__init__(msg)


Detector = Callable[[str], list[PIIMatch]]
"""Callable signature for detectors that locate sensitive values.

中文翻译:
定位敏感值的检测器的可调用签名。"""


def detect_email(content: str) -> list[PIIMatch]:
    """Detect email addresses in content.

    中文翻译:
    检测内容中的电子邮件地址。"""
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    return [
        PIIMatch(
            type="email",
            value=match.group(),
            start=match.start(),
            end=match.end(),
        )
        for match in re.finditer(pattern, content)
    ]


def detect_credit_card(content: str) -> list[PIIMatch]:
    """Detect credit card numbers in content using Luhn validation.

    中文翻译:
    使用 Luhn 验证检测内容中的信用卡号。"""
    pattern = r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"
    matches = []

    for match in re.finditer(pattern, content):
        card_number = match.group()
        if _passes_luhn(card_number):
            matches.append(
                PIIMatch(
                    type="credit_card",
                    value=card_number,
                    start=match.start(),
                    end=match.end(),
                )
            )

    return matches


def detect_ip(content: str) -> list[PIIMatch]:
    """Detect IPv4 or IPv6 addresses in content.

    中文翻译:
    检测内容中的 IPv4 或 IPv6 地址。"""
    matches: list[PIIMatch] = []
    ipv4_pattern = r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"

    for match in re.finditer(ipv4_pattern, content):
        ip_candidate = match.group()
        try:
            ipaddress.ip_address(ip_candidate)
        except ValueError:
            continue
        matches.append(
            PIIMatch(
                type="ip",
                value=ip_candidate,
                start=match.start(),
                end=match.end(),
            )
        )

    return matches


def detect_mac_address(content: str) -> list[PIIMatch]:
    """Detect MAC addresses in content.

    中文翻译:
    检测内容中的 MAC 地址。"""
    pattern = r"\b([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"
    return [
        PIIMatch(
            type="mac_address",
            value=match.group(),
            start=match.start(),
            end=match.end(),
        )
        for match in re.finditer(pattern, content)
    ]


def detect_url(content: str) -> list[PIIMatch]:
    """Detect URLs in content using regex and stdlib validation.

    中文翻译:
    使用正则表达式和 stdlib 验证检测内容中的 URL。"""
    matches: list[PIIMatch] = []

    # Pattern 1: URLs with scheme (http:// or https://)
    # 中文: 模式 1：带有方案的 URL（http:// 或 https://）
    scheme_pattern = r"https?://[^\s<>\"{}|\\^`\[\]]+"

    for match in re.finditer(scheme_pattern, content):
        url = match.group()
        result = urlparse(url)
        if result.scheme in {"http", "https"} and result.netloc:
            matches.append(
                PIIMatch(
                    type="url",
                    value=url,
                    start=match.start(),
                    end=match.end(),
                )
            )

    # Pattern 2: URLs without scheme (www.example.com or example.com/path)
    # 中文: 模式 2：没有方案的 URL（www.example.com 或 example.com/path）
    # More conservative to avoid false positives
    # 中文: 更加保守以避免误报
    bare_pattern = (
        r"\b(?:www\.)?[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
        r"(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+(?:/[^\s]*)?"
    )

    for match in re.finditer(bare_pattern, content):
        start, end = match.start(), match.end()
        # Skip if already matched with scheme
        # 中文: 如果已经与方案匹配则跳过
        if any(m["start"] <= start < m["end"] or m["start"] < end <= m["end"] for m in matches):
            continue

        url = match.group()
        # Only accept if it has a path or starts with www
        # 中文: 仅当有路径或以 www 开头时才接受
        # This reduces false positives like "example.com" in prose
        # 中文: 这减少了散文中的误报，例如“example.com”
        if "/" in url or url.startswith("www."):
            # Add scheme for validation (required for urlparse to work correctly)
            # 中文: 添加验证方案（urlparse 正常工作所需）
            test_url = f"http://{url}"
            result = urlparse(test_url)
            if result.netloc and "." in result.netloc:
                matches.append(
                    PIIMatch(
                        type="url",
                        value=url,
                        start=start,
                        end=end,
                    )
                )

    return matches


BUILTIN_DETECTORS: dict[str, Detector] = {
    "email": detect_email,
    "credit_card": detect_credit_card,
    "ip": detect_ip,
    "mac_address": detect_mac_address,
    "url": detect_url,
}
"""Registry of built-in detectors keyed by type name.

中文翻译:
按类型名称键入的内置检测器注册表。"""

_CARD_NUMBER_MIN_DIGITS = 13
_CARD_NUMBER_MAX_DIGITS = 19


def _passes_luhn(card_number: str) -> bool:
    """Validate credit card number using the Luhn checksum.

    中文翻译:
    使用 Luhn 校验和验证信用卡号。"""
    digits = [int(d) for d in card_number if d.isdigit()]
    if not _CARD_NUMBER_MIN_DIGITS <= len(digits) <= _CARD_NUMBER_MAX_DIGITS:
        return False

    checksum = 0
    for index, digit in enumerate(reversed(digits)):
        value = digit
        if index % 2 == 1:
            value *= 2
            if value > 9:  # noqa: PLR2004
                value -= 9
        checksum += value
    return checksum % 10 == 0


def _apply_redact_strategy(content: str, matches: list[PIIMatch]) -> str:
    result = content
    for match in sorted(matches, key=operator.itemgetter("start"), reverse=True):
        replacement = f"[REDACTED_{match['type'].upper()}]"
        result = result[: match["start"]] + replacement + result[match["end"] :]
    return result


_UNMASKED_CHAR_NUMBER = 4
_IPV4_PARTS_NUMBER = 4


def _apply_mask_strategy(content: str, matches: list[PIIMatch]) -> str:
    result = content
    for match in sorted(matches, key=operator.itemgetter("start"), reverse=True):
        value = match["value"]
        pii_type = match["type"]
        if pii_type == "email":
            parts = value.split("@")
            if len(parts) == 2:  # noqa: PLR2004
                domain_parts = parts[1].split(".")
                masked = (
                    f"{parts[0]}@****.{domain_parts[-1]}"
                    if len(domain_parts) > 1
                    else f"{parts[0]}@****"
                )
            else:
                masked = "****"
        elif pii_type == "credit_card":
            digits_only = "".join(c for c in value if c.isdigit())
            separator = "-" if "-" in value else " " if " " in value else ""
            if separator:
                masked = (
                    f"****{separator}****{separator}****{separator}"
                    f"{digits_only[-_UNMASKED_CHAR_NUMBER:]}"
                )
            else:
                masked = f"************{digits_only[-_UNMASKED_CHAR_NUMBER:]}"
        elif pii_type == "ip":
            octets = value.split(".")
            masked = f"*.*.*.{octets[-1]}" if len(octets) == _IPV4_PARTS_NUMBER else "****"
        elif pii_type == "mac_address":
            separator = ":" if ":" in value else "-"
            masked = (
                f"**{separator}**{separator}**{separator}**{separator}**{separator}{value[-2:]}"
            )
        elif pii_type == "url":
            masked = "[MASKED_URL]"
        else:
            masked = (
                f"****{value[-_UNMASKED_CHAR_NUMBER:]}"
                if len(value) > _UNMASKED_CHAR_NUMBER
                else "****"
            )
        result = result[: match["start"]] + masked + result[match["end"] :]
    return result


def _apply_hash_strategy(content: str, matches: list[PIIMatch]) -> str:
    result = content
    for match in sorted(matches, key=operator.itemgetter("start"), reverse=True):
        digest = hashlib.sha256(match["value"].encode()).hexdigest()[:8]
        replacement = f"<{match['type']}_hash:{digest}>"
        result = result[: match["start"]] + replacement + result[match["end"] :]
    return result


def apply_strategy(
    content: str,
    matches: list[PIIMatch],
    strategy: RedactionStrategy,
) -> str:
    """Apply the configured strategy to matches within content.

    中文翻译:
    将配置的策略应用于内容内的匹配。"""
    if not matches:
        return content
    if strategy == "redact":
        return _apply_redact_strategy(content, matches)
    if strategy == "mask":
        return _apply_mask_strategy(content, matches)
    if strategy == "hash":
        return _apply_hash_strategy(content, matches)
    if strategy == "block":
        raise PIIDetectionError(matches[0]["type"], matches)
    msg = f"Unknown redaction strategy: {strategy}"
    raise ValueError(msg)


def resolve_detector(pii_type: str, detector: Detector | str | None) -> Detector:
    """Return a callable detector for the given configuration.

    中文翻译:
    返回给定配置的可调用检测器。"""
    if detector is None:
        if pii_type not in BUILTIN_DETECTORS:
            msg = (
                f"Unknown PII type: {pii_type}. "
                f"Must be one of {list(BUILTIN_DETECTORS.keys())} or provide a custom detector."
            )
            raise ValueError(msg)
        return BUILTIN_DETECTORS[pii_type]
    if isinstance(detector, str):
        pattern = re.compile(detector)

        def regex_detector(content: str) -> list[PIIMatch]:
            return [
                PIIMatch(
                    type=pii_type,
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                )
                for match in pattern.finditer(content)
            ]

        return regex_detector
    return detector


@dataclass(frozen=True)
class RedactionRule:
    """Configuration for handling a single PII type.

    中文翻译:
    用于处理单个 PII 类型的配置。"""

    pii_type: str
    strategy: RedactionStrategy = "redact"
    detector: Detector | str | None = None

    def resolve(self) -> ResolvedRedactionRule:
        """Resolve runtime detector and return an immutable rule.

        中文翻译:
        解析运行时检测器并返回不可变规则。"""
        resolved_detector = resolve_detector(self.pii_type, self.detector)
        return ResolvedRedactionRule(
            pii_type=self.pii_type,
            strategy=self.strategy,
            detector=resolved_detector,
        )


@dataclass(frozen=True)
class ResolvedRedactionRule:
    """Resolved redaction rule ready for execution.

    中文翻译:
    已解决的密文规则可供执行。"""

    pii_type: str
    strategy: RedactionStrategy
    detector: Detector

    def apply(self, content: str) -> tuple[str, list[PIIMatch]]:
        """Apply this rule to content, returning new content and matches.

        中文翻译:
        将此规则应用于内容，返回新内容和匹配项。"""
        matches = self.detector(content)
        if not matches:
            return content, []
        updated = apply_strategy(content, matches, self.strategy)
        return updated, matches


__all__ = [
    "PIIDetectionError",
    "PIIMatch",
    "RedactionRule",
    "ResolvedRedactionRule",
    "apply_strategy",
    "detect_credit_card",
    "detect_email",
    "detect_ip",
    "detect_mac_address",
    "detect_url",
]
