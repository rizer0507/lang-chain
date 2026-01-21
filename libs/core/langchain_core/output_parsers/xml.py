"""Output parser for XML format.

中文翻译:
XML 格式的输出解析器。"""

import contextlib
import re
import xml
import xml.etree.ElementTree as ET
from collections.abc import AsyncIterator, Iterator
from typing import Any, Literal
from xml.etree.ElementTree import TreeBuilder

from typing_extensions import override

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers.transform import BaseTransformOutputParser
from langchain_core.runnables.utils import AddableDict

try:
    from defusedxml import ElementTree  # type: ignore[import-untyped]
    from defusedxml.ElementTree import XMLParser  # type: ignore[import-untyped]

    _HAS_DEFUSEDXML = True
except ImportError:
    _HAS_DEFUSEDXML = False

XML_FORMAT_INSTRUCTIONS = """The output should be formatted as a XML file.
1. Output should conform to the tags below.
2. If tags are not given, make them on your own.
3. Remember to always open and close all the tags.

As an example, for the tags ["foo", "bar", "baz"]:
1. String "<foo>\n   <bar>\n      <baz></baz>\n   </bar>\n</foo>" is a well-formatted instance of the schema.
2. String "<foo>\n   <bar>\n   </foo>" is a badly-formatted instance.
3. String "<foo>\n   <tag>\n   </tag>\n</foo>" is a badly-formatted instance.

Here are the output tags:
```
{tags}
```

 中文翻译:
 输出应格式化为 XML 文件。
1. 输出应符合以下标签。
2. 如果没有给出标签，请自行制作。
3. 请记住始终打开和关闭所有标签。
例如，对于标签 ["foo", "bar", "baz"]：
1. 字符串“<foo>\n <bar>\n <baz></baz>\n </bar>\n</foo>”是格式正确的模式实例。
2. 字符串“<foo>\n <bar>\n </foo>”是格式错误的实例。
3. 字符串“<foo>\n <tag>\n </tag>\n</foo>”是格式错误的实例。
以下是输出标签：
````
{标签}
````"""  # noqa: E501


class _StreamingParser:
    """Streaming parser for XML.

    This implementation is pulled into a class to avoid implementation
    drift between transform and atransform of the `XMLOutputParser`.
    

    中文翻译:
    XML 的流式解析器。
    该实现被拉入一个类以避免实现
    “XMLOutputParser”的转换和转换之间的偏差。"""

    def __init__(self, parser: Literal["defusedxml", "xml"]) -> None:
        """Initialize the streaming parser.

        Args:
            parser: Parser to use for XML parsing. Can be either `'defusedxml'` or
                `'xml'`. See documentation in `XMLOutputParser` for more information.

        Raises:
            ImportError: If `defusedxml` is not installed and the `defusedxml` parser is
                requested.
        

        中文翻译:
        初始化流解析器。
        参数：
            parser：用于 XML 解析的解析器。可以是“defusedxml”或
                `'xml'`。有关更多信息，请参阅“XMLOutputParser”中的文档。
        加薪：
            ImportError: 如果未安装 `defusedxml` 并且 `defusedxml` 解析器已安装
                要求。"""
        if parser == "defusedxml":
            if not _HAS_DEFUSEDXML:
                msg = (
                    "defusedxml is not installed. "
                    "Please install it to use the defusedxml parser."
                    "You can install it with `pip install defusedxml` "
                )
                raise ImportError(msg)
            parser_ = XMLParser(target=TreeBuilder())
        else:
            parser_ = None
        self.pull_parser = ET.XMLPullParser(["start", "end"], _parser=parser_)
        self.xml_start_re = re.compile(r"<[a-zA-Z:_]")
        self.current_path: list[str] = []
        self.current_path_has_children = False
        self.buffer = ""
        self.xml_started = False

    def parse(self, chunk: str | BaseMessage) -> Iterator[AddableDict]:
        """Parse a chunk of text.

        Args:
            chunk: A chunk of text to parse. This can be a `str` or a `BaseMessage`.

        Yields:
            A `dict` representing the parsed XML element.

        Raises:
            xml.etree.ElementTree.ParseError: If the XML is not well-formed.
        

        中文翻译:
        解析一段文本。
        参数：
            chunk：要解析的文本块。这可以是“str”或“BaseMessage”。
        产量：
            表示已解析的 XML 元素的“dict”。
        加薪：
            xml.etree.ElementTree.ParseError：如果 XML 格式不正确。"""
        if isinstance(chunk, BaseMessage):
            # extract text
            # 中文: 提取文本
            chunk_content = chunk.content
            if not isinstance(chunk_content, str):
                # ignore non-string messages (e.g., function calls)
                # 中文: 忽略非字符串消息（例如函数调用）
                return
            chunk = chunk_content
        # add chunk to buffer of unprocessed text
        # 中文: 将块添加到未处理文本的缓冲区
        self.buffer += chunk
        # if xml string hasn't started yet, continue to next chunk
        # 中文: 如果 xml 字符串尚未开始，则继续下一个块
        if not self.xml_started:
            if match := self.xml_start_re.search(self.buffer):
                # if xml string has started, remove all text before it
                # 中文: 如果 xml 字符串已开始，则删除其之前的所有文本
                self.buffer = self.buffer[match.start() :]
                self.xml_started = True
            else:
                return
        # feed buffer to parser
        # 中文: 将缓冲区提供给解析器
        self.pull_parser.feed(self.buffer)
        self.buffer = ""
        # yield all events
        # 中文: 产生所有事件
        try:
            events = self.pull_parser.read_events()
            for event, elem in events:  # type: ignore[misc]
                if event == "start":
                    # update current path
                    # 中文: 更新当前路径
                    self.current_path.append(elem.tag)  # type: ignore[union-attr]
                    self.current_path_has_children = False
                elif event == "end":
                    # remove last element from current path
                    # 中文: 从当前路径中删除最后一个元素
                    #
                    self.current_path.pop()
                    #
                    中文: self.current_path.pop()
                    # yield element
                    # 中文: 屈服元素
                    if not self.current_path_has_children:
                        yield nested_element(self.current_path, elem)  # type: ignore[arg-type]
                    # prevent yielding of parent element
                    # 中文: 防止父元素的屈服
                    if self.current_path:
                        self.current_path_has_children = True
                    else:
                        self.xml_started = False
        except xml.etree.ElementTree.ParseError:
            # This might be junk at the end of the XML input.
            # 中文: XML 输入末尾的这可能是垃圾。
            # Let's check whether the current path is empty.
            # 中文: 让我们检查当前路径是否为空。
            if not self.current_path:
                # If it is empty, we can ignore this error.
                # 中文: 如果为空，我们可以忽略这个错误。
                return
            else:
                raise

    def close(self) -> None:
        """Close the parser.

        This should be called after all chunks have been parsed.
        

        中文翻译:
        关闭解析器。
        这应该在解析完​​所有块之后调用。"""
        # Ignore ParseError. This will ignore any incomplete XML at the end of the input
        # 中文: 忽略解析错误。这将忽略输入末尾任何不完整的 XML
        with contextlib.suppress(xml.etree.ElementTree.ParseError):
            self.pull_parser.close()


class XMLOutputParser(BaseTransformOutputParser):
    """Parse an output using xml format.

    Returns a dictionary of tags.
    

    中文翻译:
    使用 xml 格式解析输出。
    返回标签字典。"""

    tags: list[str] | None = None
    """Tags to tell the LLM to expect in the XML output.

    Note this may not be perfect depending on the LLM implementation.

    For example, with `tags=["foo", "bar", "baz"]`:

    1. A well-formatted XML instance:
        `"<foo>\n   <bar>\n      <baz></baz>\n   </bar>\n</foo>"`

    2. A badly-formatted XML instance (missing closing tag for 'bar'):
        `"<foo>\n   <bar>\n   </foo>"`

    3. A badly-formatted XML instance (unexpected 'tag' element):
        `"<foo>\n   <tag>\n   </tag>\n</foo>"`
    

    中文翻译:
    告诉 LLM 在 XML 输出中期望的标签。
    请注意，这可能并不完美，具体取决于法学硕士的实施情况。
    例如，使用 `tags=["foo", "bar", "baz"]`：
    1. 一个格式良好的 XML 实例：
        `"<foo>\n <bar>\n <baz></baz>\n </bar>\n</foo>"`
    2. 格式错误的 XML 实例（缺少“bar”的结束标记）：
        `"<foo>\n <bar>\n </foo>"`
    3. 格式错误的 XML 实例（意外的“tag”元素）：
        `"<foo>\n <tag>\n </tag>\n</foo>"`"""
    encoding_matcher: re.Pattern = re.compile(
        r"<([^>]*encoding[^>]*)>\n(.*)", re.MULTILINE | re.DOTALL
    )
    parser: Literal["defusedxml", "xml"] = "defusedxml"
    """Parser to use for XML parsing. Can be either `'defusedxml'` or `'xml'`.

    * `'defusedxml'` is the default parser and is used to prevent XML vulnerabilities
        present in some distributions of Python's standard library xml.
        `defusedxml` is a wrapper around the standard library parser that
        sets up the parser with secure defaults.
    * `'xml'` is the standard library parser.

    Use `xml` only if you are sure that your distribution of the standard library is not
    vulnerable to XML vulnerabilities.

    Please review the following resources for more information:

    * https://docs.python.org/3/library/xml.html#xml-vulnerabilities
    * https://github.com/tiran/defusedxml

    The standard library relies on [`libexpat`](https://github.com/libexpat/libexpat)
    for parsing XML.
    

    中文翻译:
    用于 XML 解析的解析器。可以是“defusedxml”或“xml”。
    * `'defusedxml'` 是默认解析器，用于防止 XML 漏洞
        存在于 Python 标准库 xml 的某些发行版中。
        `defusedxml` 是标准库解析器的包装器，
        使用安全默认值设置解析器。
    * `'xml'` 是标准库解析器。
    仅当您确定您的标准库发行版不是时才使用“xml”
    容易受到 XML 漏洞的影响。
    请查看以下资源以获取更多信息：
    * https://docs.python.org/3/library/xml.html#xml-vulnerability
    * https://github.com/tiran/defusedxml
    标准库依赖于[`libexpat`](https://github.com/libexpat/libexpat)
    用于解析 XML。"""

    def get_format_instructions(self) -> str:
        """Return the format instructions for the XML output.

        中文翻译:
        返回 XML 输出的格式指令。"""
        return XML_FORMAT_INSTRUCTIONS.format(tags=self.tags)

    def parse(self, text: str) -> dict[str, str | list[Any]]:
        """Parse the output of an LLM call.

        Args:
            text: The output of an LLM call.

        Returns:
            A `dict` representing the parsed XML.

        Raises:
            OutputParserException: If the XML is not well-formed.
            ImportError: If defus`edxml is not installed and the `defusedxml` parser is
                requested.
        

        中文翻译:
        解析 LLM 调用的输出。
        参数：
            文本：LLM 调用的输出。
        返回：
            代表解析后的 XML 的“dict”。
        加薪：
            OutputParserException：如果 XML 格式不正确。
            ImportError: 如果未安装 defus`edxml 并且 `defusedxml` 解析器已安装
                要求。"""
        # Try to find XML string within triple backticks
        # 中文: 尝试在三个反引号内查找 XML 字符串
        # Imports are temporarily placed here to avoid issue with caching on CI
        # 中文: 导入暂时放置在此处以避免 CI 上的缓存出现问题
        # likely if you're reading this you can move them to the top of the file
        # 中文: 如果您正在阅读本文，您可能可以将它们移至文件顶部
        if self.parser == "defusedxml":
            if not _HAS_DEFUSEDXML:
                msg = (
                    "defusedxml is not installed. "
                    "Please install it to use the defusedxml parser."
                    "You can install it with `pip install defusedxml`"
                    "See https://github.com/tiran/defusedxml for more details"
                )
                raise ImportError(msg)
            et = ElementTree  # Use the defusedxml parser
        else:
            et = ET  # Use the standard library parser

        match = re.search(r"```(xml)?(.*)```", text, re.DOTALL)
        if match is not None:
            # If match found, use the content within the backticks
            # 中文: 如果找到匹配，则使用反引号内的内容
            text = match.group(2)
        encoding_match = self.encoding_matcher.search(text)
        if encoding_match:
            text = encoding_match.group(2)

        text = text.strip()
        try:
            root = et.fromstring(text)
            return self._root_to_dict(root)
        except et.ParseError as e:
            msg = f"Failed to parse XML format from completion {text}. Got: {e}"
            raise OutputParserException(msg, llm_output=text) from e

    @override
    def _transform(self, input: Iterator[str | BaseMessage]) -> Iterator[AddableDict]:
        streaming_parser = _StreamingParser(self.parser)
        for chunk in input:
            yield from streaming_parser.parse(chunk)
        streaming_parser.close()

    @override
    async def _atransform(
        self, input: AsyncIterator[str | BaseMessage]
    ) -> AsyncIterator[AddableDict]:
        streaming_parser = _StreamingParser(self.parser)
        async for chunk in input:
            for output in streaming_parser.parse(chunk):
                yield output
        streaming_parser.close()

    def _root_to_dict(self, root: ET.Element) -> dict[str, str | list[Any]]:
        """Converts xml tree to python dictionary.

        中文翻译:
        将 xml 树转换为 python 字典。"""
        if root.text and bool(re.search(r"\S", root.text)):
            # If root text contains any non-whitespace character it
            # 中文: 如果根文本包含任何非空白字符
            # returns {root.tag: root.text}
            # 中文: 返回 {root.tag: root.text}
            return {root.tag: root.text}
        result: dict = {root.tag: []}
        for child in root:
            if len(child) == 0:
                result[root.tag].append({child.tag: child.text})
            else:
                result[root.tag].append(self._root_to_dict(child))
        return result

    @property
    def _type(self) -> str:
        return "xml"


def nested_element(path: list[str], elem: ET.Element) -> Any:
    """Get nested element from path.

    Args:
        path: The path to the element.
        elem: The element to extract.

    Returns:
        The nested element.
    

    中文翻译:
    从路径中获取嵌套元素。
    参数：
        路径：元素的路径。
        elem：要提取的元素。
    返回：
        嵌套元素。"""
    if len(path) == 0:
        return AddableDict({elem.tag: elem.text})
    return AddableDict({path[0]: [nested_element(path[1:], elem)]})
