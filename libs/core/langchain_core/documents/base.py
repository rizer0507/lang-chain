"""Base classes for media and documents.

This module contains core abstractions for **data retrieval and processing workflows**:

- `BaseMedia`: Base class providing `id` and `metadata` fields
- `Blob`: Raw data loading (files, binary data) - used by document loaders
- `Document`: Text content for retrieval (RAG, vector stores, semantic search)

!!! note "Not for LLM chat messages"
    These classes are for data processing pipelines, not LLM I/O. For multimodal
    content in chat messages (images, audio in conversations), see
    `langchain.messages` content blocks instead.

中文翻译:
媒体和文档的基类。
该模块包含**数据检索和处理工作流程**的核心抽象：
- `BaseMedia`：提供`id`和`metadata`字段的基类
- `Blob`：原始数据加载（文件、二进制数据）- 由文档加载器使用
- `文档`：用于检索的文本内容（RAG、向量存储、语义搜索）
!!!注意“不适用于 LLM 聊天消息”
    这些类用于数据处理管道，而不是 LLM I/O。对于多式联运
    聊天消息中的内容（对话中的图像、音频），请参阅
    相反，“langchain.messages”内容块。
"""

from __future__ import annotations

import contextlib
import mimetypes
from io import BufferedReader, BytesIO
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import ConfigDict, Field, model_validator

from langchain_core.load.serializable import Serializable

if TYPE_CHECKING:
    from collections.abc import Generator

PathLike = str | PurePath


class BaseMedia(Serializable):
    """Base class for content used in retrieval and data processing workflows.

    Provides common fields for content that needs to be stored, indexed, or searched.

    !!! note
        For multimodal content in **chat messages** (images, audio sent to/from LLMs),
        use `langchain.messages` content blocks instead.
    

    中文翻译:
    检索和数据处理工作流程中使用的内容的基类。
    为需要存储、索引或搜索的内容提供公共字段。
    !!!注释
        对于**聊天消息**中的多模式内容（发送到/从法学硕士发送的图像、音频），
        请改用“langchain.messages”内容块。"""

    # The ID field is optional at the moment.
    # 中文: ID 字段目前是可选的。
    # It will likely become required in a future major release after
    # 中文: 在之后的未来主要版本中可能会需要它
    # it has been adopted by enough VectorStore implementations.
    # 中文: 它已被足够多的 VectorStore 实现所采用。
    id: str | None = Field(default=None, coerce_numbers_to_str=True)
    """An optional identifier for the document.

    Ideally this should be unique across the document collection and formatted
    as a UUID, but this will not be enforced.
    

    中文翻译:
    文档的可选标识符。
    理想情况下，这应该在整个文档集合中是唯一的并且格式化
    作为 UUID，但这不会被强制执行。"""

    metadata: dict = Field(default_factory=dict)
    """Arbitrary metadata associated with the content.

    中文翻译:
    与内容关联的任意元数据。"""


class Blob(BaseMedia):
    """Raw data abstraction for document loading and file processing.

    Represents raw bytes or text, either in-memory or by file reference. Used
    primarily by document loaders to decouple data loading from parsing.

    Inspired by [Mozilla's `Blob`](https://developer.mozilla.org/en-US/docs/Web/API/Blob)

    ???+ example "Initialize a blob from in-memory data"

        ```python
        from langchain_core.documents import Blob

        blob = Blob.from_data("Hello, world!")

        # Read the blob as a string
        # 中文: 将 blob 作为字符串读取
        print(blob.as_string())

        # Read the blob as bytes
        # 中文: 将 blob 读取为字节
        print(blob.as_bytes())

        # Read the blob as a byte stream
        # 中文: 将 blob 作为字节流读取
        with blob.as_bytes_io() as f:
            print(f.read())
        ```

    ??? example "Load from memory and specify MIME type and metadata"

        ```python
        from langchain_core.documents import Blob

        blob = Blob.from_data(
            data="Hello, world!",
            mime_type="text/plain",
            metadata={"source": "https://example.com"},
        )
        ```

    ??? example "Load the blob from a file"

        ```python
        from langchain_core.documents import Blob

        blob = Blob.from_path("path/to/file.txt")

        # Read the blob as a string
        # 中文: 将 blob 作为字符串读取
        print(blob.as_string())

        # Read the blob as bytes
        # 中文: 将 blob 读取为字节
        print(blob.as_bytes())

        # Read the blob as a byte stream
        # 中文: 将 blob 作为字节流读取
        with blob.as_bytes_io() as f:
            print(f.read())
        ```
    

    中文翻译:
    用于文档加载和文件处理的原始数据抽象。
    表示内存中或文件引用的原始字节或文本。二手
    主要通过文档加载器将数据加载与解析分离。
    受到 [Mozilla 的 `Blob`](https://developer.mozilla.org/en-US/docs/Web/API/Blob) 的启发
    ???+ 示例“从内存数据初始化 blob”
        ````蟒蛇
        从 langchain_core.documents 导入 Blob
        blob = Blob.from_data("你好，世界！")
        # 将 blob 作为字符串读取
        打印（blob.as_string（））
        # 将 blob 读取为字节
        打印（blob.as_bytes（））
        # 将 blob 作为字节流读取
        将 blob.as_bytes_io() 用作 f：
            打印(f.read())
        ````
    ???示例“从内存加载并指定 MIME 类型和元数据”
        ````蟒蛇
        从 langchain_core.documents 导入 Blob
        blob = Blob.from_data(
            数据=“你好，世界！”，
            mime_type =“文本/纯文本”，
            元数据={"source": "https://example.com"},
        ）
        ````
    ???示例“从文件加载 blob”
        ````蟒蛇
        从 langchain_core.documents 导入 Blob
        blob = Blob.from_path("path/to/file.txt")
        # 将 blob 作为字符串读取
        打印（blob.as_string（））
        # 将 blob 读取为字节
        打印（blob.as_bytes（））
        # 将 blob 作为字节流读取
        将 blob.as_bytes_io() 用作 f：
            打印(f.read())
        ````"""

    data: bytes | str | None = None
    """Raw data associated with the `Blob`.

    中文翻译:
    与“Blob”关联的原始数据。"""
    mimetype: str | None = None
    """MIME type, not to be confused with a file extension.

    中文翻译:
    MIME 类型，不要与文件扩展名混淆。"""
    encoding: str = "utf-8"
    """Encoding to use if decoding the bytes into a string.

    Uses `utf-8` as default encoding if decoding to string.
    

    中文翻译:
    将字节解码为字符串时使用的编码。
    如果解码为字符串，则使用“utf-8”作为默认编码。"""
    path: PathLike | None = None
    """Location where the original content was found.

    中文翻译:
    找到原始内容的位置。"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    @property
    def source(self) -> str | None:
        """The source location of the blob as string if known otherwise none.

        If a path is associated with the `Blob`, it will default to the path location.

        Unless explicitly set via a metadata field called `'source'`, in which
        case that value will be used instead.
        

        中文翻译:
        如果已知，则以字符串形式表示 blob 的源位置，否则无。
        如果路径与“Blob”关联，则它将默认为该路径位置。
        除非通过名为“source”的元数据字段显式设置，其中
        情况下将使用该值。"""
        if self.metadata and "source" in self.metadata:
            return cast("str | None", self.metadata["source"])
        return str(self.path) if self.path else None

    @model_validator(mode="before")
    @classmethod
    def check_blob_is_valid(cls, values: dict[str, Any]) -> Any:
        """Verify that either data or path is provided.

        中文翻译:
        验证是否提供了数据或路径。"""
        if "data" not in values and "path" not in values:
            msg = "Either data or path must be provided"
            raise ValueError(msg)
        return values

    def as_string(self) -> str:
        """Read data as a string.

        Raises:
            ValueError: If the blob cannot be represented as a string.

        Returns:
            The data as a string.
        

        中文翻译:
        以字符串形式读取数据。
        加薪：
            ValueError：如果 blob 无法表示为字符串。
        返回：
            数据作为字符串。"""
        if self.data is None and self.path:
            return Path(self.path).read_text(encoding=self.encoding)
        if isinstance(self.data, bytes):
            return self.data.decode(self.encoding)
        if isinstance(self.data, str):
            return self.data
        msg = f"Unable to get string for blob {self}"
        raise ValueError(msg)

    def as_bytes(self) -> bytes:
        """Read data as bytes.

        Raises:
            ValueError: If the blob cannot be represented as bytes.

        Returns:
            The data as bytes.
        

        中文翻译:
        以字节形式读取数据。
        加薪：
            ValueError：如果 blob 无法表示为字节。
        返回：
            数据以字节为单位。"""
        if isinstance(self.data, bytes):
            return self.data
        if isinstance(self.data, str):
            return self.data.encode(self.encoding)
        if self.data is None and self.path:
            return Path(self.path).read_bytes()
        msg = f"Unable to get bytes for blob {self}"
        raise ValueError(msg)

    @contextlib.contextmanager
    def as_bytes_io(self) -> Generator[BytesIO | BufferedReader, None, None]:
        """Read data as a byte stream.

        Raises:
            NotImplementedError: If the blob cannot be represented as a byte stream.

        Yields:
            The data as a byte stream.
        

        中文翻译:
        以字节流形式读取数据。
        加薪：
            NotImplementedError：如果 blob 无法表示为字节流。
        产量：
            数据作为字节流。"""
        if isinstance(self.data, bytes):
            yield BytesIO(self.data)
        elif self.data is None and self.path:
            with Path(self.path).open("rb") as f:
                yield f
        else:
            msg = f"Unable to convert blob {self}"
            raise NotImplementedError(msg)

    @classmethod
    def from_path(
        cls,
        path: PathLike,
        *,
        encoding: str = "utf-8",
        mime_type: str | None = None,
        guess_type: bool = True,
        metadata: dict | None = None,
    ) -> Blob:
        """Load the blob from a path like object.

        Args:
            path: Path-like object to file to be read
            encoding: Encoding to use if decoding the bytes into a string
            mime_type: If provided, will be set as the MIME type of the data
            guess_type: If `True`, the MIME type will be guessed from the file
                extension, if a MIME type was not provided
            metadata: Metadata to associate with the `Blob`

        Returns:
            `Blob` instance
        

        中文翻译:
        从类似对象的路径加载 blob。
        参数：
            path：要读取的文件的类似路径对象
            编码：将字节解码为字符串时使用的编码
            mime_type：如果提供，将被设置为数据的 MIME 类型
            guess_type：如果为“True”，则将从文件中猜测 MIME 类型
                扩展名（如果未提供 MIME 类型）
            元数据：与“Blob”关联的元数据
        返回：
            `Blob` 实例"""
        if mime_type is None and guess_type:
            mimetype = mimetypes.guess_type(path)[0] if guess_type else None
        else:
            mimetype = mime_type
        # We do not load the data immediately, instead we treat the blob as a
        # 中文: 我们不会立即加载数据，而是将 blob 视为
        # reference to the underlying data.
        # 中文: 参考基础数据。
        return cls(
            data=None,
            mimetype=mimetype,
            encoding=encoding,
            path=path,
            metadata=metadata if metadata is not None else {},
        )

    @classmethod
    def from_data(
        cls,
        data: str | bytes,
        *,
        encoding: str = "utf-8",
        mime_type: str | None = None,
        path: str | None = None,
        metadata: dict | None = None,
    ) -> Blob:
        """Initialize the `Blob` from in-memory data.

        Args:
            data: The in-memory data associated with the `Blob`
            encoding: Encoding to use if decoding the bytes into a string
            mime_type: If provided, will be set as the MIME type of the data
            path: If provided, will be set as the source from which the data came
            metadata: Metadata to associate with the `Blob`

        Returns:
            `Blob` instance
        

        中文翻译:
        从内存数据初始化“Blob”。
        参数：
            data：与“Blob”关联的内存数据
            编码：将字节解码为字符串时使用的编码
            mime_type：如果提供，将被设置为数据的 MIME 类型
            路径：如果提供，将被设置为数据的来源
            元数据：与“Blob”关联的元数据
        返回：
            `Blob` 实例"""
        return cls(
            data=data,
            mimetype=mime_type,
            encoding=encoding,
            path=path,
            metadata=metadata if metadata is not None else {},
        )

    def __repr__(self) -> str:
        """Return the blob representation.

        中文翻译:
        返回 blob 表示形式。"""
        str_repr = f"Blob {id(self)}"
        if self.source:
            str_repr += f" {self.source}"
        return str_repr


class Document(BaseMedia):
    """Class for storing a piece of text and associated metadata.

    !!! note
        `Document` is for **retrieval workflows**, not chat I/O. For sending text
        to an LLM in a conversation, use message types from `langchain.messages`.

    Example:
        ```python
        from langchain_core.documents import Document

        document = Document(
            page_content="Hello, world!", metadata={"source": "https://example.com"}
        )
        ```
    

    中文翻译:
    用于存储一段文本和相关元数据的类。
    ！！！注释
        “文档”用于**检索工作流程**，而不是聊天 I/O。用于发送文本
        对于对话中的 LLM，请使用“langchain.messages”中的消息类型。
    示例：
        ````蟒蛇
        从 langchain_core.documents 导入文档
        文档 = 文档（
            page_content =“你好，世界！”，元数据= {“source”：“https://example.com”}
        ）
        ````"""

    page_content: str
    """String text.

    中文翻译:
    字符串文本。"""
    type: Literal["Document"] = "Document"

    def __init__(self, page_content: str, **kwargs: Any) -> None:
        """Pass page_content in as positional or named arg.

        中文翻译:
        将 page_content 作为位置参数或命名参数传递。"""
        # my-py is complaining that page_content is not defined on the base class.
        # 中文: my-py 抱怨 page_content 没有在基类上定义。
        # Here, we're relying on pydantic base class to handle the validation.
        # 中文: 在这里，我们依靠 pydantic 基类来处理验证。
        super().__init__(page_content=page_content, **kwargs)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return `True` as this class is serializable.

        中文翻译:
        返回“True”，因为此类是可序列化的。"""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            ["langchain", "schema", "document"]
        

        中文翻译:
        获取LangChain对象的命名空间。
        返回：
            [“langchain”、“模式”、“文档”]"""
        return ["langchain", "schema", "document"]

    def __str__(self) -> str:
        """Override `__str__` to restrict it to page_content and metadata.

        Returns:
            A string representation of the `Document`.
        

        中文翻译:
        覆盖 `__str__` 将其限制为 page_content 和元数据。
        返回：
            “文档”的字符串表示形式。"""
        # The format matches pydantic format for __str__.
        # 中文: 该格式与 __str__ 的 pydantic 格式匹配。
        #
        # The purpose of this change is to make sure that user code that
        #
        中文: # 此更改的目的是确保用户代码
        # feeds Document objects directly into prompts remains unchanged
        # 中文: 将文档对象直接提供给提示保持不变
        # due to the addition of the id field (or any other fields in the future).
        # 中文: 由于添加了 id 字段（或将来的任何其他字段）。
        #
        # This override will likely be removed in the future in favor of
        #
        中文: # 这个覆盖将来可能会被删除，以支持
        # a more general solution of formatting content directly inside the prompts.
        # 中文: 直接在提示内格式化内容的更通用的解决方案。
        if self.metadata:
            return f"page_content='{self.page_content}' metadata={self.metadata}"
        return f"page_content='{self.page_content}'"
