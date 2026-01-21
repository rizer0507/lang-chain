"""Base classes for indexing.

中文翻译:
用于索引的基类。"""

from __future__ import annotations

import abc
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypedDict

from typing_extensions import override

from langchain_core._api import beta
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import run_in_executor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.documents import Document


class RecordManager(ABC):
    """Abstract base class representing the interface for a record manager.

    The record manager abstraction is used by the langchain indexing API.

    The record manager keeps track of which documents have been
    written into a `VectorStore` and when they were written.

    The indexing API computes hashes for each document and stores the hash
    together with the write time and the source id in the record manager.

    On subsequent indexing runs, the indexing API can check the record manager
    to determine which documents have already been indexed and which have not.

    This allows the indexing API to avoid re-indexing documents that have
    already been indexed, and to only index new documents.

    The main benefit of this abstraction is that it works across many vectorstores.
    To be supported, a `VectorStore` needs to only support the ability to add and
    delete documents by ID. Using the record manager, the indexing API will
    be able to delete outdated documents and avoid redundant indexing of documents
    that have already been indexed.

    The main constraints of this abstraction are:

    1. It relies on the time-stamps to determine which documents have been
        indexed and which have not. This means that the time-stamps must be
        monotonically increasing. The timestamp should be the timestamp
        as measured by the server to minimize issues.
    2. The record manager is currently implemented separately from the
        vectorstore, which means that the overall system becomes distributed
        and may create issues with consistency. For example, writing to
        record manager succeeds, but corresponding writing to `VectorStore` fails.
    

    中文翻译:
    表示记录管理器接口的抽象基类。
    记录管理器抽象由 langchain 索引 API 使用。
    记录管理器跟踪哪些文件已被处理
    写入“VectorStore”以及写入时间。
    索引 API 计算每个文档的哈希值并存储哈希值
    以及记录管理器中的写入时间和源 ID。
    在后续索引运行中，索引 API 可以检查记录管理器
    确定哪些文档已被索引，哪些尚未索引。
    这使得索引 API 可以避免对具有以下特征的文档重新建立索引：
    已经被索引，并且只索引新文档。
    这种抽象的主要好处是它可以跨许多向量存储工作。
    要获得支持，“VectorStore”只需支持添加和
    按 ID 删除文档。使用记录管理器，索引 API 将
    能够删除过时的文档并避免文档的冗余索引
    已经被索引的。
    这种抽象的主要限制是：
    1.它依靠时间戳来确定哪些文档已被
        哪些已索引，哪些尚未索引。这意味着时间戳必须是
        单调增加。时间戳应该是时间戳
        由服务器测量以尽量减少问题。
    2. 记录管理器目前与
        vectorstore，这意味着整个系统变得分布式
        并且可能会产生一致性问题。例如，写信给
        记录管理器成功，但相应写入“VectorStore”失败。"""

    def __init__(
        self,
        namespace: str,
    ) -> None:
        """Initialize the record manager.

        Args:
            namespace: The namespace for the record manager.
        

        中文翻译:
        初始化记录管理器。
        参数：
            命名空间：记录管理器的命名空间。"""
        self.namespace = namespace

    @abstractmethod
    def create_schema(self) -> None:
        """Create the database schema for the record manager.

        中文翻译:
        为记录管理器创建数据库架构。"""

    @abstractmethod
    async def acreate_schema(self) -> None:
        """Asynchronously create the database schema for the record manager.

        中文翻译:
        为记录管理器异步创建数据库架构。"""

    @abstractmethod
    def get_time(self) -> float:
        """Get the current server time as a high resolution timestamp!

        It's important to get this from the server to ensure a monotonic clock,
        otherwise there may be data loss when cleaning up old documents!

        Returns:
            The current server time as a float timestamp.
        

        中文翻译:
        获取当前服务器时间作为高分辨率时间戳！
        从服务器获取此信息以确保时钟的单调性非常重要，
        否则清理旧文档时可能会丢失数据！
        返回：
            The current server time as a float timestamp."""

    @abstractmethod
    async def aget_time(self) -> float:
        """Asynchronously get the current server time as a high resolution timestamp.

        It's important to get this from the server to ensure a monotonic clock,
        otherwise there may be data loss when cleaning up old documents!

        Returns:
            The current server time as a float timestamp.
        

        中文翻译:
        异步获取当前服务器时间作为高分辨率时间戳。
        从服务器获取此信息以确保时钟的单调性非常重要，
        否则清理旧文档时可能会丢失数据！
        返回：
            当前服务器时间作为浮点时间戳。"""

    @abstractmethod
    def update(
        self,
        keys: Sequence[str],
        *,
        group_ids: Sequence[str | None] | None = None,
        time_at_least: float | None = None,
    ) -> None:
        """Upsert records into the database.

        Args:
            keys: A list of record keys to upsert.
            group_ids: A list of group IDs corresponding to the keys.
            time_at_least: Optional timestamp. Implementation can use this
                to optionally verify that the timestamp IS at least this time
                in the system that stores the data.

                e.g., use to validate that the time in the postgres database
                is equal to or larger than the given timestamp, if not
                raise an error.

                This is meant to help prevent time-drift issues since
                time may not be monotonically increasing!

        Raises:
            ValueError: If the length of keys doesn't match the length of group_ids.
        

        中文翻译:
        将记录更新到数据库中。
        参数：
            键：要更新插入的记录键列表。
            group_ids：键对应的组ID列表。
            time_at_least：可选时间戳。实现可以用这个
                可选择验证时间戳至少在这一次
                在存储数据的系统中。
                例如，用于验证 postgres 数据库中的时间
                等于或大于给定时间戳，如果不是
                提出错误。
                这是为了帮助防止时间漂移问题，因为
                时间可能不是单调增加的！
        加薪：
            ValueError：如果键的长度与group_ids的长度不匹配。"""

    @abstractmethod
    async def aupdate(
        self,
        keys: Sequence[str],
        *,
        group_ids: Sequence[str | None] | None = None,
        time_at_least: float | None = None,
    ) -> None:
        """Asynchronously upsert records into the database.

        Args:
            keys: A list of record keys to upsert.
            group_ids: A list of group IDs corresponding to the keys.
            time_at_least: Optional timestamp. Implementation can use this
                to optionally verify that the timestamp IS at least this time
                in the system that stores the data.

                e.g., use to validate that the time in the postgres database
                is equal to or larger than the given timestamp, if not
                raise an error.

                This is meant to help prevent time-drift issues since
                time may not be monotonically increasing!

        Raises:
            ValueError: If the length of keys doesn't match the length of group_ids.
        

        中文翻译:
        将记录异步更新到数据库中。
        参数：
            键：要更新插入的记录键列表。
            group_ids：键对应的组ID列表。
            time_at_least：可选时间戳。实现可以用这个
                可选择验证时间戳至少在这一次
                在存储数据的系统中。
                例如，用于验证 postgres 数据库中的时间
                等于或大于给定时间戳，如果不是
                提出错误。
                这是为了帮助防止时间漂移问题，因为
                时间可能不是单调增加的！
        加薪：
            ValueError：如果键的长度与group_ids的长度不匹配。"""

    @abstractmethod
    def exists(self, keys: Sequence[str]) -> list[bool]:
        """Check if the provided keys exist in the database.

        Args:
            keys: A list of keys to check.

        Returns:
            A list of boolean values indicating the existence of each key.
        

        中文翻译:
        检查数据库中是否存在提供的密钥。
        参数：
            键：要检查的键列表。
        返回：
            指示每个键是否存在的布尔值列表。"""

    @abstractmethod
    async def aexists(self, keys: Sequence[str]) -> list[bool]:
        """Asynchronously check if the provided keys exist in the database.

        Args:
            keys: A list of keys to check.

        Returns:
            A list of boolean values indicating the existence of each key.
        

        中文翻译:
        异步检查数据库中是否存在提供的密钥。
        参数：
            键：要检查的键列表。
        返回：
            指示每个键是否存在的布尔值列表。"""

    @abstractmethod
    def list_keys(
        self,
        *,
        before: float | None = None,
        after: float | None = None,
        group_ids: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """List records in the database based on the provided filters.

        Args:
            before: Filter to list records updated before this time.
            after: Filter to list records updated after this time.
            group_ids: Filter to list records with specific group IDs.
            limit: optional limit on the number of records to return.

        Returns:
            A list of keys for the matching records.
        

        中文翻译:
        根据提供的过滤器列出数据库中的记录。
        参数：
            before：过滤列出在此时间之前更新的记录。
            after：过滤以列出该时间之后更新的记录。
            group_ids：过滤以列出具有特定组 ID 的记录。
            limit：返回记录数量的可选限制。
        返回：
            匹配记录的键列表。"""

    @abstractmethod
    async def alist_keys(
        self,
        *,
        before: float | None = None,
        after: float | None = None,
        group_ids: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """Asynchronously list records in the database based on the provided filters.

        Args:
            before: Filter to list records updated before this time.
            after: Filter to list records updated after this time.
            group_ids: Filter to list records with specific group IDs.
            limit: optional limit on the number of records to return.

        Returns:
            A list of keys for the matching records.
        

        中文翻译:
        根据提供的过滤器异步列出数据库中的记录。
        参数：
            before：过滤列出在此时间之前更新的记录。
            after：过滤以列出该时间之后更新的记录。
            group_ids：过滤以列出具有特定组 ID 的记录。
            limit：返回记录数量的可选限制。
        返回：
            匹配记录的键列表。"""

    @abstractmethod
    def delete_keys(self, keys: Sequence[str]) -> None:
        """Delete specified records from the database.

        Args:
            keys: A list of keys to delete.
        

        中文翻译:
        从数据库中删除指定记录。
        参数：
            键：要删除的键列表。"""

    @abstractmethod
    async def adelete_keys(self, keys: Sequence[str]) -> None:
        """Asynchronously delete specified records from the database.

        Args:
            keys: A list of keys to delete.
        

        中文翻译:
        从数据库中异步删除指定记录。
        参数：
            键：要删除的键列表。"""


class _Record(TypedDict):
    group_id: str | None
    updated_at: float


class InMemoryRecordManager(RecordManager):
    """An in-memory record manager for testing purposes.

    中文翻译:
    用于测试目的的内存记录管理器。"""

    def __init__(self, namespace: str) -> None:
        """Initialize the in-memory record manager.

        Args:
            namespace: The namespace for the record manager.
        

        中文翻译:
        初始化内存中记录管理器。
        参数：
            命名空间：记录管理器的命名空间。"""
        super().__init__(namespace)
        # Each key points to a dictionary
        # 中文: 每个键都指向一个字典
        # of {'group_id': group_id, 'updated_at': timestamp}
        # 中文: {'group_id': group_id, 'updated_at': 时间戳}
        self.records: dict[str, _Record] = {}
        self.namespace = namespace

    def create_schema(self) -> None:
        """In-memory schema creation is simply ensuring the structure is initialized.

        中文翻译:
        内存模式创建只是确保结构已初始化。"""

    async def acreate_schema(self) -> None:
        """In-memory schema creation is simply ensuring the structure is initialized.

        中文翻译:
        内存模式创建只是确保结构已初始化。"""

    @override
    def get_time(self) -> float:
        return time.time()

    @override
    async def aget_time(self) -> float:
        return self.get_time()

    def update(
        self,
        keys: Sequence[str],
        *,
        group_ids: Sequence[str | None] | None = None,
        time_at_least: float | None = None,
    ) -> None:
        """Upsert records into the database.

        Args:
            keys: A list of record keys to upsert.
            group_ids: A list of group IDs corresponding to the keys.

            time_at_least: Optional timestamp. Implementation can use this
                to optionally verify that the timestamp IS at least this time
                in the system that stores.
                E.g., use to validate that the time in the postgres database
                is equal to or larger than the given timestamp, if not
                raise an error.
                This is meant to help prevent time-drift issues since
                time may not be monotonically increasing!

        Raises:
            ValueError: If the length of keys doesn't match the length of group
                ids.
            ValueError: If time_at_least is in the future.
        

        中文翻译:
        将记录更新到数据库中。
        参数：
            键：要更新插入的记录键列表。
            group_ids：键对应的组ID列表。
            time_at_least：可选时间戳。实现可以用这个
                可选择验证时间戳至少在这一次
                在存储的系统中。
                例如，用于验证 postgres 数据库中的时间
                等于或大于给定时间戳，如果不是
                提出错误。
                这是为了帮助防止时间漂移问题，因为
                时间可能不是单调增加的！
        加薪：
            ValueError：如果键的长度与组的长度不匹配
                id。
            ValueError：如果 time_at_least 是将来的时间。"""
        if group_ids and len(keys) != len(group_ids):
            msg = "Length of keys must match length of group_ids"
            raise ValueError(msg)
        for index, key in enumerate(keys):
            group_id = group_ids[index] if group_ids else None
            if time_at_least and time_at_least > self.get_time():
                msg = "time_at_least must be in the past"
                raise ValueError(msg)
            self.records[key] = {"group_id": group_id, "updated_at": self.get_time()}

    async def aupdate(
        self,
        keys: Sequence[str],
        *,
        group_ids: Sequence[str | None] | None = None,
        time_at_least: float | None = None,
    ) -> None:
        """Async upsert records into the database.

        Args:
            keys: A list of record keys to upsert.
            group_ids: A list of group IDs corresponding to the keys.

            time_at_least: Optional timestamp. Implementation can use this
                to optionally verify that the timestamp IS at least this time
                in the system that stores.
                E.g., use to validate that the time in the postgres database
                is equal to or larger than the given timestamp, if not
                raise an error.
                This is meant to help prevent time-drift issues since
                time may not be monotonically increasing!
        

        中文翻译:
        异步将记录更新插入数据库。
        参数：
            键：要更新插入的记录键列表。
            group_ids：键对应的组ID列表。
            time_at_least：可选时间戳。实现可以用这个
                可选择验证时间戳至少在这一次
                在存储的系统中。
                例如，用于验证 postgres 数据库中的时间
                等于或大于给定时间戳，如果不是
                提出错误。
                这是为了帮助防止时间漂移问题，因为
                时间可能不是单调增加的！"""
        self.update(keys, group_ids=group_ids, time_at_least=time_at_least)

    def exists(self, keys: Sequence[str]) -> list[bool]:
        """Check if the provided keys exist in the database.

        Args:
            keys: A list of keys to check.

        Returns:
            A list of boolean values indicating the existence of each key.
        

        中文翻译:
        检查数据库中是否存在提供的密钥。
        参数：
            键：要检查的键列表。
        返回：
            指示每个键是否存在的布尔值列表。"""
        return [key in self.records for key in keys]

    async def aexists(self, keys: Sequence[str]) -> list[bool]:
        """Async check if the provided keys exist in the database.

        Args:
            keys: A list of keys to check.

        Returns:
            A list of boolean values indicating the existence of each key.
        

        中文翻译:
        异步检查数据库中是否存在提供的密钥。
        参数：
            键：要检查的键列表。
        返回：
            指示每个键是否存在的布尔值列表。"""
        return self.exists(keys)

    def list_keys(
        self,
        *,
        before: float | None = None,
        after: float | None = None,
        group_ids: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """List records in the database based on the provided filters.

        Args:
            before: Filter to list records updated before this time.

            after: Filter to list records updated after this time.

            group_ids: Filter to list records with specific group IDs.

            limit: optional limit on the number of records to return.


        Returns:
            A list of keys for the matching records.
        

        中文翻译:
        根据提供的过滤器列出数据库中的记录。
        参数：
            before：过滤列出在此时间之前更新的记录。
            after：过滤以列出该时间之后更新的记录。
            group_ids：过滤以列出具有特定组 ID 的记录。
            limit：返回记录数量的可选限制。
        返回：
            匹配记录的键列表。"""
        result = []
        for key, data in self.records.items():
            if before and data["updated_at"] >= before:
                continue
            if after and data["updated_at"] <= after:
                continue
            if group_ids and data["group_id"] not in group_ids:
                continue
            result.append(key)
        if limit:
            return result[:limit]
        return result

    async def alist_keys(
        self,
        *,
        before: float | None = None,
        after: float | None = None,
        group_ids: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """Async list records in the database based on the provided filters.

        Args:
            before: Filter to list records updated before this time.

            after: Filter to list records updated after this time.

            group_ids: Filter to list records with specific group IDs.

            limit: optional limit on the number of records to return.


        Returns:
            A list of keys for the matching records.
        

        中文翻译:
        根据提供的过滤器异步列出数据库中的记录。
        参数：
            before：过滤列出在此时间之前更新的记录。
            after：过滤以列出该时间之后更新的记录。
            group_ids：过滤以列出具有特定组 ID 的记录。
            limit：返回记录数量的可选限制。
        返回：
            匹配记录的键列表。"""
        return self.list_keys(
            before=before, after=after, group_ids=group_ids, limit=limit
        )

    def delete_keys(self, keys: Sequence[str]) -> None:
        """Delete specified records from the database.

        Args:
            keys: A list of keys to delete.
        

        中文翻译:
        从数据库中删除指定记录。
        参数：
            键：要删除的键列表。"""
        for key in keys:
            if key in self.records:
                del self.records[key]

    async def adelete_keys(self, keys: Sequence[str]) -> None:
        """Async delete specified records from the database.

        Args:
            keys: A list of keys to delete.
        

        中文翻译:
        异步从数据库中删除指定记录。
        参数：
            键：要删除的键列表。"""
        self.delete_keys(keys)


class UpsertResponse(TypedDict):
    """A generic response for upsert operations.

    The upsert response will be used by abstractions that implement an upsert
    operation for content that can be upserted by ID.

    Upsert APIs that accept inputs with IDs and generate IDs internally
    will return a response that includes the IDs that succeeded and the IDs
    that failed.

    If there are no failures, the failed list will be empty, and the order
    of the IDs in the succeeded list will match the order of the input documents.

    If there are failures, the response becomes ill defined, and a user of the API
    cannot determine which generated ID corresponds to which input document.

    It is recommended for users explicitly attach the IDs to the items being
    indexed to avoid this issue.
    

    中文翻译:
    upsert 操作的通用响应。
    upsert 响应将由实现 upsert 的抽象使用
    对可通过 ID 更新插入的内容进行操作。
    Upsert API 接受带 ID 的输入并在内部生成 ID
    将返回一个响应，其中包含成功的 ID 和
    那失败了。
    如果没有失败，则失败列表将为空，并且顺序
    成功列表中的 ID 的顺序将与输入文档的顺序匹配。
    如果出现故障，响应就会变得不明确，并且 API 的用户
    无法确定哪个生成的 ID 对应于哪个输入文档。
    建议用户将 ID 明确附加到正在使用的项目上
    索引以避免这个问题。"""

    succeeded: list[str]
    """The IDs that were successfully indexed.

    中文翻译:
    已成功建立索引的 ID。"""
    failed: list[str]
    """The IDs that failed to index.

    中文翻译:
    索引失败的 ID。"""


class DeleteResponse(TypedDict, total=False):
    """A generic response for delete operation.

    The fields in this response are optional and whether the `VectorStore`
    returns them or not is up to the implementation.
    

    中文翻译:
    删除操作的通用响应。
    此响应中的字段是可选的，并且是否包含 `VectorStore`
    返回与否取决于实现。"""

    num_deleted: int
    """The number of items that were successfully deleted.

    If returned, this should only include *actual* deletions.

    If the ID did not exist to begin with,
    it should not be included in this count.
    

    中文翻译:
    成功删除的项目数。
    如果返回，则应仅包括“实际”删除。
    如果该 ID 一开始就不存在，
    它不应该包含在这个计数中。"""

    succeeded: Sequence[str]
    """The IDs that were successfully deleted.

    If returned, this should only include *actual* deletions.

    If the ID did not exist to begin with,
    it should not be included in this list.
    

    中文翻译:
    已成功删除的 ID。
    如果返回，则应仅包括“实际”删除。
    如果该 ID 一开始就不存在，
    它不应该包含在此列表中。"""

    failed: Sequence[str]
    """The IDs that failed to be deleted.

    !!! warning
        Deleting an ID that does not exist is **NOT** considered a failure.
    

    中文翻译:
    删除失败的ID。
    ！！！警告
        删除不存在的 ID **不**被视为失败。"""

    num_failed: int
    """The number of items that failed to be deleted.

    中文翻译:
    删除失败的项目数。"""


@beta(message="Added in 0.2.29. The abstraction is subject to change.")
class DocumentIndex(BaseRetriever):
    """A document retriever that supports indexing operations.

    This indexing interface is designed to be a generic abstraction for storing and
    querying documents that has an ID and metadata associated with it.

    The interface is designed to be agnostic to the underlying implementation of the
    indexing system.

    The interface is designed to support the following operations:

    1. Storing document in the index.
    2. Fetching document by ID.
    3. Searching for document using a query.
    

    中文翻译:
    支持索引操作的文档检索器。
    该索引接口被设计为用于存储和索引的通用抽象
    查询具有与其关联的 ID 和元数据的文档。
    该接口被设计为与底层实现无关
    索引系统。
    该接口旨在支持以下操作：
    1. 将文档存储在索引中。
    2.通过ID获取文档。
    3. 使用查询搜索文档。"""

    @abc.abstractmethod
    def upsert(self, items: Sequence[Document], /, **kwargs: Any) -> UpsertResponse:
        """Upsert documents into the index.

        The upsert functionality should utilize the ID field of the content object
        if it is provided. If the ID is not provided, the upsert method is free
        to generate an ID for the content.

        When an ID is specified and the content already exists in the `VectorStore`,
        the upsert method should update the content with the new data. If the content
        does not exist, the upsert method should add the item to the `VectorStore`.

        Args:
            items: Sequence of documents to add to the `VectorStore`.
            **kwargs: Additional keyword arguments.

        Returns:
            A response object that contains the list of IDs that were
            successfully added or updated in the `VectorStore` and the list of IDs that
            failed to be added or updated.
        

        中文翻译:
        将文档更新到索引中。
        upsert 功能应利用内容对象的 ID 字段
        如果提供的话。如果不提供ID，upsert方法是免费的
        为内容生成 ID。
        当指定了 ID 并且内容已存在于 `VectorStore` 中时，
        upsert 方法应使用新数据更新内容。如果内容
        不存在，upsert 方法应将该项目添加到 `VectorStore` 中。
        参数：
            items：要添加到“VectorStore”的文档序列。
            **kwargs：附加关键字参数。
        返回：
            包含已发送的 ID 列表的响应对象
            在 `VectorStore` 和 ID 列表中成功添加或更新
            添加或更新失败。"""

    async def aupsert(
        self, items: Sequence[Document], /, **kwargs: Any
    ) -> UpsertResponse:
        """Add or update documents in the `VectorStore`. Async version of `upsert`.

        The upsert functionality should utilize the ID field of the item
        if it is provided. If the ID is not provided, the upsert method is free
        to generate an ID for the item.

        When an ID is specified and the item already exists in the `VectorStore`,
        the upsert method should update the item with the new data. If the item
        does not exist, the upsert method should add the item to the `VectorStore`.

        Args:
            items: Sequence of documents to add to the `VectorStore`.
            **kwargs: Additional keyword arguments.

        Returns:
            A response object that contains the list of IDs that were
            successfully added or updated in the `VectorStore` and the list of IDs that
            failed to be added or updated.
        

        中文翻译:
        添加或更新“VectorStore”中的文档。 `upsert` 的异步版本。
        upsert 功能应利用项目的 ID 字段
        如果提供的话。如果不提供ID，upsert方法是免费的
        为该项目生成一个 ID。
        当指定了 ID 并且该项目已存在于 `VectorStore` 中时，
        upsert 方法应使用新数据更新项目。如果该项目
        不存在，upsert 方法应将该项目添加到 `VectorStore` 中。
        参数：
            items：要添加到“VectorStore”的文档序列。
            **kwargs：附加关键字参数。
        返回：
            包含已发送的 ID 列表的响应对象
            在 `VectorStore` 和 ID 列表中成功添加或更新
            添加或更新失败。"""
        return await run_in_executor(
            None,
            self.upsert,
            items,
            **kwargs,
        )

    @abc.abstractmethod
    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> DeleteResponse:
        """Delete by IDs or other criteria.

        Calling delete without any input parameters should raise a ValueError!

        Args:
            ids: List of IDs to delete.
            **kwargs: Additional keyword arguments. This is up to the implementation.
                For example, can include an option to delete the entire index,
                or else issue a non-blocking delete etc.

        Returns:
            A response object that contains the list of IDs that were
            successfully deleted and the list of IDs that failed to be deleted.
        

        中文翻译:
        按 ID 或其他条件删除。
        在没有任何输入参数的情况下调用删除应该引发 ValueError！
        参数：
            ids：要删除的 ID 列表。
            **kwargs：附加关键字参数。这取决于实施。
                例如，可以包含删除整个索引的选项，
                或者发出非阻塞删除等。
        返回：
            包含已发送的 ID 列表的响应对象
            删除成功的ID和删除失败的ID列表。"""

    async def adelete(
        self, ids: list[str] | None = None, **kwargs: Any
    ) -> DeleteResponse:
        """Delete by IDs or other criteria. Async variant.

        Calling adelete without any input parameters should raise a ValueError!

        Args:
            ids: List of IDs to delete.
            **kwargs: Additional keyword arguments. This is up to the implementation.
                For example, can include an option to delete the entire index.

        Returns:
            A response object that contains the list of IDs that were
            successfully deleted and the list of IDs that failed to be deleted.
        

        中文翻译:
        按 ID 或其他条件删除。异步变体。
        在没有任何输入参数的情况下调用 adelete 应该引发 ValueError！
        参数：
            ids：要删除的 ID 列表。
            **kwargs：附加关键字参数。这取决于实施。
                例如，可以包括删除整个索引的选项。
        返回：
            包含已发送的 ID 列表的响应对象
            删除成功的ID和删除失败的ID列表。"""
        return await run_in_executor(
            None,
            self.delete,
            ids,
            **kwargs,
        )

    @abc.abstractmethod
    def get(
        self,
        ids: Sequence[str],
        /,
        **kwargs: Any,
    ) -> list[Document]:
        """Get documents by id.

        Fewer documents may be returned than requested if some IDs are not found or
        if there are duplicated IDs.

        Users should not assume that the order of the returned documents matches
        the order of the input IDs. Instead, users should rely on the ID field of the
        returned documents.

        This method should **NOT** raise exceptions if no documents are found for
        some IDs.

        Args:
            ids: List of IDs to get.
            **kwargs: Additional keyword arguments. These are up to the implementation.

        Returns:
            List of documents that were found.
        

        中文翻译:
        通过id获取文档。
        如果未找到某些 ID，或者返回的文件数量可能少于要求的文件数量
        如果有重复的ID。
        用户不应假设返回文档的顺序匹配
        输入 ID 的顺序。相反，用户应该依赖于 ID 字段
        返回的文件。
        如果没有找到任何文档，此方法**不**引发异常
        一些ID。
        参数：
            ids：要获取的ID列表。
            **kwargs：附加关键字参数。这些都得看执行情况。
        返回：
            找到的文件列表。"""

    async def aget(
        self,
        ids: Sequence[str],
        /,
        **kwargs: Any,
    ) -> list[Document]:
        """Get documents by id.

        Fewer documents may be returned than requested if some IDs are not found or
        if there are duplicated IDs.

        Users should not assume that the order of the returned documents matches
        the order of the input IDs. Instead, users should rely on the ID field of the
        returned documents.

        This method should **NOT** raise exceptions if no documents are found for
        some IDs.

        Args:
            ids: List of IDs to get.
            **kwargs: Additional keyword arguments. These are up to the implementation.

        Returns:
            List of documents that were found.
        

        中文翻译:
        通过id获取文档。
        如果未找到某些 ID，或者返回的文件数量可能少于要求的文件数量
        如果有重复的ID。
        用户不应假设返回文档的顺序匹配
        输入 ID 的顺序。相反，用户应该依赖于 ID 字段
        返回的文件。
        如果没有找到任何文档，此方法**不**引发异常
        一些ID。
        参数：
            ids：要获取的ID列表。
            **kwargs：附加关键字参数。这些都得看执行情况。
        返回：
            找到的文件列表。"""
        return await run_in_executor(
            None,
            self.get,
            ids,
            **kwargs,
        )
