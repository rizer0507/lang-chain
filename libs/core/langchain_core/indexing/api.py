"""Module contains logic for indexing documents into vector stores.

中文翻译:
模块包含将文档索引到向量存储中的逻辑。"""

from __future__ import annotations

import hashlib
import json
import uuid
import warnings
from itertools import islice
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypedDict,
    TypeVar,
    cast,
)

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_core.exceptions import LangChainException
from langchain_core.indexing.base import DocumentIndex, RecordManager
from langchain_core.vectorstores import VectorStore

if TYPE_CHECKING:
    from collections.abc import (
        AsyncIterable,
        AsyncIterator,
        Callable,
        Iterable,
        Iterator,
        Sequence,
    )

# Magic UUID to use as a namespace for hashing.
# 中文: 用作哈希命名空间的 Magic UUID。
# Used to try and generate a unique UUID for each document
# 中文: 用于尝试为每个文档生成唯一的 UUID
# from hashing the document content and metadata.
# 中文: 对文档内容和元数据进行哈希处理。
NAMESPACE_UUID = uuid.UUID(int=1984)


T = TypeVar("T")


def _hash_string_to_uuid(input_string: str) -> str:
    """Hashes a string and returns the corresponding UUID.

    中文翻译:
    对字符串进行哈希处理并返回相应的 UUID。"""
    hash_value = hashlib.sha1(
        input_string.encode("utf-8"), usedforsecurity=False
    ).hexdigest()
    return str(uuid.uuid5(NAMESPACE_UUID, hash_value))


_WARNED_ABOUT_SHA1: bool = False


def _warn_about_sha1() -> None:
    """Emit a one-time warning about SHA-1 collision weaknesses.

    中文翻译:
    发出有关 SHA-1 冲突弱点的一次性警告。"""
    # Global variable OK in this case
    # 中文: 在这种情况下全局变量OK
    global _WARNED_ABOUT_SHA1  # noqa: PLW0603
    if not _WARNED_ABOUT_SHA1:
        warnings.warn(
            "Using SHA-1 for document hashing. SHA-1 is *not* "
            "collision-resistant; a motivated attacker can construct distinct inputs "
            "that map to the same fingerprint. If this matters in your "
            "threat model, switch to a stronger algorithm such "
            "as 'blake2b', 'sha256', or 'sha512' by specifying "
            " `key_encoder` parameter in the `index` or `aindex` function. ",
            category=UserWarning,
            stacklevel=2,
        )
        _WARNED_ABOUT_SHA1 = True


def _hash_string(
    input_string: str, *, algorithm: Literal["sha1", "sha256", "sha512", "blake2b"]
) -> uuid.UUID:
    """Hash *input_string* to a deterministic UUID using the configured algorithm.

    中文翻译:
    使用配置的算法将 *input_string* 哈希为确定性 UUID。"""
    if algorithm == "sha1":
        _warn_about_sha1()
    hash_value = _calculate_hash(input_string, algorithm)
    return uuid.uuid5(NAMESPACE_UUID, hash_value)


def _hash_nested_dict(
    data: dict[Any, Any], *, algorithm: Literal["sha1", "sha256", "sha512", "blake2b"]
) -> uuid.UUID:
    """Hash a nested dictionary to a UUID using the configured algorithm.

    中文翻译:
    使用配置的算法将嵌套字典哈希为 UUID。"""
    serialized_data = json.dumps(data, sort_keys=True)
    return _hash_string(serialized_data, algorithm=algorithm)


def _batch(size: int, iterable: Iterable[T]) -> Iterator[list[T]]:
    """Utility batching function.

    中文翻译:
    实用的批处理功能。"""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            return
        yield chunk


async def _abatch(size: int, iterable: AsyncIterable[T]) -> AsyncIterator[list[T]]:
    """Utility batching function.

    中文翻译:
    实用的批处理功能。"""
    batch: list[T] = []
    async for element in iterable:
        if len(batch) < size:
            batch.append(element)

        if len(batch) >= size:
            yield batch
            batch = []

    if batch:
        yield batch


def _get_source_id_assigner(
    source_id_key: str | Callable[[Document], str] | None,
) -> Callable[[Document], str | None]:
    """Get the source id from the document.

    中文翻译:
    从文档中获取源 ID。"""
    if source_id_key is None:
        return lambda _doc: None
    if isinstance(source_id_key, str):
        return lambda doc: doc.metadata[source_id_key]
    if callable(source_id_key):
        return source_id_key
    msg = (
        f"source_id_key should be either None, a string or a callable. "
        f"Got {source_id_key} of type {type(source_id_key)}."
    )
    raise ValueError(msg)


def _deduplicate_in_order(
    hashed_documents: Iterable[Document],
) -> Iterator[Document]:
    """Deduplicate a list of hashed documents while preserving order.

    中文翻译:
    对散列文档列表进行重复数据删除，同时保留顺序。"""
    seen: set[str] = set()

    for hashed_doc in hashed_documents:
        if hashed_doc.id not in seen:
            # At this stage, the id is guaranteed to be a string.
            # 中文: 在这个阶段，id 保证是一个字符串。
            # Avoiding unnecessary run time checks.
            # 中文: 避免不必要的运行时检查。
            seen.add(cast("str", hashed_doc.id))
            yield hashed_doc


class IndexingException(LangChainException):
    """Raised when an indexing operation fails.

    中文翻译:
    当索引操作失败时引发。"""


def _calculate_hash(
    text: str, algorithm: Literal["sha1", "sha256", "sha512", "blake2b"]
) -> str:
    """Return a hexadecimal digest of *text* using *algorithm*.

    中文翻译:
    使用*算法*返回*文本*的十六进制摘要。"""
    if algorithm == "sha1":
        # Calculate the SHA-1 hash and return it as a UUID.
        # 中文: 计算 SHA-1 哈希并将其作为 UUID 返回。
        digest = hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()
        return str(uuid.uuid5(NAMESPACE_UUID, digest))
    if algorithm == "blake2b":
        return hashlib.blake2b(text.encode("utf-8")).hexdigest()
    if algorithm == "sha256":
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    if algorithm == "sha512":
        return hashlib.sha512(text.encode("utf-8")).hexdigest()
    msg = f"Unsupported hashing algorithm: {algorithm}"
    raise ValueError(msg)


def _get_document_with_hash(
    document: Document,
    *,
    key_encoder: Callable[[Document], str]
    | Literal["sha1", "sha256", "sha512", "blake2b"],
) -> Document:
    """Calculate a hash of the document, and assign it to the uid.

    When using one of the predefined hashing algorithms, the hash is calculated
    by hashing the content and the metadata of the document.

    Args:
        document: Document to hash.
        key_encoder: Hashing algorithm to use for hashing the document.
            If not provided, a default encoder using SHA-1 will be used.
            SHA-1 is not collision-resistant, and a motivated attacker
            could craft two different texts that hash to the
            same cache key.

            New applications should use one of the alternative encoders
            or provide a custom and strong key encoder function to avoid this risk.

            When changing the key encoder, you must change the
            index as well to avoid duplicated documents in the cache.

    Raises:
        ValueError: If the metadata cannot be serialized using json.

    Returns:
        Document with a unique identifier based on the hash of the content and metadata.
    

    中文翻译:
    计算文档的哈希值，并将其分配给 uid。
    当使用预定义的哈希算法之一时，计算哈希值
    通过散列文档的内容和元数据。
    参数：
        document：要散列的文档。
        key_encoder：用于对文档进行哈希处理的哈希算法。
            如果未提供，将使用使用 SHA-1 的默认编码器。
            SHA-1 不具有抗碰撞性，且攻击者有动机
            可以制作两个不同的文本，散列到
            相同的缓存键。
            新应用程序应使用替代编码器之一
            或者提供自定义且强大的按键编码器功能来避免这种风险。
            更改按键编码器时，必须更改
            索引也可以避免缓存中出现重复的文档。
    加薪：
        ValueError: 如果元数据无法使用 json 序列化。
    返回：
        具有基于内容和元数据的哈希值的唯一标识符的文档。"""
    metadata: dict[str, Any] = dict(document.metadata or {})

    if callable(key_encoder):
        # If key_encoder is a callable, we use it to generate the hash.
        # 中文: 如果 key_encoder 是可调用的，我们用它来生成哈希。
        hash_ = key_encoder(document)
    else:
        # The hashes are calculated separate for the content and the metadata.
        # 中文: 内容和元数据的哈希值是分开计算的。
        content_hash = _calculate_hash(document.page_content, algorithm=key_encoder)
        try:
            serialized_meta = json.dumps(metadata, sort_keys=True)
        except Exception as e:
            msg = (
                f"Failed to hash metadata: {e}. "
                f"Please use a dict that can be serialized using json."
            )
            raise ValueError(msg) from e
        metadata_hash = _calculate_hash(serialized_meta, algorithm=key_encoder)
        hash_ = _calculate_hash(content_hash + metadata_hash, algorithm=key_encoder)

    return Document(
        # Assign a unique identifier based on the hash.
        # 中文: 根据哈希值分配唯一标识符。
        id=hash_,
        page_content=document.page_content,
        metadata=document.metadata,
    )


# This internal abstraction was imported by the langchain package internally, so
# 中文: 这个内部抽象是由 langchain 包内部导入的，所以
# we keep it here for backwards compatibility.
# 中文: 我们将其保留在这里是为了向后兼容。
class _HashedDocument:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Raise an error if this class is instantiated.

        中文翻译:
        如果实例化此类，则会引发错误。"""
        msg = (
            "_HashedDocument is an internal abstraction that was deprecated in "
            " langchain-core 0.3.63. This abstraction is marked as private and "
            " should not have been used directly. If you are seeing this error, please "
            " update your code appropriately."
        )
        raise NotImplementedError(msg)


def _delete(
    vector_store: VectorStore | DocumentIndex,
    ids: list[str],
) -> None:
    """Delete documents from a vector store or document index by their IDs.

    Args:
        vector_store: The vector store or document index to delete from.
        ids: List of document IDs to delete.

    Raises:
        IndexingException: If the delete operation fails.
        TypeError: If the `vector_store` is neither a `VectorStore` nor a
            `DocumentIndex`.
    

    中文翻译:
    按 ID 从向量存储或文档索引中删除文档。
    参数：
        vector_store：要从中删除的向量存储或文档索引。
        ids：要删除的文档 ID 列表。
    加薪：
        IndexingException：如果删除操作失败。
        类型错误：如果 `vector_store` 既不是 `VectorStore` 也不是
            `文档索引`。"""
    if isinstance(vector_store, VectorStore):
        delete_ok = vector_store.delete(ids)
        if delete_ok is not None and delete_ok is False:
            msg = "The delete operation to VectorStore failed."
            raise IndexingException(msg)
    elif isinstance(vector_store, DocumentIndex):
        delete_response = vector_store.delete(ids)
        if "num_failed" in delete_response and delete_response["num_failed"] > 0:
            msg = "The delete operation to DocumentIndex failed."
            raise IndexingException(msg)
    else:
        msg = (
            f"Vectorstore should be either a VectorStore or a DocumentIndex. "
            f"Got {type(vector_store)}."
        )
        raise TypeError(msg)


# PUBLIC API
# 中文: 公共API


class IndexingResult(TypedDict):
    """Return a detailed a breakdown of the result of the indexing operation.

    中文翻译:
    返回索引操作结果的详细分类。"""

    num_added: int
    """Number of added documents.

    中文翻译:
    添加的文档数量。"""
    num_updated: int
    """Number of updated documents because they were not up to date.

    中文翻译:
    由于不是最新的而更新的文档数量。"""
    num_deleted: int
    """Number of deleted documents.

    中文翻译:
    已删除文档的数量。"""
    num_skipped: int
    """Number of skipped documents because they were already up to date.

    中文翻译:
    由于文档已是最新而跳过的文档数量。"""


def index(
    docs_source: BaseLoader | Iterable[Document],
    record_manager: RecordManager,
    vector_store: VectorStore | DocumentIndex,
    *,
    batch_size: int = 100,
    cleanup: Literal["incremental", "full", "scoped_full"] | None = None,
    source_id_key: str | Callable[[Document], str] | None = None,
    cleanup_batch_size: int = 1_000,
    force_update: bool = False,
    key_encoder: Literal["sha1", "sha256", "sha512", "blake2b"]
    | Callable[[Document], str] = "sha1",
    upsert_kwargs: dict[str, Any] | None = None,
) -> IndexingResult:
    """Index data from the loader into the vector store.

    Indexing functionality uses a manager to keep track of which documents
    are in the vector store.

    This allows us to keep track of which documents were updated, and which
    documents were deleted, which documents should be skipped.

    For the time being, documents are indexed using their hashes, and users
    are not able to specify the uid of the document.

    !!! warning "Behavior changed in `langchain-core` 0.3.25"

        Added `scoped_full` cleanup mode.

    !!! warning

        * In full mode, the loader should be returning
            the entire dataset, and not just a subset of the dataset.
            Otherwise, the auto_cleanup will remove documents that it is not
            supposed to.
        * In incremental mode, if documents associated with a particular
            source id appear across different batches, the indexing API
            will do some redundant work. This will still result in the
            correct end state of the index, but will unfortunately not be
            100% efficient. For example, if a given document is split into 15
            chunks, and we index them using a batch size of 5, we'll have 3 batches
            all with the same source id. In general, to avoid doing too much
            redundant work select as big a batch size as possible.
        * The `scoped_full` mode is suitable if determining an appropriate batch size
            is challenging or if your data loader cannot return the entire dataset at
            once. This mode keeps track of source IDs in memory, which should be fine
            for most use cases. If your dataset is large (10M+ docs), you will likely
            need to parallelize the indexing process regardless.

    Args:
        docs_source: Data loader or iterable of documents to index.
        record_manager: Timestamped set to keep track of which documents were
            updated.
        vector_store: `VectorStore` or DocumentIndex to index the documents into.
        batch_size: Batch size to use when indexing.
        cleanup: How to handle clean up of documents.

            - incremental: Cleans up all documents that haven't been updated AND
                that are associated with source IDs that were seen during indexing.
                Clean up is done continuously during indexing helping to minimize the
                probability of users seeing duplicated content.
            - full: Delete all documents that have not been returned by the loader
                during this run of indexing.
                Clean up runs after all documents have been indexed.
                This means that users may see duplicated content during indexing.
            - scoped_full: Similar to Full, but only deletes all documents
                that haven't been updated AND that are associated with
                source IDs that were seen during indexing.
            - None: Do not delete any documents.
        source_id_key: Optional key that helps identify the original source
            of the document.
        cleanup_batch_size: Batch size to use when cleaning up documents.
        force_update: Force update documents even if they are present in the
            record manager. Useful if you are re-indexing with updated embeddings.
        key_encoder: Hashing algorithm to use for hashing the document content and
            metadata. Options include "blake2b", "sha256", and "sha512".

            !!! version-added "Added in `langchain-core` 0.3.66"

        key_encoder: Hashing algorithm to use for hashing the document.
            If not provided, a default encoder using SHA-1 will be used.
            SHA-1 is not collision-resistant, and a motivated attacker
            could craft two different texts that hash to the
            same cache key.

            New applications should use one of the alternative encoders
            or provide a custom and strong key encoder function to avoid this risk.

            When changing the key encoder, you must change the
            index as well to avoid duplicated documents in the cache.
        upsert_kwargs: Additional keyword arguments to pass to the add_documents
            method of the `VectorStore` or the upsert method of the DocumentIndex.
            For example, you can use this to specify a custom vector_field:
            upsert_kwargs={"vector_field": "embedding"}
            !!! version-added "Added in `langchain-core` 0.3.10"

    Returns:
        Indexing result which contains information about how many documents
        were added, updated, deleted, or skipped.

    Raises:
        ValueError: If cleanup mode is not one of 'incremental', 'full' or None
        ValueError: If cleanup mode is incremental and source_id_key is None.
        ValueError: If `VectorStore` does not have
            "delete" and "add_documents" required methods.
        ValueError: If source_id_key is not None, but is not a string or callable.
        TypeError: If `vectorstore` is not a `VectorStore` or a DocumentIndex.
        AssertionError: If `source_id` is None when cleanup mode is incremental.
            (should be unreachable code).
    

    中文翻译:
    将加载器中的数据索引到向量存储中。
    索引功能使用管理器来跟踪哪些文档
    都在矢量存储中。
    这使我们能够跟踪哪些文档已更新以及哪些文档已更新
    文档已被删除，应跳过哪些文档。
    目前，文档使用哈希值进行索引，用户
    无法指定文档的 uid。
    !!!警告“‘langchain-core’ 0.3.25 中的行为已更改”
        添加了“scoped_full”清理模式。
    !!!警告
        * 在完整模式下，加载程序应该返回
            整个数据集，而不仅仅是数据集的子集。
            否则，auto_cleanup 将删除不属于它的文档
            应该。
        * 在增量模式下，如果文档与特定
            源id出现在不同的批次中，索引API
            会做一些多余的工作。这仍然会导致
            索引的正确结束状态，但不幸的是不是
            100% 有效。例如，如果给定文档被分成 15 个
            块，我们使用批量大小 5 对它们进行索引，我们将有 3 个批量
            全部具有相同的源 ID。一般来说，避免做太多
            冗余工作选择尽可能大的批量大小。
        * 如果确定合适的批量大小，则适合使用 `scoped_full` 模式
            具有挑战性，或者如果您的数据加载器无法返回整个数据集
            一次。此模式会跟踪内存中的源 ID，这应该没问题
            对于大多数用例。如果您的数据集很大（10M+ 文档），您可能会
            无论如何都需要并行化索引过程。
    参数：
        docs_source：要索引的数据加载器或可迭代文档。
        record_manager：设置时间戳以跟踪哪些文档
            已更新。
        vector_store：用于索引文档的“VectorStore”或 DocumentIndex。
        batch_size：索引时使用的批量大小。
        cleanup：如何处理文档的清理。
            - 增量：清理所有尚未更新的文档并且
                与索引期间看到的源 ID 关联。
                在索引过程中不断进行清理，有助于最大限度地减少
                用户看到重复内容的概率。
            - full：删除加载器未返回的所有文档
                在本次索引运行期间。
                所有文档都建立索引后运行清理。
                这意味着用户在索引过程中可能会看到重复的内容。
            -scoped_full：与Full类似，但只删除所有文档
                尚未更新且关联的
                在索引过程中看到的源 ID。
            - 无：不删除任何文档。
        source_id_key：可选密钥，有助于识别原始来源
            该文件的。
        cleanup_batch_size：清理文档时使用的批量大小。
        force_update：强制更新文档，即使它们存在于
            记录经理。如果您使用更新的嵌入重新索引，则非常有用。
        key_encoder：用于对文档内容进行哈希处理的哈希算法
            元数据。选项包括“blake2b”、“sha256”和“sha512”。
            !!! version-added “在 `langchain-core` 0.3.66 中添加”
        key_encoder：用于对文档进行哈希处理的哈希算法。
            如果未提供，将使用使用 SHA-1 的默认编码器。
            SHA-1 不具有抗碰撞性，且攻击者有动机
            可以制作两个不同的文本，散列到
            相同的缓存键。
            新应用程序应使用替代编码器之一
            或者提供自定义且强大的按键编码器功能来避免这种风险。
            更改按键编码器时，必须更改
            索引也可以避免缓存中出现重复的文档。
        upsert_kwargs：传递给 add_documents 的附加关键字参数
            `VectorStore` 的方法或 DocumentIndex 的 upsert 方法。
            例如，您可以使用它来指定自定义向量字段：
            upsert_kwargs={"vector_field": "嵌入"}!!! version-added “在 `langchain-core` 0.3.10 中添加”
    返回：
        索引结果包含有关多少文档的信息
        被添加、更新、删除或跳过。
    加薪：
        ValueError：如果清理模式不是“增量”、“完整”或“无”之一
        ValueError：如果清理模式为增量且 source_id_key 为 None。
        ValueError：如果 `VectorStore` 没有
            “delete”和“add_documents”必需的方法。
        ValueError：如果 source_id_key 不是 None，但不是字符串或可调用。
        类型错误：如果“vectorstore”不是“VectorStore”或 DocumentIndex。
        AssertionError：当清理模式为增量时，如果 `source_id` 为 None。
            （应该是无法访问的代码）。"""
    # Behavior is deprecated, but we keep it for backwards compatibility.
    # 中文: 该行为已被弃用，但我们保留它是为了向后兼容。
    # # Warn only once per process.
    # 中文: # 每个进程仅警告一次。
    if key_encoder == "sha1":
        _warn_about_sha1()

    if cleanup not in {"incremental", "full", "scoped_full", None}:
        msg = (
            f"cleanup should be one of 'incremental', 'full', 'scoped_full' or None. "
            f"Got {cleanup}."
        )
        raise ValueError(msg)

    if (cleanup in {"incremental", "scoped_full"}) and source_id_key is None:
        msg = (
            "Source id key is required when cleanup mode is incremental or scoped_full."
        )
        raise ValueError(msg)

    destination = vector_store  # Renaming internally for clarity

    # If it's a vectorstore, let's check if it has the required methods.
    # 中文: 如果它是一个向量存储，让我们检查它是否具有所需的方法。
    if isinstance(destination, VectorStore):
        # Check that the Vectorstore has required methods implemented
        # 中文: 检查 Vectorstore 是否已实现所需的方法
        methods = ["delete", "add_documents"]

        for method in methods:
            if not hasattr(destination, method):
                msg = (
                    f"Vectorstore {destination} does not have required method {method}"
                )
                raise ValueError(msg)

        if type(destination).delete == VectorStore.delete:
            # Checking if the VectorStore has overridden the default delete method
            # 中文: 检查 VectorStore 是否覆盖了默认的删除方法
            # implementation which just raises a NotImplementedError
            # 中文: 仅引发 NotImplementedError 的实现
            msg = "Vectorstore has not implemented the delete method"
            raise ValueError(msg)
    elif isinstance(destination, DocumentIndex):
        pass
    else:
        msg = (
            f"Vectorstore should be either a VectorStore or a DocumentIndex. "
            f"Got {type(destination)}."
        )
        raise TypeError(msg)

    if isinstance(docs_source, BaseLoader):
        try:
            doc_iterator = docs_source.lazy_load()
        except NotImplementedError:
            doc_iterator = iter(docs_source.load())
    else:
        doc_iterator = iter(docs_source)

    source_id_assigner = _get_source_id_assigner(source_id_key)

    # Mark when the update started.
    # 中文: 标记更新开始的时间。
    index_start_dt = record_manager.get_time()
    num_added = 0
    num_skipped = 0
    num_updated = 0
    num_deleted = 0
    scoped_full_cleanup_source_ids: set[str] = set()

    for doc_batch in _batch(batch_size, doc_iterator):
        # Track original batch size before deduplication
        # 中文: 跟踪重复数据删除前的原始批次大小
        original_batch_size = len(doc_batch)

        hashed_docs = list(
            _deduplicate_in_order(
                [
                    _get_document_with_hash(doc, key_encoder=key_encoder)
                    for doc in doc_batch
                ]
            )
        )
        # Count documents removed by within-batch deduplication
        # 中文: 计算批内重复数据删除删除的文档数量
        num_skipped += original_batch_size - len(hashed_docs)

        source_ids: Sequence[str | None] = [
            source_id_assigner(hashed_doc) for hashed_doc in hashed_docs
        ]

        if cleanup in {"incremental", "scoped_full"}:
            # Source IDs are required.
            # 中文: 需要源 ID。
            for source_id, hashed_doc in zip(source_ids, hashed_docs, strict=False):
                if source_id is None:
                    msg = (
                        f"Source IDs are required when cleanup mode is "
                        f"incremental or scoped_full. "
                        f"Document that starts with "
                        f"content: {hashed_doc.page_content[:100]} "
                        f"was not assigned as source id."
                    )
                    raise ValueError(msg)
                if cleanup == "scoped_full":
                    scoped_full_cleanup_source_ids.add(source_id)
            # Source IDs cannot be None after for loop above.
            # 中文: 在上面的 for 循环之后，源 ID 不能为 None。
            source_ids = cast("Sequence[str]", source_ids)

        exists_batch = record_manager.exists(
            cast("Sequence[str]", [doc.id for doc in hashed_docs])
        )

        # Filter out documents that already exist in the record store.
        # 中文: 过滤掉记录存储中已存在的文档。
        uids = []
        docs_to_index = []
        uids_to_refresh = []
        seen_docs: set[str] = set()
        for hashed_doc, doc_exists in zip(hashed_docs, exists_batch, strict=False):
            hashed_id = cast("str", hashed_doc.id)
            if doc_exists:
                if force_update:
                    seen_docs.add(hashed_id)
                else:
                    uids_to_refresh.append(hashed_id)
                    continue
            uids.append(hashed_id)
            docs_to_index.append(hashed_doc)

        # Update refresh timestamp
        # 中文: 更新刷新时间戳
        if uids_to_refresh:
            record_manager.update(uids_to_refresh, time_at_least=index_start_dt)
            num_skipped += len(uids_to_refresh)

        # Be pessimistic and assume that all vector store write will fail.
        # 中文: 持悲观态度并假设所有向量存储写入都会失败。
        # First write to vector store
        # 中文: 首先写入向量存储
        if docs_to_index:
            if isinstance(destination, VectorStore):
                destination.add_documents(
                    docs_to_index,
                    ids=uids,
                    batch_size=batch_size,
                    **(upsert_kwargs or {}),
                )
            elif isinstance(destination, DocumentIndex):
                destination.upsert(
                    docs_to_index,
                    **(upsert_kwargs or {}),
                )

            num_added += len(docs_to_index) - len(seen_docs)
            num_updated += len(seen_docs)

        # And only then update the record store.
        # 中文: 然后才更新记录存储。
        # Update ALL records, even if they already exist since we want to refresh
        # 中文: 更新所有记录，即使它们已经存在，因为我们要刷新
        # their timestamp.
        # 中文: 他们的时间戳。
        record_manager.update(
            cast("Sequence[str]", [doc.id for doc in hashed_docs]),
            group_ids=source_ids,
            time_at_least=index_start_dt,
        )

        # If source IDs are provided, we can do the deletion incrementally!
        # 中文: 如果提供了源 ID，我们可以增量删除！
        if cleanup == "incremental":
            # Get the uids of the documents that were not returned by the loader.
            # 中文: 获取加载器未返回的文档的 uid。
            # mypy isn't good enough to determine that source IDs cannot be None
            # 中文: mypy 不足以确定源 ID 不能为 None
            # here due to a check that's happening above, so we check again.
            # 中文: 由于上面发生了检查，所以我们再次检查。
            for source_id in source_ids:
                if source_id is None:
                    msg = (
                        "source_id cannot be None at this point. "
                        "Reached unreachable code."
                    )
                    raise AssertionError(msg)

            source_ids_ = cast("Sequence[str]", source_ids)

            while uids_to_delete := record_manager.list_keys(
                group_ids=source_ids_, before=index_start_dt, limit=cleanup_batch_size
            ):
                # Then delete from vector store.
                # 中文: 然后从矢量存储中删除。
                _delete(destination, uids_to_delete)
                # First delete from record store.
                # 中文: 首先从记录存储中删除。
                record_manager.delete_keys(uids_to_delete)
                num_deleted += len(uids_to_delete)

    if cleanup == "full" or (
        cleanup == "scoped_full" and scoped_full_cleanup_source_ids
    ):
        delete_group_ids: Sequence[str] | None = None
        if cleanup == "scoped_full":
            delete_group_ids = list(scoped_full_cleanup_source_ids)
        while uids_to_delete := record_manager.list_keys(
            group_ids=delete_group_ids, before=index_start_dt, limit=cleanup_batch_size
        ):
            # First delete from record store.
            # 中文: 首先从记录存储中删除。
            _delete(destination, uids_to_delete)
            # Then delete from record manager.
            # 中文: 然后从记录管理器中删除。
            record_manager.delete_keys(uids_to_delete)
            num_deleted += len(uids_to_delete)

    return {
        "num_added": num_added,
        "num_updated": num_updated,
        "num_skipped": num_skipped,
        "num_deleted": num_deleted,
    }


# Define an asynchronous generator function
# 中文: 定义异步生成器函数
async def _to_async_iterator(iterator: Iterable[T]) -> AsyncIterator[T]:
    """Convert an iterable to an async iterator.

    中文翻译:
    将可迭代器转换为异步迭代器。"""
    for item in iterator:
        yield item


async def _adelete(
    vector_store: VectorStore | DocumentIndex,
    ids: list[str],
) -> None:
    if isinstance(vector_store, VectorStore):
        delete_ok = await vector_store.adelete(ids)
        if delete_ok is not None and delete_ok is False:
            msg = "The delete operation to VectorStore failed."
            raise IndexingException(msg)
    elif isinstance(vector_store, DocumentIndex):
        delete_response = await vector_store.adelete(ids)
        if "num_failed" in delete_response and delete_response["num_failed"] > 0:
            msg = "The delete operation to DocumentIndex failed."
            raise IndexingException(msg)
    else:
        msg = (
            f"Vectorstore should be either a VectorStore or a DocumentIndex. "
            f"Got {type(vector_store)}."
        )
        raise TypeError(msg)


async def aindex(
    docs_source: BaseLoader | Iterable[Document] | AsyncIterator[Document],
    record_manager: RecordManager,
    vector_store: VectorStore | DocumentIndex,
    *,
    batch_size: int = 100,
    cleanup: Literal["incremental", "full", "scoped_full"] | None = None,
    source_id_key: str | Callable[[Document], str] | None = None,
    cleanup_batch_size: int = 1_000,
    force_update: bool = False,
    key_encoder: Literal["sha1", "sha256", "sha512", "blake2b"]
    | Callable[[Document], str] = "sha1",
    upsert_kwargs: dict[str, Any] | None = None,
) -> IndexingResult:
    """Async index data from the loader into the vector store.

    Indexing functionality uses a manager to keep track of which documents
    are in the vector store.

    This allows us to keep track of which documents were updated, and which
    documents were deleted, which documents should be skipped.

    For the time being, documents are indexed using their hashes, and users
    are not able to specify the uid of the document.

    !!! warning "Behavior changed in `langchain-core` 0.3.25"

        Added `scoped_full` cleanup mode.

    !!! warning

        * In full mode, the loader should be returning
            the entire dataset, and not just a subset of the dataset.
            Otherwise, the auto_cleanup will remove documents that it is not
            supposed to.
        * In incremental mode, if documents associated with a particular
            source id appear across different batches, the indexing API
            will do some redundant work. This will still result in the
            correct end state of the index, but will unfortunately not be
            100% efficient. For example, if a given document is split into 15
            chunks, and we index them using a batch size of 5, we'll have 3 batches
            all with the same source id. In general, to avoid doing too much
            redundant work select as big a batch size as possible.
        * The `scoped_full` mode is suitable if determining an appropriate batch size
            is challenging or if your data loader cannot return the entire dataset at
            once. This mode keeps track of source IDs in memory, which should be fine
            for most use cases. If your dataset is large (10M+ docs), you will likely
            need to parallelize the indexing process regardless.

    Args:
        docs_source: Data loader or iterable of documents to index.
        record_manager: Timestamped set to keep track of which documents were
            updated.
        vector_store: `VectorStore` or DocumentIndex to index the documents into.
        batch_size: Batch size to use when indexing.
        cleanup: How to handle clean up of documents.

            - incremental: Cleans up all documents that haven't been updated AND
                that are associated with source IDs that were seen during indexing.
                Clean up is done continuously during indexing helping to minimize the
                probability of users seeing duplicated content.
            - full: Delete all documents that have not been returned by the loader
                during this run of indexing.
                Clean up runs after all documents have been indexed.
                This means that users may see duplicated content during indexing.
            - scoped_full: Similar to Full, but only deletes all documents
                that haven't been updated AND that are associated with
                source IDs that were seen during indexing.
            - None: Do not delete any documents.
        source_id_key: Optional key that helps identify the original source
            of the document.
        cleanup_batch_size: Batch size to use when cleaning up documents.
        force_update: Force update documents even if they are present in the
            record manager. Useful if you are re-indexing with updated embeddings.
        key_encoder: Hashing algorithm to use for hashing the document content and
            metadata. Options include "blake2b", "sha256", and "sha512".

            !!! version-added "Added in `langchain-core` 0.3.66"

        key_encoder: Hashing algorithm to use for hashing the document.
            If not provided, a default encoder using SHA-1 will be used.
            SHA-1 is not collision-resistant, and a motivated attacker
            could craft two different texts that hash to the
            same cache key.

            New applications should use one of the alternative encoders
            or provide a custom and strong key encoder function to avoid this risk.

            When changing the key encoder, you must change the
            index as well to avoid duplicated documents in the cache.
        upsert_kwargs: Additional keyword arguments to pass to the add_documents
            method of the `VectorStore` or the upsert method of the DocumentIndex.
            For example, you can use this to specify a custom vector_field:
            upsert_kwargs={"vector_field": "embedding"}
            !!! version-added "Added in `langchain-core` 0.3.10"

    Returns:
        Indexing result which contains information about how many documents
        were added, updated, deleted, or skipped.

    Raises:
        ValueError: If cleanup mode is not one of 'incremental', 'full' or None
        ValueError: If cleanup mode is incremental and source_id_key is None.
        ValueError: If `VectorStore` does not have
            "adelete" and "aadd_documents" required methods.
        ValueError: If source_id_key is not None, but is not a string or callable.
        TypeError: If `vector_store` is not a `VectorStore` or DocumentIndex.
        AssertionError: If `source_id_key` is None when cleanup mode is
            incremental or `scoped_full` (should be unreachable).
    

    中文翻译:
    将异步索引数据从加载器加载到向量存储中。
    索引功能使用管理器来跟踪哪些文档
    都在矢量存储中。
    这使我们能够跟踪哪些文档已更新以及哪些文档已更新
    文档已被删除，应跳过哪些文档。
    目前，文档使用哈希值进行索引，用户
    无法指定文档的 uid。
    ！！！警告“‘langchain-core’ 0.3.25 中的行为已更改”
        添加了“scoped_full”清理模式。
    ！！！警告
        * 在完整模式下，加载程序应该返回
            整个数据集，而不仅仅是数据集的子集。
            否则，auto_cleanup 将删除不属于它的文档
            应该。
        * 在增量模式下，如果文档与特定
            源id出现在不同的批次中，索引API
            会做一些多余的工作。这仍然会导致
            索引的正确结束状态，但不幸的是不是
            100% 有效。例如，如果给定文档被分成 15 个
            块，我们使用批量大小 5 对它们进行索引，我们将有 3 个批量
            全部具有相同的源 ID。一般来说，避免做太多
            冗余工作选择尽可能大的批量大小。
        * 如果确定合适的批量大小，则适合使用 `scoped_full` 模式
            具有挑战性，或者如果您的数据加载器无法返回整个数据集
            一次。此模式会跟踪内存中的源 ID，这应该没问题
            对于大多数用例。如果您的数据集很大（10M+ 文档），您可能会
            无论如何都需要并行化索引过程。
    参数：
        docs_source：要索引的数据加载器或可迭代文档。
        record_manager：设置时间戳以跟踪哪些文档
            已更新。
        vector_store：用于索引文档的“VectorStore”或 DocumentIndex。
        batch_size：索引时使用的批量大小。
        cleanup：如何处理文档的清理。
            - 增量：清理所有尚未更新的文档并且
                与索引期间看到的源 ID 关联。
                在索引过程中不断进行清理，有助于最大限度地减少
                用户看到重复内容的概率。
            - full：删除加载器未返回的所有文档
                在本次索引运行期间。
                所有文档都建立索引后运行清理。
                这意味着用户在索引过程中可能会看到重复的内容。
            -scoped_full：与Full类似，但只删除所有文档
                尚未更新且关联的
                在索引过程中看到的源 ID。
            - 无：不删除任何文档。
        source_id_key：可选密钥，有助于识别原始来源
            该文件的。
        cleanup_batch_size：清理文档时使用的批量大小。
        force_update：强制更新文档，即使它们存在于
            记录经理。如果您使用更新的嵌入重新索引，则非常有用。
        key_encoder：用于对文档内容进行哈希处理的哈希算法
            元数据。选项包括“blake2b”、“sha256”和“sha512”。
            ！！！ version-added “在 `langchain-core` 0.3.66 中添加”
        key_encoder：用于对文档进行哈希处理的哈希算法。
            如果未提供，将使用使用 SHA-1 的默认编码器。
            SHA-1 不具有抗碰撞性，且攻击者有动机
            可以制作两个不同的文本，散列到
            相同的缓存键。
            新应用程序应使用替代编码器之一
            或者提供自定义且强大的按键编码器功能来避免这种风险。
            更改按键编码器时，必须更改
            索引也可以避免缓存中出现重复的文档。
        upsert_kwargs：传递给 add_documents 的附加关键字参数
            `VectorStore` 的方法或 DocumentIndex 的 upsert 方法。
            例如，您可以使用它来指定自定义向量字段：
            upsert_kwargs={"vector_field": "嵌入"}！！！ version-added “在 `langchain-core` 0.3.10 中添加”
    返回：
        索引结果包含有关多少文档的信息
        被添加、更新、删除或跳过。
    加薪：
        ValueError：如果清理模式不是“增量”、“完整”或“无”之一
        ValueError：如果清理模式为增量且 source_id_key 为 None。
        ValueError：如果 `VectorStore` 没有
            “adelete”和“aadd_documents”必需方法。
        ValueError：如果 source_id_key 不是 None，但不是字符串或可调用。
        类型错误：如果“vector_store”不是“VectorStore”或 DocumentIndex。
        AssertionError：当清理模式为“source_id_key”时，如果“source_id_key”为 None
            增量或“scoped_full”（应该无法访问）。"""
    # Behavior is deprecated, but we keep it for backwards compatibility.
    # 中文: 该行为已被弃用，但我们保留它是为了向后兼容。
    # # Warn only once per process.
    # 中文: # 每个进程仅警告一次。
    if key_encoder == "sha1":
        _warn_about_sha1()

    if cleanup not in {"incremental", "full", "scoped_full", None}:
        msg = (
            f"cleanup should be one of 'incremental', 'full', 'scoped_full' or None. "
            f"Got {cleanup}."
        )
        raise ValueError(msg)

    if (cleanup in {"incremental", "scoped_full"}) and source_id_key is None:
        msg = (
            "Source id key is required when cleanup mode is incremental or scoped_full."
        )
        raise ValueError(msg)

    destination = vector_store  # Renaming internally for clarity

    # If it's a vectorstore, let's check if it has the required methods.
    # 中文: 如果它是一个向量存储，让我们检查它是否具有所需的方法。
    if isinstance(destination, VectorStore):
        # Check that the Vectorstore has required methods implemented
        # 中文: 检查 Vectorstore 是否已实现所需的方法
        # Check that the Vectorstore has required methods implemented
        # 中文: 检查 Vectorstore 是否已实现所需的方法
        methods = ["adelete", "aadd_documents"]

        for method in methods:
            if not hasattr(destination, method):
                msg = (
                    f"Vectorstore {destination} does not have required method {method}"
                )
                raise ValueError(msg)

        if (
            type(destination).adelete == VectorStore.adelete
            and type(destination).delete == VectorStore.delete
        ):
            # Checking if the VectorStore has overridden the default adelete or delete
            # 中文: 检查VectorStore是否覆盖了默认的adelete或delete
            # methods implementation which just raises a NotImplementedError
            # 中文: 方法实现只会引发 NotImplementedError
            msg = "Vectorstore has not implemented the adelete or delete method"
            raise ValueError(msg)
    elif isinstance(destination, DocumentIndex):
        pass
    else:
        msg = (
            f"Vectorstore should be either a VectorStore or a DocumentIndex. "
            f"Got {type(destination)}."
        )
        raise TypeError(msg)
    async_doc_iterator: AsyncIterator[Document]
    if isinstance(docs_source, BaseLoader):
        try:
            async_doc_iterator = docs_source.alazy_load()
        except NotImplementedError:
            # Exception triggered when neither lazy_load nor alazy_load are implemented.
            # 中文: 当lazy_load和alazy_load都没有实现时触发异常。
            # * The default implementation of alazy_load uses lazy_load.
            # 中文: * alazy_load的默认实现使用lazy_load。
            # * The default implementation of lazy_load raises NotImplementedError.
            # 中文: *lazy_load 的默认实现会引发 NotImplementedError。
            # In such a case, we use the load method and convert it to an async
            # 中文: 在这种情况下，我们使用 load 方法并将其转换为异步方法
            # iterator.
            # 中文: 迭代器。
            async_doc_iterator = _to_async_iterator(docs_source.load())
    elif hasattr(docs_source, "__aiter__"):
        async_doc_iterator = docs_source  # type: ignore[assignment]
    else:
        async_doc_iterator = _to_async_iterator(docs_source)

    source_id_assigner = _get_source_id_assigner(source_id_key)

    # Mark when the update started.
    # 中文: 标记更新开始的时间。
    index_start_dt = await record_manager.aget_time()
    num_added = 0
    num_skipped = 0
    num_updated = 0
    num_deleted = 0
    scoped_full_cleanup_source_ids: set[str] = set()

    async for doc_batch in _abatch(batch_size, async_doc_iterator):
        # Track original batch size before deduplication
        # 中文: 跟踪重复数据删除前的原始批次大小
        original_batch_size = len(doc_batch)

        hashed_docs = list(
            _deduplicate_in_order(
                [
                    _get_document_with_hash(doc, key_encoder=key_encoder)
                    for doc in doc_batch
                ]
            )
        )
        # Count documents removed by within-batch deduplication
        # 中文: 计算批内重复数据删除删除的文档数量
        num_skipped += original_batch_size - len(hashed_docs)

        source_ids: Sequence[str | None] = [
            source_id_assigner(doc) for doc in hashed_docs
        ]

        if cleanup in {"incremental", "scoped_full"}:
            # If the cleanup mode is incremental, source IDs are required.
            # 中文: 如果清理模式是增量的，则需要源ID。
            for source_id, hashed_doc in zip(source_ids, hashed_docs, strict=False):
                if source_id is None:
                    msg = (
                        f"Source IDs are required when cleanup mode is "
                        f"incremental or scoped_full. "
                        f"Document that starts with "
                        f"content: {hashed_doc.page_content[:100]} "
                        f"was not assigned as source id."
                    )
                    raise ValueError(msg)
                if cleanup == "scoped_full":
                    scoped_full_cleanup_source_ids.add(source_id)
            # Source IDs cannot be None after for loop above.
            # 中文: 在上面的 for 循环之后，源 ID 不能为 None。
            source_ids = cast("Sequence[str]", source_ids)

        exists_batch = await record_manager.aexists(
            cast("Sequence[str]", [doc.id for doc in hashed_docs])
        )

        # Filter out documents that already exist in the record store.
        # 中文: 过滤掉记录存储中已存在的文档。
        uids: list[str] = []
        docs_to_index: list[Document] = []
        uids_to_refresh = []
        seen_docs: set[str] = set()
        for hashed_doc, doc_exists in zip(hashed_docs, exists_batch, strict=False):
            hashed_id = cast("str", hashed_doc.id)
            if doc_exists:
                if force_update:
                    seen_docs.add(hashed_id)
                else:
                    uids_to_refresh.append(hashed_id)
                    continue
            uids.append(hashed_id)
            docs_to_index.append(hashed_doc)

        if uids_to_refresh:
            # Must be updated to refresh timestamp.
            # 中文: 必须更新以刷新时间戳。
            await record_manager.aupdate(uids_to_refresh, time_at_least=index_start_dt)
            num_skipped += len(uids_to_refresh)

        # Be pessimistic and assume that all vector store write will fail.
        # 中文: 持悲观态度并假设所有向量存储写入都会失败。
        # First write to vector store
        # 中文: 首先写入向量存储
        if docs_to_index:
            if isinstance(destination, VectorStore):
                await destination.aadd_documents(
                    docs_to_index,
                    ids=uids,
                    batch_size=batch_size,
                    **(upsert_kwargs or {}),
                )
            elif isinstance(destination, DocumentIndex):
                await destination.aupsert(
                    docs_to_index,
                    **(upsert_kwargs or {}),
                )
            num_added += len(docs_to_index) - len(seen_docs)
            num_updated += len(seen_docs)

        # And only then update the record store.
        # 中文: 然后才更新记录存储。
        # Update ALL records, even if they already exist since we want to refresh
        # 中文: 更新所有记录，即使它们已经存在，因为我们要刷新
        # their timestamp.
        # 中文: 他们的时间戳。
        await record_manager.aupdate(
            cast("Sequence[str]", [doc.id for doc in hashed_docs]),
            group_ids=source_ids,
            time_at_least=index_start_dt,
        )

        # If source IDs are provided, we can do the deletion incrementally!
        # 中文: 如果提供了源 ID，我们可以增量删除！

        if cleanup == "incremental":
            # Get the uids of the documents that were not returned by the loader.
            # 中文: 获取加载器未返回的文档的 uid。

            # mypy isn't good enough to determine that source IDs cannot be None
            # 中文: mypy 不足以确定源 ID 不能为 None
            # here due to a check that's happening above, so we check again.
            # 中文: 由于上面发生了检查，所以我们再次检查。
            for source_id in source_ids:
                if source_id is None:
                    msg = (
                        "source_id cannot be None at this point. "
                        "Reached unreachable code."
                    )
                    raise AssertionError(msg)

            source_ids_ = cast("Sequence[str]", source_ids)

            while uids_to_delete := await record_manager.alist_keys(
                group_ids=source_ids_, before=index_start_dt, limit=cleanup_batch_size
            ):
                # Then delete from vector store.
                # 中文: 然后从矢量存储中删除。
                await _adelete(destination, uids_to_delete)
                # First delete from record store.
                # 中文: 首先从记录存储中删除。
                await record_manager.adelete_keys(uids_to_delete)
                num_deleted += len(uids_to_delete)

    if cleanup == "full" or (
        cleanup == "scoped_full" and scoped_full_cleanup_source_ids
    ):
        delete_group_ids: Sequence[str] | None = None
        if cleanup == "scoped_full":
            delete_group_ids = list(scoped_full_cleanup_source_ids)
        while uids_to_delete := await record_manager.alist_keys(
            group_ids=delete_group_ids, before=index_start_dt, limit=cleanup_batch_size
        ):
            # First delete from record store.
            # 中文: 首先从记录存储中删除。
            await _adelete(destination, uids_to_delete)
            # Then delete from record manager.
            # 中文: 然后从记录管理器中删除。
            await record_manager.adelete_keys(uids_to_delete)
            num_deleted += len(uids_to_delete)

    return {
        "num_added": num_added,
        "num_updated": num_updated,
        "num_skipped": num_skipped,
        "num_deleted": num_deleted,
    }
