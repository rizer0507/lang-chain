from contextlib import asynccontextmanager, contextmanager

from langgraph.store.memory import InMemoryStore


@contextmanager
def _store_memory():
    store = InMemoryStore()
    yield store


@asynccontextmanager
async def _store_memory_aio():
    store = InMemoryStore()
    yield store


# Placeholder functions for other store types that aren't available
# 中文: 其他商店类型不可用的占位符功能
@contextmanager
def _store_postgres():
    # Fallback to memory for now
    # 中文: 现在回退到内存
    store = InMemoryStore()
    yield store


@contextmanager
def _store_postgres_pipe():
    # Fallback to memory for now
    # 中文: 现在回退到内存
    store = InMemoryStore()
    yield store


@contextmanager
def _store_postgres_pool():
    # Fallback to memory for now
    # 中文: 现在回退到内存
    store = InMemoryStore()
    yield store


@asynccontextmanager
async def _store_postgres_aio():
    # Fallback to memory for now
    # 中文: 现在回退到内存
    store = InMemoryStore()
    yield store


@asynccontextmanager
async def _store_postgres_aio_pipe():
    # Fallback to memory for now
    # 中文: 现在回退到内存
    store = InMemoryStore()
    yield store


@asynccontextmanager
async def _store_postgres_aio_pool():
    # Fallback to memory for now
    # 中文: 现在回退到内存
    store = InMemoryStore()
    yield store
