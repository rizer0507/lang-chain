from contextlib import asynccontextmanager, contextmanager

from .memory_assert import (
    MemorySaverAssertImmutable,
)


@contextmanager
def _checkpointer_memory():
    yield MemorySaverAssertImmutable()


@asynccontextmanager
async def _checkpointer_memory_aio():
    yield MemorySaverAssertImmutable()


# Placeholder functions for other checkpointer types that aren't available
# 中文: 其他不可用的检查指针类型的占位符函数
@contextmanager
def _checkpointer_sqlite():
    # Fallback to memory for now
    # 中文: 现在回退到内存
    yield MemorySaverAssertImmutable()


@contextmanager
def _checkpointer_postgres():
    # Fallback to memory for now
    # 中文: 现在回退到内存
    yield MemorySaverAssertImmutable()


@contextmanager
def _checkpointer_postgres_pipe():
    # Fallback to memory for now
    # 中文: 现在回退到内存
    yield MemorySaverAssertImmutable()


@contextmanager
def _checkpointer_postgres_pool():
    # Fallback to memory for now
    # 中文: 现在回退到内存
    yield MemorySaverAssertImmutable()


@asynccontextmanager
async def _checkpointer_sqlite_aio():
    # Fallback to memory for now
    # 中文: 现在回退到内存
    yield MemorySaverAssertImmutable()


@asynccontextmanager
async def _checkpointer_postgres_aio():
    # Fallback to memory for now
    # 中文: 现在回退到内存
    yield MemorySaverAssertImmutable()


@asynccontextmanager
async def _checkpointer_postgres_aio_pipe():
    # Fallback to memory for now
    # 中文: 现在回退到内存
    yield MemorySaverAssertImmutable()


@asynccontextmanager
async def _checkpointer_postgres_aio_pool():
    # Fallback to memory for now
    # 中文: 现在回退到内存
    yield MemorySaverAssertImmutable()
