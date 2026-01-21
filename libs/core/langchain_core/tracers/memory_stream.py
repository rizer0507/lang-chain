"""Module implements a memory stream for communication between two co-routines.

This module provides a way to communicate between two co-routines using a memory
channel. The writer and reader can be in the same event loop or in different event
loops. When they're in different event loops, they will also be in different
threads.

Useful in situations when there's a mix of synchronous and asynchronous
used in the code.

中文翻译:
模块实现两个协同例程之间通信的内存流。
该模块提供了一种使用内存在两个协同例程之间进行通信的方法
频道。写入器和读取器可以位于同一事件循环中，也可以位于不同事件中
循环。当它们处于不同的事件循环时，它们也会处于不同的状态
线程。
在同步和异步混合的情况下很有用
在代码中使用。
"""

import asyncio
from asyncio import AbstractEventLoop, Queue
from collections.abc import AsyncIterator
from typing import Generic, TypeVar

T = TypeVar("T")


class _SendStream(Generic[T]):
    def __init__(
        self, reader_loop: AbstractEventLoop, queue: Queue, done: object
    ) -> None:
        """Create a writer for the queue and done object.

        Args:
            reader_loop: The event loop to use for the writer. This loop will be used
                         to schedule the writes to the queue.
            queue: The queue to write to. This is an asyncio queue.
            done: Special sentinel object to indicate that the writer is done.
        

        中文翻译:
        为队列和完成对象创建一个编写器。
        参数：
            reader_loop：供 writer 使用的事件循环。将使用此循环
                         安排对队列的写入。
            队列：要写入的队列。这是一个异步队列。
            完成：特殊哨兵对象，指示作者已完成。"""
        self._reader_loop = reader_loop
        self._queue = queue
        self._done = done

    async def send(self, item: T) -> None:
        """Schedule the item to be written to the queue using the original loop.

        This is a coroutine that can be awaited.

        Args:
            item: The item to write to the queue.
        

        中文翻译:
        使用原始循环安排要写入队列的项目。
        这是一个可以等待的协程。
        参数：
            item：要写入队列的项目。"""
        return self.send_nowait(item)

    def send_nowait(self, item: T) -> None:
        """Schedule the item to be written to the queue using the original loop.

        This is a non-blocking call.

        Args:
            item: The item to write to the queue.

        Raises:
            RuntimeError: If the event loop is already closed when trying to write
                            to the queue.
        

        中文翻译:
        使用原始循环安排要写入队列的项目。
        这是一个非阻塞调用。
        参数：
            item：要写入队列的项目。
        加薪：
            RuntimeError：如果尝试写入时事件循环已关闭
                            到队列中。"""
        try:
            self._reader_loop.call_soon_threadsafe(self._queue.put_nowait, item)
        except RuntimeError:
            if not self._reader_loop.is_closed():
                raise  # Raise the exception if the loop is not closed

    async def aclose(self) -> None:
        """Async schedule the done object write the queue using the original loop.

        中文翻译:
        异步调度完成的对象使用原始循环写入队列。"""
        return self.close()

    def close(self) -> None:
        """Schedule the done object write the queue using the original loop.

        This is a non-blocking call.

        Raises:
            RuntimeError: If the event loop is already closed when trying to write
                            to the queue.
        

        中文翻译:
        使用原始循环安排完成的对象写入队列。
        这是一个非阻塞调用。
        加薪：
            RuntimeError：如果尝试写入时事件循环已关闭
                            到队列中。"""
        try:
            self._reader_loop.call_soon_threadsafe(self._queue.put_nowait, self._done)
        except RuntimeError:
            if not self._reader_loop.is_closed():
                raise  # Raise the exception if the loop is not closed


class _ReceiveStream(Generic[T]):
    def __init__(self, queue: Queue, done: object) -> None:
        """Create a reader for the queue and done object.

        This reader should be used in the same loop as the loop that was passed
        to the channel.
        

        中文翻译:
        为队列和完成对象创建一个读取器。
        该读取器应该在与传递的循环相同的循环中使用
        到频道。"""
        self._queue = queue
        self._done = done
        self._is_closed = False

    async def __aiter__(self) -> AsyncIterator[T]:
        while True:
            item = await self._queue.get()
            if item is self._done:
                self._is_closed = True
                break
            yield item


class _MemoryStream(Generic[T]):
    """Stream data from a writer to a reader even if they are in different threads.

    Uses asyncio queues to communicate between two co-routines. This implementation
    should work even if the writer and reader co-routines belong to two different
    event loops (e.g. one running from an event loop in the main thread
    and the other running in an event loop in a background thread).

    This implementation is meant to be used with a single writer and a single reader.

    This is an internal implementation to LangChain. Please do not use it directly.
    

    中文翻译:
    将数据从写入器流式传输到读取器，即使它们位于不同的线程中。
    使用异步队列在两个协同例程之间进行通信。本次实施
    即使编写器和读取器协同例程属于两个不同的程序，也应该起作用
    事件循环（例如，从主线程中的事件循环运行）
    另一个在后台线程的事件循环中运行）。
    此实现旨在与单个写入器和单个读取器一起使用。
    这是LangChain的内部实现。请不要直接使用它。"""

    def __init__(self, loop: AbstractEventLoop) -> None:
        """Create a channel for the given loop.

        Args:
            loop: The event loop to use for the channel. The reader is assumed
                  to be running in the same loop as the one passed to this constructor.
                  This will NOT be validated at run time.
        

        中文翻译:
        为给定循环创建一个通道。
        参数：
            循环：用于通道的事件循环。假设读者
                  与传递给此构造函数的循环在同一循环中运行。
                  这不会在运行时验证。"""
        self._loop = loop
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=0)
        self._done = object()

    def get_send_stream(self) -> _SendStream[T]:
        """Get a writer for the channel.

        Returns:
            The writer for the channel.
        

        中文翻译:
        为该频道找一位作家。
        返回：
            该频道的撰稿人。"""
        return _SendStream[T](
            reader_loop=self._loop, queue=self._queue, done=self._done
        )

    def get_receive_stream(self) -> _ReceiveStream[T]:
        """Get a reader for the channel.

        Returns:
            The reader for the channel.
        

        中文翻译:
        为该频道找一个阅读器。
        返回：
            频道的读者。"""
        return _ReceiveStream[T](queue=self._queue, done=self._done)
