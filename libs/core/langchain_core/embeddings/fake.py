"""Module contains a few fake embedding models for testing purposes.

中文翻译:
模块包含一些用于测试目的的假嵌入模型。"""

# Please do not add additional fake embedding model implementations here.
# 中文: 请不要在此处添加额外的假嵌入模型实现。
import contextlib
import hashlib

from pydantic import BaseModel
from typing_extensions import override

from langchain_core.embeddings import Embeddings

with contextlib.suppress(ImportError):
    import numpy as np


class FakeEmbeddings(Embeddings, BaseModel):
    """Fake embedding model for unit testing purposes.

    This embedding model creates embeddings by sampling from a normal distribution.

    !!! danger "Toy model"
        Do not use this outside of testing, as it is not a real embedding model.

    Instantiate:
        ```python
        from langchain_core.embeddings import FakeEmbeddings

        embed = FakeEmbeddings(size=100)
        ```

    Embed single text:
        ```python
        input_text = "The meaning of life is 42"
        vector = embed.embed_query(input_text)
        print(vector[:3])
        ```
        ```python
        [-0.700234640213188, -0.581266257710429, -1.1328482266445354]
        ```

    Embed multiple texts:
        ```python
        input_texts = ["Document 1...", "Document 2..."]
        vectors = embed.embed_documents(input_texts)
        print(len(vectors))
        # The first 3 coordinates for the first vector
        # 中文: 第一个向量的前 3 个坐标
        print(vectors[0][:3])
        ```
        ```python
        2
        [-0.5670477847544458, -0.31403828652395727, -0.5840547508955257]
        ```
    

    中文翻译:
    用于单元测试目的的假嵌入模型。
    该嵌入模型通过从正态分布中采样来创建嵌入。
    !!!危险“玩具模型”
        不要在测试之外使用它，因为它不是真正的嵌入模型。
    实例化：
        ````蟒蛇
        从 langchain_core.embeddings 导入 FakeEmbeddings
        嵌入 = FakeEmbeddings(大小=100)
        ````
    嵌入单个文本：
        ````蟒蛇
        input_text = "生命的意义是42"
        向量 = embed.embed_query(input_text)
        打印（向量[:3]）
        ````
        ````蟒蛇
        [-0.700234640213188，-0.581266257710429，-1.1328482266445354]
        ````
    嵌入多个文本：
        ````蟒蛇
        input_texts = ["文档 1...", "文档 2..."]
        向量 = embed.embed_documents(input_texts)
        打印（长度（向量））
        # 第一个向量的前 3 个坐标
        打印（向量[0][:3]）
        ````
        ````蟒蛇
        2
        [-0.5670477847544458，-0.31403828652395727，-0.5840547508955257]
        ````"""

    size: int
    """The size of the embedding vector.

    中文翻译:
    嵌入向量的大小。"""

    def _get_embedding(self) -> list[float]:
        return list(np.random.default_rng().normal(size=self.size))

    @override
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._get_embedding() for _ in texts]

    @override
    def embed_query(self, text: str) -> list[float]:
        return self._get_embedding()


class DeterministicFakeEmbedding(Embeddings, BaseModel):
    """Deterministic fake embedding model for unit testing purposes.

    This embedding model creates embeddings by sampling from a normal distribution
    with a seed based on the hash of the text.

    !!! danger "Toy model"
        Do not use this outside of testing, as it is not a real embedding model.

    Instantiate:
        ```python
        from langchain_core.embeddings import DeterministicFakeEmbedding

        embed = DeterministicFakeEmbedding(size=100)
        ```

    Embed single text:
        ```python
        input_text = "The meaning of life is 42"
        vector = embed.embed_query(input_text)
        print(vector[:3])
        ```
        ```python
        [-0.700234640213188, -0.581266257710429, -1.1328482266445354]
        ```

    Embed multiple texts:
        ```python
        input_texts = ["Document 1...", "Document 2..."]
        vectors = embed.embed_documents(input_texts)
        print(len(vectors))
        # The first 3 coordinates for the first vector
        # 中文: 第一个向量的前 3 个坐标
        print(vectors[0][:3])
        ```
        ```python
        2
        [-0.5670477847544458, -0.31403828652395727, -0.5840547508955257]
        ```
    

    中文翻译:
    用于单元测试目的的确定性假嵌入模型。
    该嵌入模型通过从正态分布中采样来创建嵌入
    带有基于文本哈希的种子。
    ！！！危险“玩具模型”
        不要在测试之外使用它，因为它不是真正的嵌入模型。
    实例化：
        ````蟒蛇
        从langchain_core.embeddings导入DeterministicFakeEmbedding
        嵌入=确定性假嵌入（大小= 100）
        ````
    嵌入单个文本：
        ````蟒蛇
        input_text = "生命的意义是42"
        向量 = embed.embed_query(input_text)
        打印（向量[:3]）
        ````
        ````蟒蛇
        [-0.700234640213188，-0.581266257710429，-1.1328482266445354]
        ````
    嵌入多个文本：
        ````蟒蛇
        input_texts = ["文档 1...", "文档 2..."]
        向量 = embed.embed_documents(input_texts)
        打印（长度（向量））
        # 第一个向量的前 3 个坐标
        打印（向量[0][:3]）
        ````
        ````蟒蛇
        2
        [-0.5670477847544458，-0.31403828652395727，-0.5840547508955257]
        ````"""

    size: int
    """The size of the embedding vector.

    中文翻译:
    嵌入向量的大小。"""

    def _get_embedding(self, seed: int) -> list[float]:
        # set the seed for the random generator
        # 中文: 设置随机生成器的种子
        rng = np.random.default_rng(seed)
        return list(rng.normal(size=self.size))

    @staticmethod
    def _get_seed(text: str) -> int:
        """Get a seed for the random generator, using the hash of the text.

        中文翻译:
        使用文本的哈希值获取随机生成器的种子。"""
        return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % 10**8

    @override
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._get_embedding(seed=self._get_seed(_)) for _ in texts]

    @override
    def embed_query(self, text: str) -> list[float]:
        return self._get_embedding(seed=self._get_seed(text))
