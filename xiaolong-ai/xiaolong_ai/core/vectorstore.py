# -*- coding: utf-8 -*-

# 小龙智脑 (XiaoLong Brain) - 全新原创项目

# 作者 / 版权人: 小龙 (XiaoLong)

# License: MIT。本项目所有代码均为原创，保留署名即可自由使用。



import math





def cosine(a, b):

    dot = sum(x * y for x, y in zip(a, b))

    na = math.sqrt(sum(x * x for x in a))

    nb = math.sqrt(sum(x * x for x in b))

    if na == 0 or nb == 0:

        return 0.0

    return dot / (na * nb)





class InMemoryVectorStore:

    def __init__(self, embedding_fn):

        self.embedding_fn = embedding_fn

        self.items = []



    def add(self, text, metadata=None):

        vec = self.embedding_fn.embed(text)

        self.items.append({"text": text, "vec": vec, "metadata": metadata or {}})



    def add_documents(self, docs):

        for d in docs:

            self.add(d["text"], d.get("metadata"))



    def search(self, query, top_k=3):

        qv = self.embedding_fn.embed(query)

        scored = [(cosine(qv, it["vec"]), it) for it in self.items]

        scored.sort(key=lambda x: x[0], reverse=True)

        return [{"score": s, "text": it["text"], "metadata": it["metadata"]}

                for s, it in scored[:top_k]]

