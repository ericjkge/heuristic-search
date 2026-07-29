"""BM25 retrieval over the pooled MuSiQue paragraph corpus via Elasticsearch.

Documents are indexed with a single `contents` field ("Title"\\ntext), so
match queries score Lucene BM25 over title+text together.
"""

from __future__ import annotations

from elasticsearch import Elasticsearch

DEFAULT_URL = "http://localhost:9200"
INDEX = "musique"


class ESRetriever:
    def __init__(self, url: str = DEFAULT_URL, index: str = INDEX):
        self.es = Elasticsearch(url, request_timeout=30)
        self.index = index

    def search(self, query: str, topk: int = 3) -> list[dict[str, str]]:
        """Returns [{"title": ..., "text": ...}] for the top-k BM25 hits."""
        if not query.strip():
            return []
        resp = self.es.search(
            index=self.index,
            query={"match": {"contents": query}},
            size=topk,
            _source=["contents"],
        )
        out = []
        for hit in resp["hits"]["hits"]:
            contents = hit["_source"]["contents"]
            title, _, text = contents.partition("\n")
            out.append({"title": title.strip('" '), "text": text.strip()})
        return out

    def count(self) -> int:
        return self.es.count(index=self.index)["count"]
