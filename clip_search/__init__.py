"""CLIP text-query search over the FAISS index produced by ``clip_index``.

Public API:
    - :func:`search_by_text`  : 单条文本查询 → top-k 目标元数据
    - :func:`batch_search`     : 批量文本查询, 复用同一份 CLIP 加载
    - :func:`compute_text_image_similarity`: 给定查询与图片路径列表, 返回余弦相似度

上游依赖: ``clip_index`` 已生成 ``processed/clip_index/`` 目录下的
``faiss.index`` 与 ``clip_metadata.json``。
"""

from .search_clip import (
    batch_search,
    compute_text_image_similarity,
    search_by_text,
)

__all__ = [
    "search_by_text",
    "batch_search",
    "compute_text_image_similarity",
]
