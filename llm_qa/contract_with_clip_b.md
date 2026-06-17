# LLM 问答模块 ↔ CLIP B 文本检索模块 接口契约 (v1.0)

> 状态: **已对齐真实实现** — 字段与函数名均直接来自
> `项目代码/CampusVision/clip_search/search_clip.py`
> 作者: LLM 问答模块负责人
> 适用版本: 智瞳 CampusVision v1 (课设终稿)

## 1. 背景

LLM 问答模块 (`llm_qa/qa_engine.py`) 的输入是 CLIP B 模块
(`clip_search/search_clip.py`) 的输出。
本文件约定该接口, 确保两个模块可独立开发、最后联调。

## 2. CLIP B → LLM 问答 的数据流

```
用户问题
   │
   ▼
search_by_text(query, index_dir, top_k, ...)
   │
   ▼
list[dict]   (字段: rank, score, id, crop_path, class_name,
              timestamp, confidence, frame_id, bbox)
   │
   ▼
qa_engine.answer_question(question, retrieval_results, video_meta)
   │
   ▼
DeepSeek API
   │
   ▼
自然语言回答 (中文, 基于证据)
```

## 3. 接口约定 (v1.0, 已对齐)

### 3.1 CLIP B 模块对外函数

```python
# 项目代码/CampusVision/clip_search/search_clip.py
from clip_search import search_by_text

results: list[dict] = search_by_text(
    query="视频中是否出现自行车?",
    index_dir="processed/clip_index",   # CLIP A 输出目录
    top_k=5,
    model_name="openai/clip-vit-base-patch32",   # 可选, 默认
    device=None,                                # 可选, 自动选 cuda/cpu
)
```

#### 入参

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `query` | `str` | (必填) | 自然语言查询, 如 "a person walking" |
| `index_dir` | `str` | `"processed/clip_index"` | 包含 `faiss.index` 与 `clip_metadata.json` |
| `top_k` | `int` | `5` | 返回条数, 必须 > 0 |
| `model_name` | `str` | `"openai/clip-vit-base-patch32"` | 必须与 CLIP A 一致 |
| `device` | `str \| None` | `None` | `"cuda"` / `"cpu"`, None 时自动选 |

#### 出参 (每条 dict 字段)

字段名大小写敏感, 严格按 CLIP B 真实实现:

| 字段 | 类型 | 范围 | 说明 |
|---|---|---|---|
| `rank` | `int` | 1-based | FAISS 排名 |
| `score` | `float` | `[-1, 1]` | **余弦相似度** (cosine), 越大越相关 |
| `id` | `int` | ≥ 0 | FAISS vector 索引号 |
| `crop_path` | `str` | — | 目标裁剪图**相对路径**, 供前端 `gr.Image` 渲染 |
| `class_name` | `str` | — | YOLOv8 类别英文名, 如 `person` / `bicycle` / `car` |
| `timestamp` | `str` | — | 视频内时间戳, 格式 `mm:ss` 或 `hh:mm:ss` |
| `confidence` | `float` | `[0, 1]` | YOLOv8 检测置信度 (4 位小数) |
| `frame_id` | `int` | ≥ 0 | 帧编号 |
| `bbox` | `list[float]` | — | 边界框 `[x1, y1, x2, y2]`, LLM 暂不使用 |

### 3.2 兜底约定 (CLIP B 真实行为)

- **检索成功但无结果** → 返回 `[]` (空列表, 不抛异常)
- **FAISS 索引未构建 / `index_dir` 不存在** → 抛 `FileNotFoundError`
- **`top_k <= 0`** → 抛 `ValueError`
- **CLIP 模型加载失败** → 抛运行时异常 (由 transformers / huggingface_hub 抛出)
- **用户问题为空字符串** → 仍会调用 CLIP 编码空串, 建议上层拦截空查询

## 4. LLM 问答侧兼容性

`qa_engine.py` 中 `normalize_results()` 已经做了字段名兼容, 具体规则:

- `crop_path` ↔ `image_path` (旧版 mock 字段)
- `class_name` ↔ `class` ↔ `label`
- `score` (cosine, [-1, 1]) ↔ `similarity` (旧版)
- `rank` / `id` / `frame_id` / `bbox` 原样保留
- 缺字段值用 `0.0` 或 `"-"` 填充, 不会抛 `KeyError`
- 非 `dict` 类型条目会被静默跳过

因此 CLIP B 同学即使后续微调字段名, LLM 问答侧仍能容忍。

## 5. 端到端最小示例

```python
# 1) CLIP B 检索
from clip_search import search_by_text
results = search_by_text(
    query="视频中是否出现自行车?",
    index_dir="processed/clip_index",
    top_k=5,
)

# 2) LLM 问答
from llm_qa.qa_engine import answer_question
text = answer_question(
    question="视频中是否出现自行车?",
    retrieval_results=results,
    video_meta={
        "video_name": "campus_demo.mp4",
        "duration": "00:45",
        "detection_count": len(results),
    },
)
print(text)

# 3) 前端同时渲染证据图
image_paths = [r["crop_path"] for r in results]
# gr.Gallery(value=image_paths).render()
```

## 6. 上下游依赖图

```
[YOLO 同学]                  [CLIP A 同学]               [CLIP B 同学]              [LLM 问答 (我)]
processed/object_crops/  ──►  CLIP 编码             ──►  faiss.index             ──►  qa_engine
processed/detections.csv ──►  image_embeddings.npy ──►  clip_metadata.json      ──►  DeepSeek API
                            ──►  faiss.index          ──►  search_by_text()              │
                            ──►  clip_metadata.json                                         ▼
                                                                                      中文回答
```

## 7. 待确认事项 (OPEN QUESTIONS)

- [x] ~~`search_clip.search()` 的函数名~~ — 确认为 `search_by_text`
- [x] ~~返回列表是否要按 `score` 降序~~ — 是, FAISS 天然按相似度降序
- [x] ~~`timestamp` 格式~~ — 字符串 `"mm:ss"` 或 `"hh:mm:ss"`
- [x] ~~`image_path` vs `crop_path`~~ — 统一为 `crop_path`
- [x] ~~`class` vs `class_name`~~ — 统一为 `class_name`
- [x] ~~`score` 范围~~ — `[-1, 1]` cosine similarity

> 所有契约项已与 `clip_search/search_clip.py` 实际实现对齐, 无未决事项。
