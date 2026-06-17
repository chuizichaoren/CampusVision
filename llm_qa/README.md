# LLM 问答模块 (智瞳 CampusVision)

> 模块负责: 大语言模型问答
> 配套项目: 智瞳 CampusVision - 基于 YOLO-CLIP 的校园场景视频目标检索与智能问答系统
> 后端: DeepSeek API (兼容 OpenAI 协议)
> 上游: `clip_search.search_by_text()` (CLIP B 同学)
> 下游: Gradio 前端 (`main.py`)

## 目录结构

```
llm_qa/
├── qa_engine.py            # 核心引擎 (CampusVisionQA 类 + answer_question 便捷函数)
├── prompt_template.txt     # prompt 模板 (含 [SYSTEM] 防幻觉规则 + [USER] 占位符)
├── demo_questions.md       # 10 个示例问题与预期回答 (5 命中 + 5 未命中)
├── contract_with_clip_b.md # 与 CLIP B 模块的接口契约 (已对齐真实字段)
├── README.md               # 本文件
├── .env.example            # 环境变量样例
└── tests/
    └── test_qa_engine.py   # 单元测试 (mock DeepSeek 响应, 17 个测试)
```

## 安装依赖

新增以下依赖到项目根 `requirements.txt` (已由本 PR 追加):

```
openai>=1.0.0
python-dotenv>=1.0.0
```

安装命令:

```bash
pip install openai>=1.0.0 python-dotenv>=1.0.0
```

## 配置

1. 复制环境变量样例:

   ```bash
   cp llm_qa/.env.example llm_qa/.env
   ```

2. 编辑 `llm_qa/.env`, 填入真实 API Key:

   ```text
   DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

   > 也可以放到项目根 `.env` 文件, 或直接设为系统环境变量。

3. (可选) 自定义模型与 base_url:

   ```python
   from llm_qa.qa_engine import CampusVisionQA
   qa = CampusVisionQA(
       api_key="sk-xxx",
       model="deepseek-chat",          # 或 deepseek-reasoner
       base_url="https://api.deepseek.com",
       temperature=0.2,
       max_tokens=512,
   )
   ```

## 快速开始

```python
from clip_search import search_by_text
from llm_qa.qa_engine import answer_question

# 1) CLIP B 检索
results = search_by_text(
    query="a bicycle on campus",
    index_dir="processed/clip_index",
    top_k=5,
)

# 2) LLM 问答
text = answer_question(
    question="视频中是否出现自行车?",
    retrieval_results=results,
    video_meta={"video_name": "campus_demo.mp4",
                "duration": "00:45",
                "detection_count": len(results)},
)
print(text)
```

## 在 Gradio 中调用

```python
import gradio as gr
from llm_qa.qa_engine import CampusVisionQA
from clip_search import search_by_text
from pathlib import Path

qa = CampusVisionQA.from_env()

def on_user_question(question, video_state):
    # 1) CLIP B 检索
    results = search_by_text(
        query=question,
        index_dir="processed/clip_index",
        top_k=5,
    )
    # 2) 组装 video_meta (默认从 video_state 推断)
    if isinstance(video_state, dict):
        video_name = Path(video_state.get("video_path", "uploaded.mp4")).name
        duration = video_state.get("duration", "-")
    else:
        video_name, duration = "uploaded.mp4", "-"

    # 3) LLM 问答
    answer = qa.answer_question(
        question=question,
        retrieval_results=results,
        video_meta={"video_name": video_name,
                    "duration": duration,
                    "detection_count": len(results)},
    )
    # 4) 返回: 文本 + 目标裁剪图路径列表 (Gradio 用 gr.Gallery 渲染)
    image_paths = [r["crop_path"] for r in results]
    return answer.answer, image_paths
```

> Gradio 集成示例的完整注释已附在 `main.py` 文件末尾, 可直接参考。

## 输入字段约定 (与 CLIP B 真实输出一致)

| 字段 | 类型 | 范围 | 说明 |
|---|---|---|---|
| `crop_path` | `str` | — | 目标裁剪图本地路径, 前端可直接 `gr.Image` 渲染 |
| `class_name` | `str` | — | YOLOv8 类别英文名, 如 `person` / `bicycle` / `car` |
| `confidence` | `float` | `[0, 1]` | YOLOv8 检测置信度 |
| `timestamp` | `str` | — | 视频内时间戳, 格式 `mm:ss` 或 `hh:mm:ss` |
| `score` | `float` | `[-1, 1]` | **cosine 相似度**, 越大越相关 |
| `rank` | `int` | 1-based | FAISS 排名 |
| `id` | `int` | ≥ 0 | FAISS vector 索引号 |
| `frame_id` | `int` | ≥ 0 | 帧编号 |
| `bbox` | `list[float]` | — | 边界框 `[x1, y1, x2, y2]` (LLM 暂不使用) |

为兼容旧实现, `qa_engine.normalize_results()` 也会接受以下别名:

- `crop_path` ↔ `image_path`
- `class_name` ↔ `class` ↔ `label`
- `score` ↔ `similarity`

## 防幻觉机制

| 层次 | 措施 | 效果 |
|---|---|---|
| 1. 系统提示词 | `prompt_template.txt` 硬性约束, 严禁编造 | 主防线 |
| 2. 检索为空 | 跳过 API, 返回固定中文拒答 | 节省 token, 杜绝"假装检索过" |
| 3. 上下文表格化 | 检索结果渲染为结构化 Markdown 表格 | 模型引用更稳定 |
| 4. 低温度采样 | `temperature=0.2` | 减少随机性 |
| 5. API 失败兜底 | 异常时返回"服务暂时不可用", 不抛错 | 前端不会崩溃 |
| 6. 字段容错 | `normalize_results()` 兼容字段名差异 | 与 CLIP B 联调时不易崩 |

## 单元测试

```bash
cd /home/bertha/study/class_res/课程笔记库/软件过程与管理/小组项目/项目代码/CampusVision
python -m pytest llm_qa/tests/ -v
```

测试覆盖 (17 个):
- 模板解析 (`[SYSTEM]` / `[USER]` 拆分, 行首正则匹配)
- 上下文表格渲染 (含 `Rank` 列)
- 字段容错 (`crop_path`/`class_name`/`score` 真实字段 + 旧别名)
- 检索为空 → 拒答 (不调用 API)
- 检索正常 → mock DeepSeek 返回, 验证 prompt 构造
- API 异常 (`APIError` / `APITimeoutError`) → 兜底回复
- 便捷函数 `answer_question`

## 依赖注入与扩展

- 替换模型: 传入 `model="deepseek-reasoner"` 或其他 OpenAI 兼容服务
- 替换 prompt: 传入 `template_path="my_template.txt"`, 只需保留 `[SYSTEM]` 与 `[USER]` 行首标记
- 替换 base_url: 兼容其他 OpenAI 协议服务 (Moonshot / Together / Groq 等)

## 已知限制

- 单轮对话, 无上下文记忆
- 不做回答缓存
- 不做 prompt 自动调优
- 不做答案评估 (仅依赖 prompt 工程)

## 维护者

- LLM 问答模块负责人
- 协作方: CLIP A/B 同学 (检索输入) / Gradio 同学 (前端渲染) / YOLO 同学 (元数据)
