"""
LLM 问答模块 - 核心引擎
================================

职责:
    1. 加载 prompt 模板
    2. 接收 CLIP B 同学 ``clip_search.search_by_text`` 的检索结果
    3. 构造结构化 prompt (视频元信息 + 检索证据表格 + 用户问题)
    4. 调用 DeepSeek API (兼容 OpenAI 协议)
    5. 返回自然语言回答 (附带可被 Gradio 渲染的图像证据路径)

设计原则:
    - 防幻觉优先: 系统提示词硬性约束 + 检索为空时拒答
    - 失败兜底: 任何 API 异常均返回中文友好提示, 不向上抛出
    - 可注入: prompt 模板、模型名、温度等均可参数化
    - 字段容错: CLIP B 真实字段为 ``crop_path`` / ``class_name`` / ``score``;
      本模块同时兼容旧版 ``image_path`` / ``class`` / ``similarity`` 等命名

典型调用:
    >>> from llm_qa.qa_engine import CampusVisionQA
    >>> from clip_search import search_by_text
    >>>
    >>> results = search_by_text(query="视频中是否出现自行车?",
    ...                           index_dir="processed/clip_index",
    ...                           top_k=5)
    >>>
    >>> qa = CampusVisionQA.from_env()
    >>> result = qa.answer_question(
    ...     question="视频中是否出现自行车?",
    ...     retrieval_results=results,
    ...     video_meta={"video_name": "campus_demo.mp4",
    ...                 "duration": "00:45",
    ...                 "detection_count": len(results)},
    ... )
    >>> print(result.answer)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 第三方依赖按需导入, 允许在没有安装 openai 时仍可被 import
# (例如: 静态分析、文档生成场景)
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - 极少见
    load_dotenv = None  # type: ignore[assignment]

try:
    from openai import OpenAI
    from openai import APIError, APITimeoutError, RateLimitError
except ImportError:  # pragma: no cover - 极少见
    OpenAI = None  # type: ignore[assignment]
    APIError = APITimeoutError = RateLimitError = Exception  # type: ignore[assignment, misc]


logger = logging.getLogger(__name__)

# ---- 常量 ---------------------------------------------------------------

# 默认模板路径 (相对于本文件)
_DEFAULT_TEMPLATE_PATH = Path(__file__).parent / "prompt_template.txt"

# 当检索结果为空时, 不调用 API, 直接返回该中文固定回复
_EMPTY_RESULT_REPLY = "未在视频中检索到与问题相关的目标,无法回答。"

# 任何 API 异常时的兜底回复
_API_FALLBACK_REPLY = "服务暂时不可用,请稍后重试。"

# DeepSeek 官方 base_url (兼容 OpenAI 协议)
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 512
DEFAULT_TIMEOUT_S = 30.0
MAX_RETRIES = 2  # 含首次调用共 2 次


# ---- 数据类 -------------------------------------------------------------

@dataclass
class QAResult:
    """问答引擎的统一返回结构。

    Attributes:
        answer: 最终面向用户的自然语言回答 (中文)
        retrieved_count: 输入的检索证据条数
        raw_response: DeepSeek 原始回复文本 (供调试)
        model: 实际使用的模型名
    """

    answer: str
    retrieved_count: int
    raw_response: str = ""
    model: str = DEFAULT_MODEL
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "retrieved_count": self.retrieved_count,
            "raw_response": self.raw_response,
            "model": self.model,
            **self.extra,
        }


# ---- 核心引擎 -----------------------------------------------------------

class CampusVisionQA:
    """校园视频智能问答引擎。

    Args:
        api_key: DeepSeek API Key
        model: 模型名, 默认 ``deepseek-chat``
        base_url: 兼容 OpenAI 协议的服务地址
        template_path: prompt 模板文件路径
        temperature: 采样温度, 默认 0.2 (低温度, 利于减少幻觉)
        max_tokens: 单次回复最大 token
        timeout_s: API 超时秒数
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        template_path: str | os.PathLike[str] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        if OpenAI is None:
            raise ImportError(
                "缺少 openai 依赖, 请先执行: pip install openai>=1.0.0"
            )
        if not api_key:
            raise ValueError("api_key 不能为空")

        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s

        template_file = Path(template_path) if template_path else _DEFAULT_TEMPLATE_PATH
        self.template_text = self._load_template(template_file)
        self.system_prompt, self.user_template = self._parse_template(self.template_text)

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_s,
        )

    # ---------- 工厂方法 ----------

    @classmethod
    def from_env(
        cls,
        env_var: str = "DEEPSEEK_API_KEY",
        **kwargs: Any,
    ) -> "CampusVisionQA":
        """从环境变量 (或 .env 文件) 读取 API Key 并构造实例。

        若 ``llm_qa/.env`` 或工作目录 ``.env`` 存在, 会自动加载。
        """
        if load_dotenv is not None:
            # 优先加载 llm_qa/.env, 再退回到工作目录 .env
            env_path = Path(__file__).parent / ".env"
            if env_path.exists():
                load_dotenv(env_path)
            else:
                load_dotenv()
        api_key = os.environ.get(env_var, "").strip()
        if not api_key:
            raise ValueError(
                f"未找到环境变量 {env_var}, 请在 .env 中配置或显式传入 api_key"
            )
        return cls(api_key=api_key, **kwargs)

    # ---------- 模板与 prompt 构造 ----------

    @staticmethod
    def _load_template(path: Path) -> str:
        if not path.exists():
            raise FileNotFoundError(f"prompt 模板文件不存在: {path}")
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _parse_template(template_text: str) -> tuple[str, str]:
        """将模板按 ``[SYSTEM]`` / ``[USER]`` 标记拆分为 (system, user_template)。

        模板格式约定::

            [SYSTEM]
            ... 任意系统提示词 ...

            [USER]
            ... 用户消息模板,含 {video_name} 等占位符 ...

        标记必须出现在行首 (前面允许有空白) 才算有效, 避免与模板注释中的
        字符串意外匹配。
        ``[USER]`` 块可省略, 省略时 user_template 为空字符串。
        """
        import re

        sys_re = re.compile(r"^\s*\[SYSTEM\]\s*$", re.MULTILINE)
        user_re = re.compile(r"^\s*\[USER\]\s*$", re.MULTILINE)

        sys_match = sys_re.search(template_text)
        if not sys_match:
            # 兼容: 模板无 [SYSTEM] 标记时, 整段视为 user 模板
            return "", template_text.strip()

        after_sys = template_text[sys_match.end():]
        user_match = user_re.search(after_sys)
        if user_match:
            sys_part = after_sys[: user_match.start()]
            user_part = after_sys[user_match.end():]
            return sys_part.strip(), user_part.strip()
        return after_sys.strip(), ""

    # ---------- 字段名兼容辅助 ----------

    @staticmethod
    def _pick(r: dict[str, Any], *keys: str, default: Any = None) -> Any:
        """按顺序从 dict 中取第一个存在的 key, 全部缺失返回 default。"""
        for k in keys:
            if k in r and r[k] is not None:
                return r[k]
        return default

    def _row_dict(self, r: dict[str, Any]) -> dict[str, Any]:
        """将单条 CLIP B 检索结果规整为内部统一字段。

        真实 CLIP B 输出字段 (来自 ``clip_search/search_clip.py``):
            - rank: int
            - score: float (cosine similarity in [-1, 1])
            - id: int (FAISS vector id)
            - crop_path: str (目标裁剪图相对路径)
            - class_name: str (YOLO 类别名)
            - timestamp: str
            - confidence: float (YOLO 置信度, [0, 1])
            - frame_id: int
            - bbox: list[float]

        为兼容旧版 mock / 其他实现, 同时接受:
            - image_path ↔ crop_path
            - class / label ↔ class_name
            - similarity ↔ score
        """
        if not isinstance(r, dict):
            return {
                "rank": "-",
                "timestamp": "-",
                "class": "-",
                "confidence": None,
                "score": None,
                "crop_path": "-",
            }
        return {
            "rank": self._pick(r, "rank", default="-"),
            "timestamp": self._pick(r, "timestamp", default="-"),
            "class": self._pick(r, "class_name", "class", "label", default="-"),
            "confidence": self._pick(r, "confidence", default=None),
            "score": self._pick(r, "score", "similarity", default=None),
            "crop_path": self._pick(r, "crop_path", "image_path", default="-"),
        }

    def build_context_table(self, retrieval_results: list[dict[str, Any]]) -> str:
        """将 CLIP B 检索结果渲染为 Markdown 表格行。

        表格列: ``Rank | 时间戳 | 类别 | YOLO置信度 | CLIP相似度 (cosine) | 目标裁剪图路径``
        缺字段用 ``-`` 填充, 避免 prompt 中出现 KeyError。
        """
        if not retrieval_results:
            return ""
        rows: list[str] = []
        for raw in retrieval_results:
            r = self._row_dict(raw)
            conf = r["confidence"]
            score = r["score"]
            conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else "-"
            # cosine 相似度保留 4 位小数 (因为范围窄, 精度差异有意义)
            score_s = f"{score:.4f}" if isinstance(score, (int, float)) else "-"
            rows.append(
                f"| {r['rank']} | {r['timestamp']} | {r['class']} | "
                f"{conf_s} | {score_s} | {r['crop_path']} |"
            )
        return "\n".join(rows)

    def build_user_prompt(
        self,
        question: str,
        retrieval_results: list[dict[str, Any]],
        video_meta: dict[str, Any] | None = None,
    ) -> str:
        """构造 user 消息的完整正文 (含视频元信息 + 检索证据 + 用户问题)。"""
        meta = video_meta or {}
        context_table = self.build_context_table(retrieval_results)

        # 若模板本身不含占位符, 直接返回原模板
        if not self.user_template or "{" not in self.user_template:
            return self.user_template

        # 若模板中没有 {context_table} 占位符, 退化为 append 方式
        if "{context_table}" in self.user_template:
            return self.user_template.format(
                video_name=meta.get("video_name", "未命名视频"),
                duration=meta.get("duration", "-"),
                detection_count=meta.get("detection_count", "-"),
                context_table=context_table if context_table else "(无检索证据)",
                question=question.strip(),
            )
        return (
            f"{self.user_template}\n\n"
            f"视频: {meta.get('video_name', '未命名视频')}  |  "
            f"时长: {meta.get('duration', '-')}  |  "
            f"检测目标总数: {meta.get('detection_count', '-')}\n\n"
            f"【检索证据】(按相似度降序)\n"
            f"{context_table if context_table else '(无检索证据)'}\n\n"
            f"用户问题: {question.strip()}"
        )

    # ---------- 检索结果规整化 ----------

    @staticmethod
    def normalize_results(
        retrieval_results: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """对上游 CLIP B 的检索结果做轻量校验与字段补齐。

        保留 CLIP B 真实字段 (``crop_path`` / ``class_name`` / ``score`` 等) 原样,
        同时把旧版 ``image_path`` / ``class`` / ``similarity`` 等命名映射到
        标准字段, 供 prompt 表格渲染使用。
        """
        if not retrieval_results:
            return []
        normed: list[dict[str, Any]] = []
        for r in retrieval_results:
            if not isinstance(r, dict):
                continue
            crop_path = (
                r.get("crop_path")
                or r.get("image_path")
                or "-"
            )
            class_name = (
                r.get("class_name")
                or r.get("class")
                or r.get("label")
                or "-"
            )
            score = r.get("score")
            if score is None:
                score = r.get("similarity", 0.0)
            normed.append(
                {
                    # 真实 CLIP B 字段 (优先保留)
                    "rank": r.get("rank"),
                    "id": r.get("id"),
                    "frame_id": r.get("frame_id"),
                    "bbox": r.get("bbox"),
                    "crop_path": crop_path,
                    "class_name": class_name,
                    "timestamp": r.get("timestamp", "-"),
                    "confidence": r.get("confidence", 0.0),
                    "score": score,
                }
            )
        return normed

    # ---------- 主入口 ----------

    def answer_question(
        self,
        question: str,
        retrieval_results: list[dict[str, Any]] | None = None,
        video_meta: dict[str, Any] | None = None,
    ) -> QAResult:
        """生成对用户问题的回答。

        Args:
            question: 用户自然语言问题
            retrieval_results: CLIP B 检索结果 (list[dict])
            video_meta: 视频元信息 (dict, 可选)

        Returns:
            :class:`QAResult` 实例, 通过 ``.answer`` 获取回答文本
        """
        results = self.normalize_results(retrieval_results)

        # 1) 检索为空 → 不调用 API, 直接拒答 (节省 token, 强化防幻觉)
        if not results:
            logger.info("检索结果为空, 触发拒答分支")
            return QAResult(
                answer=_EMPTY_RESULT_REPLY,
                retrieved_count=0,
                model=self.model,
            )

        # 2) 构造 user prompt
        user_prompt = self.build_user_prompt(
            question=question,
            retrieval_results=results,
            video_meta=video_meta,
        )

        # 3) 调用 DeepSeek (带简单重试)
        raw_text = self._call_llm(user_prompt)

        if raw_text is None:
            # 调用失败, 已记录日志, 返回兜底
            return QAResult(
                answer=_API_FALLBACK_REPLY,
                retrieved_count=len(results),
                model=self.model,
            )

        # 4) 简单清理: 去掉首尾空白
        answer = raw_text.strip() or _EMPTY_RESULT_REPLY
        return QAResult(
            answer=answer,
            retrieved_count=len(results),
            raw_response=raw_text,
            model=self.model,
        )

    # ---------- DeepSeek 调用 ----------

    def _call_llm(self, user_prompt: str) -> str | None:
        """调用 DeepSeek Chat Completions 接口, 失败返回 None。"""
        last_err: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                messages: list[dict[str, str]] = []
                if self.system_prompt:
                    messages.append({"role": "system", "content": self.system_prompt})
                messages.append({"role": "user", "content": user_prompt})
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = (response.choices[0].message.content or "").strip()
                if not content:
                    logger.warning("DeepSeek 返回空内容, attempt=%d", attempt)
                    continue
                return content
            except RateLimitError as e:
                last_err = e
                logger.warning("触发限流, attempt=%d: %s", attempt, e)
                time.sleep(min(2 ** attempt, 5))
            except APITimeoutError as e:
                last_err = e
                logger.warning("调用超时, attempt=%d: %s", attempt, e)
                time.sleep(min(attempt, 2))
            except APIError as e:
                last_err = e
                logger.error("DeepSeek API 错误, attempt=%d: %s", attempt, e)
                # 非限流/超时类 API 错误, 重试意义不大, 直接退出
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.exception("调用 DeepSeek 出现未预期异常: %s", e)
                break
        if last_err is not None:
            logger.error("DeepSeek 调用最终失败: %s", last_err)
        return None


# ---- 便捷函数 (供 Gradio 同学直接 import) -------------------------------

_default_qa: CampusVisionQA | None = None


def get_default_qa() -> CampusVisionQA:
    """获取一个默认的 QA 引擎 (懒加载, 失败抛出明确异常)。"""
    global _default_qa
    if _default_qa is None:
        _default_qa = CampusVisionQA.from_env()
    return _default_qa


def answer_question(
    question: str,
    retrieval_results: list[dict[str, Any]] | None = None,
    video_meta: dict[str, Any] | None = None,
) -> str:
    """便捷函数: 直接返回回答文本, 供 Gradio 一行调用。

    用法::

        from llm_qa.qa_engine import answer_question
        text = answer_question("视频里有自行车吗?", retrieval_results, video_meta)
    """
    return get_default_qa().answer_question(
        question=question,
        retrieval_results=retrieval_results,
        video_meta=video_meta,
    ).answer


__all__ = [
    "CampusVisionQA",
    "QAResult",
    "answer_question",
    "get_default_qa",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
]
