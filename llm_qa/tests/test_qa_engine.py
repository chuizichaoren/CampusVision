"""
LLM 问答模块单元测试

覆盖场景:
1. 模板解析 ([SYSTEM] / [USER] 拆分, 行首正则匹配)
2. 检索为空 → 拒答 (不调用 API)
3. 检索正常 → mock DeepSeek 返回, 验证 prompt 内容与 answer
4. API 异常 → 兜底回复
5. 字段容错 (真实字段 crop_path/class_name/score + 旧别名 image_path/class/similarity)
6. 表格渲染 (含 Rank 列, cosine 相似度 4 位小数)
7. 便捷函数 answer_question
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# 确保可导入 llm_qa 与 clip_search
LLM_QA_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = LLM_QA_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_qa.qa_engine import (  # noqa: E402
    CampusVisionQA,
    QAResult,
    _EMPTY_RESULT_REPLY,
    _API_FALLBACK_REPLY,
    answer_question,
)


# ============== 测试夹具 ==============

@pytest.fixture
def template_path() -> Path:
    return LLM_QA_DIR / "prompt_template.txt"


@pytest.fixture
def sample_results() -> list[dict]:
    """CLIP B ``search_by_text`` 真实输出格式的样本。"""
    return [
        {
            "rank": 1,
            "score": 0.2814,             # cosine similarity in [-1, 1]
            "id": 412,
            "crop_path": "processed/object_crops/bicycle_0007.jpg",
            "class_name": "bicycle",
            "timestamp": "00:21",
            "confidence": 0.81,
            "frame_id": 251,
            "bbox": [120, 240, 360, 480],
        },
        {
            "rank": 2,
            "score": 0.2541,
            "id": 738,
            "crop_path": "processed/object_crops/bicycle_0011.jpg",
            "class_name": "bicycle",
            "timestamp": "00:38",
            "confidence": 0.74,
            "frame_id": 456,
            "bbox": [80, 200, 320, 500],
        },
    ]


@pytest.fixture
def video_meta() -> dict:
    return {
        "video_name": "campus_demo.mp4",
        "duration": "00:45",
        "detection_count": 23,
    }


@pytest.fixture
def qa(template_path) -> CampusVisionQA:
    """构造一个不连接真实 API 的 QA 实例 (openai 客户端被 mock)。"""
    with patch("llm_qa.qa_engine.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        return CampusVisionQA(api_key="sk-test-fake-key", template_path=template_path)


# ============== 模板解析测试 ==============

def test_parse_template_split(qa: CampusVisionQA) -> None:
    """模板应正确拆分为 (system, user_template)。"""
    assert qa.system_prompt, "system 提示词不应为空"
    assert "智瞳" in qa.system_prompt
    assert "严禁编造" in qa.system_prompt
    assert "{question}" in qa.user_template
    assert "{context_table}" in qa.user_template


def test_parse_template_without_user_marker() -> None:
    """模板无 [USER] 标记时, user_template 应为空字符串。"""
    sys_part, user_part = CampusVisionQA._parse_template("[SYSTEM]\n只包含系统")
    assert "只包含系统" in sys_part
    assert user_part == ""


def test_parse_template_without_system_marker() -> None:
    """模板无 [SYSTEM] 标记时, 整段视为 user 模板。"""
    sys_part, user_part = CampusVisionQA._parse_template("整段都是 user 内容 {x}")
    assert sys_part == ""
    assert "{x}" in user_part


# ============== 上下文表格渲染测试 ==============

def test_build_context_table_normal(qa: CampusVisionQA, sample_results) -> None:
    table = qa.build_context_table(sample_results)
    # 真实字段都应出现
    assert "bicycle" in table
    assert "00:21" in table
    assert "00:38" in table
    assert "0.81" in table
    assert "bicycle_0007.jpg" in table
    # Rank 列
    assert "| 1 |" in table
    assert "| 2 |" in table
    # cosine 相似度保留 4 位小数
    assert "0.2814" in table
    # Markdown 表格行数 = 检索条数
    assert table.count("\n") == len(sample_results) - 1


def test_build_context_table_empty(qa: CampusVisionQA) -> None:
    assert qa.build_context_table([]) == ""


def test_build_context_table_missing_fields(qa: CampusVisionQA) -> None:
    """缺字段时应填充 '-', 不抛 KeyError。"""
    table = qa.build_context_table([{"class_name": "person"}])
    assert "person" in table
    assert " - " in table or "|-|" in table or "|" in table


def test_build_context_table_with_legacy_alias(qa: CampusVisionQA) -> None:
    """旧版字段名 (image_path / class / similarity) 也能正确渲染。"""
    legacy = [
        {"image_path": "x.jpg", "class": "car", "confidence": 0.9,
         "timestamp": "00:01", "similarity": 0.42},
    ]
    table = qa.build_context_table(legacy)
    assert "car" in table
    assert "x.jpg" in table
    assert "0.42" in table


# ============== normalize_results 测试 ==============

def test_normalize_results_keeps_real_fields() -> None:
    """真实字段 crop_path / class_name / score 应原样保留。"""
    raw = [
        {"rank": 1, "score": 0.3, "id": 5,
         "crop_path": "a.jpg", "class_name": "person",
         "timestamp": "00:01", "confidence": 0.9,
         "frame_id": 100, "bbox": [1, 2, 3, 4]},
    ]
    norm = CampusVisionQA.normalize_results(raw)
    assert len(norm) == 1
    n = norm[0]
    assert n["crop_path"] == "a.jpg"
    assert n["class_name"] == "person"
    assert n["score"] == 0.3
    assert n["rank"] == 1
    assert n["id"] == 5
    assert n["frame_id"] == 100
    assert n["bbox"] == [1, 2, 3, 4]
    assert n["timestamp"] == "00:01"
    assert n["confidence"] == 0.9


def test_normalize_results_with_alias() -> None:
    """'class' / 'label' 应回退到 'class_name'; 'image_path' 应回退到 'crop_path'。"""
    raw = [
        {"image_path": "a.jpg", "label": "car", "confidence": 0.9,
         "timestamp": "00:01", "similarity": 0.5},
    ]
    norm = CampusVisionQA.normalize_results(raw)
    assert norm[0]["crop_path"] == "a.jpg"
    assert norm[0]["class_name"] == "car"
    assert norm[0]["score"] == 0.5


def test_normalize_results_drops_non_dict() -> None:
    raw = [{"class_name": "ok"}, "not-a-dict", None, 123]
    norm = CampusVisionQA.normalize_results(raw)
    assert len(norm) == 1


def test_normalize_results_empty() -> None:
    assert CampusVisionQA.normalize_results(None) == []
    assert CampusVisionQA.normalize_results([]) == []


# ============== 主流程: 检索为空 ==============

def test_answer_question_empty_results_skips_api(
    qa: CampusVisionQA, video_meta
) -> None:
    """检索为空时, 不应调用 DeepSeek, 直接返回固定拒答。"""
    with patch.object(qa._client.chat.completions, "create") as mock_create:
        result = qa.answer_question(
            question="视频中有没有狗?",
            retrieval_results=[],
            video_meta=video_meta,
        )
        assert isinstance(result, QAResult)
        assert result.answer == _EMPTY_RESULT_REPLY
        assert result.retrieved_count == 0
        assert mock_create.call_count == 0  # 关键: 没有调用 API


# ============== 主流程: 检索正常 (mock DeepSeek) ==============

def test_answer_question_normal_calls_api_and_returns_answer(
    qa: CampusVisionQA, sample_results, video_meta
) -> None:
    """正常检索 → 调用 DeepSeek → 返回模型答案。"""
    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = "视频中检测到 2 辆自行车。"

    with patch.object(qa._client.chat.completions, "create",
                      return_value=fake_response) as mock_create:
        result = qa.answer_question(
            question="视频中是否出现自行车?",
            retrieval_results=sample_results,
            video_meta=video_meta,
        )

    assert result.answer == "视频中检测到 2 辆自行车。"
    assert result.retrieved_count == 2
    assert result.raw_response == "视频中检测到 2 辆自行车。"

    # 验证 API 被以正确的 system/user 消息调用
    mock_create.assert_called_once()
    call_kwargs = mock_create.call_args.kwargs
    messages = call_kwargs["messages"]
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user"]
    # system 应包含防幻觉规则关键词
    assert "严禁编造" in messages[0]["content"]
    # user 应包含用户问题与检索证据
    assert "视频中是否出现自行车" in messages[1]["content"]
    assert "bicycle" in messages[1]["content"]
    # prompt 表格应含 Rank 列
    assert "| 1 |" in messages[1]["content"]


# ============== 主流程: API 异常兜底 ==============

def test_answer_question_api_error_falls_back(
    qa: CampusVisionQA, sample_results, video_meta
) -> None:
    """API 抛异常时, 应返回兜底中文回复, 不向上抛。"""
    from openai import APIError

    with patch.object(qa._client.chat.completions, "create",
                      side_effect=APIError("server down", request=MagicMock(), body=None)):
        result = qa.answer_question(
            question="视频中是否出现自行车?",
            retrieval_results=sample_results,
            video_meta=video_meta,
        )

    assert result.answer == _API_FALLBACK_REPLY
    assert result.retrieved_count == 2  # 仍记录检索条数, 便于调试


def test_answer_question_timeout_falls_back(
    qa: CampusVisionQA, sample_results, video_meta
) -> None:
    """APITimeoutError 也应兜底。"""
    from openai import APITimeoutError

    with patch.object(qa._client.chat.completions, "create",
                      side_effect=APITimeoutError("timeout")):
        result = qa.answer_question(
            question="视频中是否出现自行车?",
            retrieval_results=sample_results,
            video_meta=video_meta,
        )

    assert result.answer == _API_FALLBACK_REPLY


# ============== 主流程: 字段容错 ==============

def test_answer_question_with_missing_fields_still_calls_api(
    qa: CampusVisionQA, video_meta
) -> None:
    """检索条目缺字段时, 不应崩溃, 仍能调用 API。"""
    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = "已回答。"

    messy_results = [
        {"class_name": "person"},   # 缺 crop_path, timestamp, score
        {"label": "car", "image_path": "y.jpg", "similarity": 0.3},  # 字段名差异
    ]

    with patch.object(qa._client.chat.completions, "create",
                      return_value=fake_response) as mock_create:
        result = qa.answer_question(
            question="test", retrieval_results=messy_results, video_meta=video_meta
        )

    assert result.answer == "已回答。"
    assert mock_create.call_count == 1


# ============== 便捷函数测试 ==============

def test_answer_question_helper_uses_default_qa(sample_results) -> None:
    """便捷函数应通过 get_default_qa 工作。"""
    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = "通过便捷函数回答。"

    with patch("llm_qa.qa_engine.OpenAI") as mock_openai_cls, \
         patch.dict("os.environ", {"DEEPSEEK_API_KEY": "sk-test"}):
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = fake_response
        mock_openai_cls.return_value = mock_client

        # 重置单例
        import llm_qa.qa_engine as engine_mod
        engine_mod._default_qa = None

        try:
            ans = answer_question(
                question="测试",
                retrieval_results=sample_results,
                video_meta={"video_name": "v.mp4", "duration": "00:10",
                            "detection_count": 5},
            )
        finally:
            engine_mod._default_qa = None

    assert ans == "通过便捷函数回答。"


# ============== 构造函数校验 ==============

def test_init_requires_api_key(template_path) -> None:
    with patch("llm_qa.qa_engine.OpenAI"):
        with pytest.raises(ValueError, match="api_key 不能为空"):
            CampusVisionQA(api_key="", template_path=template_path)


def test_init_requires_openai_dependency(template_path) -> None:
    with patch("llm_qa.qa_engine.OpenAI", None):
        with pytest.raises(ImportError, match="openai"):
            CampusVisionQA(api_key="sk-x", template_path=template_path)


# ============== 入口 ==============

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
