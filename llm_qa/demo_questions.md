# Demo 示例问题 (10 条)

> 用途: 答辩演示与系统测试。
> 编排: 5 条"命中类"(检索有结果 → 正常回答) + 5 条"未命中/部分命中类"(用于演示拒答与防幻觉能力)。
> 字段名约定: 全部使用 `clip_search.search_by_text()` 真实输出格式
> (`crop_path` / `class_name` / `score` cosine in [-1, 1] / `confidence` / `timestamp` / `rank`)。

## 使用方法

```python
from llm_qa.qa_engine import CampusVisionQA
from clip_search import search_by_text

qa = CampusVisionQA.from_env()

# 1) 命中类 (真实检索)
results = search_by_text(query="a bicycle on campus",
                         index_dir="processed/clip_index", top_k=5)
result = qa.answer_question(
    question="视频中是否出现自行车?",
    retrieval_results=results,
    video_meta={"video_name": "campus_demo.mp4",
                "duration": "00:45", "detection_count": len(results)},
)
print(result.answer)

# 2) 未命中类 (检索为空, 演示拒答)
result = qa.answer_question(
    question="视频中有没有狗或猫等动物?",
    retrieval_results=[],
    video_meta={"video_name": "campus_demo.mp4",
                "duration": "00:45", "detection_count": 0},
)
print(result.answer)  # 输出: "未在视频中检索到与问题相关的目标,无法回答。"
```

---

## A. 命中类 (5 条)

### Q1. 自行车 (bicycle)

- **问题**: 视频中是否出现自行车?出现在什么时间?
- **假设检索结果** (top-3, 使用 `search_by_text` 真实字段):

  ```python
  [
      {"rank": 1, "score": 0.2814, "id": 412,
       "crop_path": "processed/object_crops/bicycle_0007.jpg",
       "class_name": "bicycle", "timestamp": "00:21",
       "confidence": 0.81, "frame_id": 251, "bbox": [120, 240, 360, 480]},
      {"rank": 2, "score": 0.2541, "id": 738,
       "crop_path": "processed/object_crops/bicycle_0011.jpg",
       "class_name": "bicycle", "timestamp": "00:38",
       "confidence": 0.74, "frame_id": 456, "bbox": [80, 200, 320, 500]},
      {"rank": 3, "score": 0.2287, "id": 901,
       "crop_path": "processed/object_crops/bicycle_0015.jpg",
       "class_name": "bicycle", "timestamp": "00:52",
       "confidence": 0.68, "frame_id": 624, "bbox": [150, 220, 380, 470]},
  ]
  ```
- **预期回答方向**: 视频中检测到 3 辆自行车 (Rank 1~3), 分别出现在 00:21、00:38、00:52,
  YOLO 置信度均高于 0.65, CLIP 相似度在 0.22~0.28 之间。
  建议前端展示这 3 张目标图 (按 `crop_path` 渲染)。
- **答辩亮点**: 时间戳证据 + 置信度与 cosine 相似度透明展示。

### Q2. 行人 (person)

- **问题**: 有没有行人在校园道路上行走?
- **假设检索结果** (top-5, 略): class_name=`person` 的目标
- **预期回答方向**: 视频中出现 N 个 `person` 目标, 主要时间集中在 00:0X~00:XX。
  引用前 2~3 个 Rank 最高的目标。
- **答辩亮点**: 多目标归纳, 但仍基于证据。

### Q3. 实验室电脑/显示器 (laptop/monitor)

- **问题**: 实验室里能看到显示器或电脑吗?
- **假设检索结果**: class_name=`laptop` / `monitor` 各 2 条左右
- **预期回答方向**: 检测到 N 台笔记本电脑 + M 台显示器, 出现在 00:XX 附近。
- **答辩亮点**: 同一语义下多类别合并回答。

### Q4. 车辆汇总 (car/bus/truck)

- **问题**: 视频里检测到了哪些车辆?分别在什么时间?
- **假设检索结果**: 混合 `car`/`bus`/`motorcycle` 多条
- **预期回答方向**: 列出 3 类车辆, 每类给出首次出现的 Rank 与时间戳。
  **不应**编造未出现的车型。
- **答辩亮点**: 类别多样性 + 严谨性。

### Q5. 背包 (backpack)

- **问题**: 教学楼附近有没有人背着包?
- **假设检索结果**: class_name=`backpack` 类别 + class_name=`person` 类别混合
- **预期回答方向**: 检索到 N 个 `backpack` 目标, 同时附带出现该目标的 `person` 时间戳。
  注意: 模型不应主观推断"人背着包"——只能基于检索证据。
- **答辩亮点**: 跨类别关联的边界说明。

---

## B. 未命中/部分命中类 (5 条, 演示拒答能力)

### Q6. 动物 (YOLO COCO 一般无 dog/cat)

- **问题**: 视频中有没有狗或猫等动物?
- **假设检索结果**: `[]` (空)
- **预期回答**: **"未在视频中检索到与问题相关的目标,无法回答。"** (固定拒答, LLM 不会调用)
- **答辩亮点**: 检索为空 → 走兜底分支, 节省 token, 演示明确拒答。

### Q7. 摔倒检测 (复杂语义)

- **问题**: 视频里有没有人摔倒?
- **假设检索结果**: 仅返回 class_name=`person` (无"摔倒"标签)
- **预期回答方向**: 检测到 N 个 `person` 目标, 但**当前系统无法基于检索证据判断是否摔倒**,
  因为 YOLOv8 不输出姿态/动作标签。
- **答辩亮点**: 主动暴露系统能力边界, 不强行编造。

### Q8. 行人计数 (去重问题)

- **问题**: 请统计视频中共出现多少个不同的人?
- **假设检索结果**: 10 条 class_name=`person`, 时间戳跨度 00:00~00:45
- **预期回答方向**: 检测到 10 个人形目标, 但**当前系统无法确认其中是否有重复个体**
  (缺少行人重识别/跟踪模块), 因此无法给出"不同人数"。
- **答辩亮点**: 客观说明 YOLO 输出粒度, 不强答。

### Q9. 天气/日期 (超出能力)

- **问题**: 这段视频拍摄于哪一天、什么天气?
- **假设检索结果**: 仅常规目标
- **预期回答方向**: **当前系统无法基于检索证据回答该问题**。
  系统仅提供目标检测与检索, 不解析元数据/环境。
- **答辩亮点**: 明确拒绝超出能力范围的问题。

### Q10. 语音内容 (无音频模态)

- **问题**: 视频中的人在说什么?
- **假设检索结果**: 常规 `person` 目标
- **预期回答方向**: **当前系统无法基于检索证据回答该问题**。
  本系统仅基于视觉目标检索, 不含 ASR 模块。
- **答辩亮点**: 区分视觉/听觉模态边界。

---

## C. 演示流程建议

1. **准备视频**: 选 1 段 30~60 秒的校园场景视频, 确保 YOLO 能检测出多种目标。
2. **依次输入 5 条命中类问题**: 展示 LLM 给出有依据的回答。
3. **依次输入 5 条未命中类问题**:
   - Q6 检索为空 → 展示"未检索到"固定回复
   - Q7/Q8 展示"无法判断"边界
   - Q9/Q10 展示"超出能力"拒答
4. **现场对比**: 关闭本模块, 直接让 DeepSeek 看视频元信息问答 → 展示幻觉;
   再开启本模块 → 展示防幻觉。
