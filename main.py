import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")

import gradio as gr
import tempfile
import shutil
import logging
import csv
from datetime import datetime
from PIL import Image, ImageDraw

# Add logging for device detection and imports
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FRONTEND_USE_MOCK = os.getenv("VIDEO_RAG_USE_MOCK", "1") == "1"

# Optional backend dependencies. The Gradio frontend can run in mock mode before
# YOLO/CLIP/LLM modules and heavy model dependencies are ready.
try:
    import cv2
except ImportError:
    cv2 = None

try:
    import whisper
except ImportError:
    whisper = None

try:
    import torch
except ImportError:
    torch = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    import faiss
except ImportError:
    faiss = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from moviepy.editor import VideoFileClip
except ImportError:
    VideoFileClip = None

try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    CLIPProcessor = None
    CLIPModel = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# Comment out the path setting to use system PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Comment out FFMPEG path to use default
# change_settings({"FFMPEG_BINARY": r"C:\Program Files\FFmpeg\bin\ffmpeg.exe"})

device = "cpu"
whisper_model = None
yolo_model = None
clip_model = None
clip_processor = None
text_model = None

try:
    if FRONTEND_USE_MOCK:
        logging.info("Frontend mock mode enabled; skipping heavy backend model loading.")
    elif all([torch, whisper, YOLO, CLIPModel, CLIPProcessor, SentenceTransformer]):
        if torch.cuda.is_available():
            device = "cuda"
            logging.info("GPU detected and enabled for PyTorch models (Whisper, YOLO, CLIP).")
        else:
            device = "cpu"
            logging.info("No GPU detected; falling back to CPU for PyTorch models.")

        whisper_model = whisper.load_model("base", device=device)
        yolo_model = YOLO('yolov8n.pt').to(device)
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        text_model = SentenceTransformer('all-MiniLM-L6-v2')
    else:
        logging.warning("Backend dependencies are incomplete; frontend mock mode is recommended.")

except ImportError as e:
    logging.warning(f"Backend import error: {e}. Frontend mock mode can still be used.")

# Directories
UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
CLIPS_DIR = "clips"

# Ensure directories exist
for dir_path in [UPLOAD_DIR, PROCESSED_DIR, CLIPS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

DETECTION_HEADERS = ["时间戳", "类别", "置信度", "目标图像"]
RETRIEVAL_HEADERS = ["排名", "相似度", "时间戳", "类别", "说明"]

MOCK_DETECTIONS = [
    {
        "timestamp": "00:00:03.000",
        "label_zh": "行人",
        "label_en": "person",
        "confidence": 0.87,
        "bbox": "[88, 42, 220, 360]",
        "description": "校园道路上的行人",
    },
    {
        "timestamp": "00:00:08.000",
        "label_zh": "自行车",
        "label_en": "bicycle",
        "confidence": 0.79,
        "bbox": "[240, 180, 430, 350]",
        "description": "校园道路或停车区附近的自行车",
    },
    {
        "timestamp": "00:00:15.000",
        "label_zh": "汽车",
        "label_en": "car",
        "confidence": 0.83,
        "bbox": "[20, 160, 190, 320]",
        "description": "校园道路、校门或停车区附近的汽车",
    },
]


def _safe_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _status_html(state_text="等待处理...", sampled_frames=0, total_detections=0, note="请上传视频后点击处理。"):
    return f"""
    <div class="status-panel">
        <div class="status-strip">
            <span>当前状态</span>
            <strong>{state_text}</strong>
        </div>
        <div class="status-metrics">
            <div class="metric-box">
                <span>已检测帧数</span>
                <strong>{sampled_frames}</strong>
            </div>
            <div class="metric-box">
                <span>发现目标</span>
                <strong>{total_detections}</strong>
            </div>
        </div>
        <p class="status-note">{note}</p>
    </div>
    """


def _answer_html(text="", filled=False):
    if filled:
        return f"""
        <div class="answer-content">
            <p>{text}</p>
        </div>
        """

    return """
    <div class="answer-empty">
        <div class="answer-empty-icon">▱</div>
        <p>提问后，AI 将基于视频检测内容为您生成详细分析回答</p>
    </div>
    """


def _ensure_mock_outputs(video_path):
    """
    Build small local files so the Gradio frontend can be demonstrated before
    the YOLO/CLIP/LLM modules are submitted by other team members.
    """
    run_dir = os.path.join(PROCESSED_DIR, f"frontend_mock_{_safe_timestamp()}")
    crops_dir = os.path.join(run_dir, "object_crops")
    os.makedirs(crops_dir, exist_ok=True)

    csv_path = os.path.join(run_dir, "detections.csv")
    crop_colors = [(64, 132, 232), (33, 166, 120), (230, 137, 46)]

    rows = []
    for idx, item in enumerate(MOCK_DETECTIONS, start=1):
        crop_path = os.path.join(crops_dir, f"mock_{item['label_en']}_{idx:03d}.jpg")
        image = Image.new("RGB", (320, 220), crop_colors[idx - 1])
        draw = ImageDraw.Draw(image)
        draw.rectangle((18, 150, 302, 202), fill=(255, 255, 255))
        draw.text((30, 162), f"{item['timestamp']}  {item['label_zh']}", fill=(30, 42, 56))
        draw.text((30, 182), f"{item['label_en']}  conf={item['confidence']:.2f}", fill=(70, 82, 96))
        rows.append([
            item["timestamp"],
            item["label_zh"],
            f"{item['confidence']:.2f}",
            crop_path,
        ])
        try:
            image.save(crop_path)
        except Exception as exc:
            logging.warning(f"Could not save mock crop image: {exc}")

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(DETECTION_HEADERS)
        writer.writerows(rows)

    return {
        "run_dir": run_dir,
        "annotated_video_path": video_path,
        "detections_csv": csv_path,
        "detections_table": rows,
        "total_detections": len(rows),
    }


def frontend_process_video(video_path, sample_interval_sec, conf_threshold):
    """
    Frontend adapter. Keep Gradio connected to this function, then replace the
    internals with the real backend pipeline when teammates submit it.
    """
    if not video_path:
        return (
            _status_html("等待上传", 0, 0, "请先上传视频。"),
            None,
            [],
            None,
            None,
        )

    if FRONTEND_USE_MOCK:
        mock = _ensure_mock_outputs(video_path)
        status = _status_html(
            "mock 处理完成",
            sampled_frames=mock["total_detections"],
            total_detections=mock["total_detections"],
            note=(
                f"抽帧间隔 {sample_interval_sec:.1f}s，YOLO 置信度阈值 {conf_threshold:.2f}。"
                "当前使用模拟数据，后续只需替换 frontend_process_video() 内部调用。"
            ),
        )
        return (
            status,
            mock["annotated_video_path"],
            mock["detections_table"],
            mock["run_dir"],
            mock["detections_csv"],
        )

    result = process_video(video_path)
    if result["status"] == "Processed successfully":
        index_data(result["data"])
        rows = []
        for frame in result["data"]:
            timestamp = frame.get("timestamp", 0)
            for detection in frame.get("detections", []):
                bbox = detection.get("bbox", [[]])[0]
                rows.append([
                    f"{timestamp:.2f}s",
                    detection.get("class", ""),
                    f"{detection.get('confidence', 0):.2f}",
                    "",
                ])
        status = _status_html(
            "真实后端处理完成",
            sampled_frames=len(result["data"]),
            total_detections=len(rows),
            note="已调用当前 main.py 中的旧后端处理流程。",
        )
        return status, video_path, rows, PROCESSED_DIR, None

    return _status_html("处理失败", 0, 0, result["status"]), None, [], None, None


def frontend_query(question, run_dir, top_k):
    """
    Frontend adapter for CLIP retrieval + LLM QA. It returns answer text,
    gallery items and a retrieval table in the shape Gradio needs.
    """
    if not run_dir:
        return _answer_html("请先上传并处理视频，再进行问答。", filled=True), [], []

    if not question or not question.strip():
        return _answer_html("请输入一个关于视频内容的问题。", filled=True), [], []

    if FRONTEND_USE_MOCK:
        selected = MOCK_DETECTIONS[: max(1, min(int(top_k), len(MOCK_DETECTIONS)))]
        gallery_items = []
        retrieval_rows = []
        for rank, item in enumerate(selected, start=1):
            crop_name = f"mock_{item['label_en']}_{rank:03d}.jpg"
            crop_path = os.path.join(run_dir, "object_crops", crop_name)
            score = 0.86 - (rank - 1) * 0.08
            caption = f"{item['timestamp']} {item['label_zh']} score={score:.2f}"
            gallery_items.append((crop_path, caption))
            retrieval_rows.append([
                rank,
                f"{score:.2f}",
                item["timestamp"],
                item["label_zh"],
                item["description"],
            ])

        labels = "、".join(item["label_zh"] for item in selected)
        answer = (
            "基于当前模拟检索结果，视频中可以看到"
            f"{labels}。最相关证据出现在 {selected[0]['timestamp']}，"
            f"目标为{selected[0]['label_zh']}。真实 YOLO、CLIP 和 LLM 模块接入后，"
            "这里会展示模型基于检索结果生成的回答。"
        )
        return _answer_html(answer, filled=True), gallery_items, retrieval_rows

    result = query_system(question)
    return _answer_html(result.get("answer", "未获得回答。"), filled=True), [], []

def process_video(video_path):
    """
    Process uploaded video: Extract audio, detect objects, OCR, generate embeddings.
    Returns processed data for indexing.
    """
    missing = []
    for name, module in [
        ("cv2", cv2),
        ("whisper", whisper),
        ("torch", torch),
        ("YOLO", YOLO),
        ("pytesseract", pytesseract),
        ("VideoFileClip", VideoFileClip),
        ("CLIPModel", CLIPModel),
        ("CLIPProcessor", CLIPProcessor),
    ]:
        if module is None:
            missing.append(name)
    if missing or yolo_model is None or clip_model is None or clip_processor is None:
        return {
            "status": (
                "Backend dependencies or models are not ready. "
                f"Missing: {', '.join(missing) if missing else 'model initialization'}. "
                "Use the default mock frontend mode until the backend modules are submitted."
            ),
            "data": [],
        }

    if not video_path:
        return {"status": "No video uploaded", "data": []}
    
    # Ensure the file exists
    if not os.path.exists(video_path):
        return {"status": f"Video file not found: {video_path}", "data": []}
    
    # Get absolute path to ensure accessibility
    video_path = os.path.abspath(video_path)
    logging.info(f"Processing video at: {video_path}")
    
    # Copy the video to uploads directory to ensure it's accessible
    try:
        import shutil
        video_name = os.path.basename(video_path)
        local_video_path = os.path.join(UPLOAD_DIR, video_name)
        shutil.copy2(video_path, local_video_path)
        video_path = local_video_path  # Use the copied path
        logging.info(f"Copied video to: {video_path}")
        
        # Verify the copied file exists and is valid
        if not os.path.exists(video_path):
            return {"status": f"Copied video file not found: {video_path}", "data": []}
        
        # Check if the video file is valid by trying to open it
        test_cap = cv2.VideoCapture(video_path)
        if not test_cap.isOpened():
            test_cap.release()
            return {"status": f"Invalid video file: {video_path}", "data": []}
        test_cap.release()
    except Exception as e:
        return {"status": f"Error copying video file: {str(e)}", "data": []}
    
    global current_video_path
    current_video_path = video_path
    
    try:
        # Load video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            return {"status": f"Cannot open video file: {video_path}", "data": []}
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Extract audio for transcription (optional)
        audio_path = os.path.join(UPLOAD_DIR, "temp_audio.wav")
        transcription = ""
        segments = []
        video_clip = None
        try:
            video_clip = VideoFileClip(video_path)
            if video_clip.audio is None:
                logging.warning("No audio found in video")
            else:
                logging.info(f"Writing audio to: {audio_path}")
                video_clip.audio.write_audiofile(audio_path, verbose=False, logger=None)
                logging.info("Audio extracted successfully")
                # Transcribe audio with Whisper
                result = whisper_model.transcribe(audio_path)
                transcription = result['text']
                segments = result['segments']
                logging.info("Audio transcription completed")
        except Exception as e:
            logging.error(f"Audio extraction/transcription failed: {str(e)}. Proceeding without audio.")
            transcription = "No audio available"
            segments = []
        finally:
            if video_clip:
                video_clip.close()
            if os.path.exists(audio_path):
                os.remove(audio_path)
        
        processed_data = []
        
        # Process frames at intervals
        for i in range(0, total_frames, int(fps * 2)):  # Every 2 seconds
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = i / fps
            
            # Object detection with YOLO
            results = yolo_model(frame)
            detections = []
            for result in results:
                for box in result.boxes:
                    detections.append({
                        'class': result.names[int(box.cls)],
                        'confidence': float(box.conf),
                        'bbox': box.xyxy.tolist()
                    })
            
            # OCR with Tesseract
            ocr_text = ""
            try:
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                ocr_text = pytesseract.image_to_string(pil_image)
            except Exception as e:
                logging.error(f"OCR failed: {str(e)}. Proceeding without OCR.")
                ocr_text = "OCR not available"
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Fallback
            
            # CLIP embeddings for the frame
            inputs = clip_processor(images=pil_image, return_tensors="pt").to(device)
            with torch.no_grad():
                embeddings = clip_model.get_image_features(**inputs).cpu().numpy()
            
            # Store data
            processed_data.append({
                'timestamp': timestamp,
                'transcription': transcription,
                'detections': detections,
                'ocr_text': ocr_text,
                'embeddings': embeddings.flatten().tolist()
            })
        
        cap.release()
        if video_clip:
            video_clip.close()
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        return {"status": "Processed successfully", "data": processed_data}
    except Exception as e:
        if 'cap' in locals():
            cap.release()
        if video_clip:
            video_clip.close()
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return {"status": f"Error: {str(e)}", "data": []}

def index_data(data):
    """
    Index processed data using FAISS.
    """
    if faiss is None or np is None:
        logging.warning("FAISS or NumPy is unavailable; skipping indexing.")
        return

    if not data:
        return
    
    # Extract embeddings and metadata
    embeddings = np.array([item['embeddings'] for item in data], dtype=np.float32)
    metadata = data  # Keep full metadata
    
    # Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance index
    index.add(embeddings)
    
    # Save index and metadata
    faiss.write_index(index, os.path.join(PROCESSED_DIR, "faiss_index.idx"))
    np.save(os.path.join(PROCESSED_DIR, "metadata.npy"), metadata)

def query_system(query):
    """
    Process query and return results with timestamps and clips.
    """
    if faiss is None or np is None or text_model is None:
        return {"answer": "Backend retrieval dependencies are not ready. Please use mock mode or install backend modules.", "clips": []}

    try:
        # Load FAISS index and metadata
        if not os.path.exists(os.path.join(PROCESSED_DIR, "faiss_index.idx")):
            return {"answer": "No processed video data available. Please upload and process a video first.", "clips": []}
        
        index = faiss.read_index(os.path.join(PROCESSED_DIR, "faiss_index.idx"))
        metadata = np.load(os.path.join(PROCESSED_DIR, "metadata.npy"), allow_pickle=True)
        
        # Generate query embedding
        query_embedding = text_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        
        # Search for top-k similar frames
        k = 5  # Number of results
        distances, indices = index.search(query_embedding, k)
        
        # Retrieve metadata for top results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # Valid index
                result = metadata[idx]
                results.append(result)
        
        # Generate video clips
        clips = []
        for result in results:
            timestamp = result['timestamp']
            clip_path = generate_clip(video_path, timestamp, duration=10)  # 10-second clip
            clips.append(clip_path)
        
        # Generate answer
        answer = f"Found {len(results)} relevant segments. Key events around timestamps: {[r['timestamp'] for r in results]}"
        
        return {"answer": answer, "clips": clips}
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "clips": []}

def generate_clip(video_path, start_time, duration=10):
    """
    Generate a video clip from start_time for duration seconds.
    """
    if VideoFileClip is None:
        logging.warning("MoviePy is unavailable; cannot generate clips.")
        return None

    try:
        clip = VideoFileClip(video_path).subclip(start_time, start_time + duration)
        output_path = os.path.join(CLIPS_DIR, f"clip_{start_time}.mp4")
        clip.write_videofile(output_path, verbose=False, logger=None)
        clip.close()
        return output_path
    except Exception as e:
        logging.error(f"Error generating clip: {str(e)}")
        return None

def gradio_interface():
    css = """
    :root {
        --cv-bg: #f7fafb;
        --cv-card: #ffffff;
        --cv-card-soft: #f1f4f5;
        --cv-border: #dfe6e7;
        --cv-border-strong: #c0c8c9;
        --cv-text: #181c1d;
        --cv-muted: #627173;
        --cv-primary: #3b656a;
        --cv-primary-soft: #8fbabf;
        --cv-primary-pale: #d7ecef;
        --cv-tertiary: #dca78c;
    }
    .gradio-container {
        background: var(--cv-bg) !important;
        color: var(--cv-text);
        font-family: Inter, "Microsoft YaHei", Arial, sans-serif;
        --color-accent: var(--cv-primary-soft) !important;
        --color-accent-soft: var(--cv-primary-pale) !important;
        --button-primary-background-fill: var(--cv-primary-soft) !important;
        --button-primary-background-fill-hover: #9fc8cd !important;
        --button-primary-text-color: #0f2b2f !important;
        --slider-color: var(--cv-primary-soft) !important;
        --checkbox-label-background-fill-selected: var(--cv-primary-pale) !important;
        --checkbox-label-text-color-selected: var(--cv-primary) !important;
        --input-border-color-focus: var(--cv-primary-soft) !important;
        --input-shadow-focus: 0 0 0 1px var(--cv-primary-soft) !important;
    }
    footer,
    .footer,
    .built-with,
    .api-docs {
        display: none !important;
    }
    .gradio-container a,
    .gradio-container button,
    .gradio-container label.selected,
    .gradio-container .text-blue-500,
    .gradio-container .text-blue-600 {
        color: var(--cv-primary) !important;
    }
    .gradio-container .bg-blue-500,
    .gradio-container .bg-blue-600 {
        background: var(--cv-primary-soft) !important;
    }
    input[type="range"] {
        accent-color: var(--cv-primary-soft) !important;
    }
    input[type="range"]::-webkit-slider-thumb {
        background: var(--cv-primary-soft) !important;
    }
    input[type="range"]::-moz-range-thumb {
        background: var(--cv-primary-soft) !important;
    }
    .gradio-container input:focus,
    .gradio-container textarea:focus {
        border-color: var(--cv-primary-soft) !important;
        box-shadow: 0 0 0 1px var(--cv-primary-soft) !important;
    }
    .app-shell {
        max-width: 1180px;
        margin: 0 auto;
        padding: 0 6px 96px;
    }
    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 24px;
        padding: 24px 4px 20px;
        border-bottom: 1px solid var(--cv-border);
        margin-bottom: 22px;
    }
    .brand {
        display: flex;
        align-items: center;
        gap: 18px;
    }
    .brand-mark {
        width: 38px;
        height: 26px;
        border-radius: 999px;
        border: 3px solid var(--cv-primary-soft);
        position: relative;
        transform: rotate(0deg);
    }
    .brand-mark::after {
        content: "";
        position: absolute;
        width: 10px;
        height: 10px;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        border-radius: 999px;
        background: var(--cv-primary-soft);
    }
    .brand-name {
        display: flex;
        align-items: end;
        gap: 10px;
    }
    .brand-title {
        margin: 0;
        font-size: 38px;
        font-weight: 800;
        line-height: 1;
        letter-spacing: 0;
    }
    .brand-subtitle {
        color: var(--cv-muted);
        font-size: 16px;
        font-weight: 650;
        padding-bottom: 4px;
    }
    .mode-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 14px;
        border-radius: 999px;
        background: #ebeeef;
        color: #465557;
        font-size: 12px;
        white-space: nowrap;
    }
    .mode-dot {
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: var(--cv-primary-soft);
    }
    .intro {
        color: #3f4d4f;
        font-size: 15px;
        line-height: 1.7;
        max-width: 860px;
        margin: 0 0 22px;
    }
    .cv-card {
        background: var(--cv-card);
        border: 1px solid var(--cv-border);
        border-radius: 12px;
        box-shadow: 0 6px 22px rgba(24, 28, 29, 0.035);
        padding: 22px;
    }
    .cv-card-compact {
        background: var(--cv-card);
        border: 1px solid var(--cv-border);
        border-radius: 12px;
        box-shadow: 0 6px 22px rgba(24, 28, 29, 0.035);
        overflow: hidden;
    }
    .cv-card h3,
    .cv-card-compact h3 {
        margin-top: 0;
    }
    .cv-section-title {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 16px;
        font-size: 18px;
        font-weight: 800;
    }
    .cv-title-left {
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    .cv-icon {
        width: 21px;
        height: 21px;
        color: var(--cv-primary);
        stroke: currentColor;
        stroke-width: 2;
        fill: none;
        stroke-linecap: round;
        stroke-linejoin: round;
        flex: 0 0 auto;
    }
    .cv-title-icon {
        color: var(--cv-primary);
        font-weight: 800;
    }
    .cv-chip {
        display: inline-flex;
        align-items: center;
        padding: 3px 9px;
        border-radius: 6px;
        background: #d5e6e8;
        color: #455c5f;
        font-size: 12px;
        font-weight: 650;
    }
    .cv-button button {
        background: var(--cv-primary-soft) !important;
        border: 0 !important;
        color: #0f2b2f !important;
        min-height: 52px;
        border-radius: 8px !important;
        font-size: 16px !important;
        font-weight: 800 !important;
        box-shadow: none !important;
    }
    .cv-button button *,
    .cv-button button span {
        color: #0f2b2f !important;
    }
    .cv-button button:hover {
        filter: brightness(1.03);
        transform: translateY(-1px);
    }
    .cv-upload video,
    .cv-visual video {
        border-radius: 10px !important;
        border: 1px dashed var(--cv-border-strong) !important;
        background: #eef2f3 !important;
    }
    .cv-upload,
    .cv-upload > div {
        min-height: 250px !important;
    }
    .cv-upload .wrap,
    .cv-upload .upload-container,
    .cv-upload [data-testid="block-info"],
    .cv-upload .empty {
        border-style: dashed !important;
        border-color: var(--cv-border-strong) !important;
        background: #f1f4f5 !important;
    }
    .cv-upload svg,
    .cv-upload .icon-wrap,
    .cv-upload .upload-icon {
        color: var(--cv-primary-soft) !important;
    }
    .cv-slider {
        padding-top: 2px;
    }
    .slider-caption {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        margin: 12px 0 8px;
        font-weight: 700;
        color: #263234;
    }
    .slider-caption strong {
        color: var(--cv-primary);
        font-family: "JetBrains Mono", ui-monospace, monospace;
    }
    .field-label {
        display: block;
        margin: 4px 0 8px;
        color: #263234;
        font-size: 14px;
        font-weight: 650;
    }
    .cv-slider input[type="range"] {
        accent-color: var(--cv-primary-soft);
    }
    .status-panel {
        display: flex;
        flex-direction: column;
        gap: 18px;
    }
    .status-strip {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        padding: 18px 20px;
        background: #f0f3f4;
        border-radius: 8px;
        color: #3f4e50;
    }
    .status-strip strong {
        color: var(--cv-primary);
        font-weight: 800;
    }
    .status-metrics {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 18px;
    }
    .metric-box {
        padding: 18px;
        border: 1px solid var(--cv-border-strong);
        border-radius: 8px;
        background: #fff;
    }
    .metric-box span {
        display: block;
        color: var(--cv-muted);
        font-size: 13px;
        margin-bottom: 8px;
    }
    .metric-box strong {
        color: #000;
        font-size: 30px;
        line-height: 1;
    }
    .status-note {
        color: var(--cv-muted);
        line-height: 1.65;
        margin: 0;
    }
    .panel-head {
        padding: 16px 20px;
        border-bottom: 1px solid var(--cv-border);
        background: #f1f4f5;
        font-size: 18px;
        font-weight: 800;
    }
    .panel-body {
        padding: 18px;
    }
    .answer-shell {
        min-height: 300px;
        background: #fff;
        display: flex;
        align-items: stretch;
    }
    .answer-shell > div {
        width: 100%;
    }
    .answer-empty {
        min-height: 300px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 16px;
        color: #9ba5a7;
        text-align: center;
    }
    .answer-empty-icon {
        font-size: 48px;
        line-height: 1;
        color: #a5adaf;
    }
    .answer-content {
        min-height: 300px;
        padding: 24px;
        color: #263234;
        line-height: 1.8;
        background: #fff;
        border-top: 1px solid transparent;
    }
    .answer-content p {
        margin: 0;
    }
    .query-box textarea {
        min-height: 128px !important;
        background: #f4f7f8 !important;
    }
    .cv-dataframe table {
        border-radius: 0 !important;
    }
    .cv-dataframe th {
        background: #f1f4f5 !important;
        color: #303b3d !important;
        font-weight: 800 !important;
    }
    .cv-gallery {
        min-height: 320px;
    }
    .cv-gallery > div {
        min-height: 300px !important;
        border: 1px dashed var(--cv-border) !important;
        border-radius: 10px !important;
        background: #fbfcfc !important;
    }
    .download-button button,
    .download-button a {
        width: 100% !important;
        min-height: 64px !important;
        justify-content: space-between !important;
        border: 1px solid var(--cv-primary-pale) !important;
        border-radius: 8px !important;
        background: #fff !important;
        color: var(--cv-primary) !important;
        font-weight: 750 !important;
        box-shadow: none !important;
    }
    .download-button button:hover,
    .download-button a:hover {
        background: #f1f7f8 !important;
    }
    @media (max-width: 780px) {
        .topbar {
            align-items: flex-start;
            flex-direction: column;
        }
        .brand-title {
            font-size: 32px;
        }
    }
    """

    with gr.Blocks(
        title="CampusVision 视频目标检索问答",
        css=css,
        theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"),
    ) as interface:
        current_run_dir = gr.State(value=None)

        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(
                """
                <div class="topbar">
                    <div class="brand">
                        <div class="brand-mark"></div>
                        <div class="brand-name">
                            <h1 class="brand-title">智瞳</h1>
                            <span class="brand-subtitle">CampusVision</span>
                        </div>
                    </div>
                    <div class="mode-pill">
                        <span class="mode-dot"></span>
                        演示模式 / 模拟数据已启用
                    </div>
                </div>
                <p class="intro">
                    校园场景视频目标检测、检索与智能问答前端。当前页面已预留 YOLO、CLIP、FAISS、LLM 接口，可先用模拟数据完成演示。
                </p>
                """
            )

            with gr.Row(equal_height=True):
                with gr.Column(scale=7, elem_classes=["cv-card"]):
                    gr.HTML(
                        """
                        <div class="cv-section-title">
                            <span class="cv-title-left">
                                <svg class="cv-icon" viewBox="0 0 24 24"><path d="M14 2H7a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7z"/><path d="M14 2v5h5"/><path d="M12 18v-6"/><path d="M9 15l3-3 3 3"/></svg>
                                1. 视频上传与处理
                            </span>
                            <span class="cv-chip">支持 MP4, AVI</span>
                        </div>
                        """
                    )
                    video_upload = gr.Video(label="上传校园场景视频", show_label=False, elem_classes=["cv-upload"])
                    with gr.Row():
                        with gr.Column():
                            gr.HTML("<div class='slider-caption'><span>抽帧间隔（秒）</span><strong>1.0s</strong></div>")
                            sample_interval = gr.Slider(
                                minimum=0.5,
                                maximum=3.0,
                                value=1.0,
                                step=0.5,
                                label="",
                                show_label=False,
                                elem_classes=["cv-slider"],
                            )
                        with gr.Column():
                            gr.HTML("<div class='slider-caption'><span>YOLO 置信度阈值</span><strong>0.25</strong></div>")
                            conf_threshold = gr.Slider(
                                minimum=0.1,
                                maximum=0.8,
                                value=0.25,
                                step=0.05,
                                label="",
                                show_label=False,
                                elem_classes=["cv-slider"],
                            )
                    process_button = gr.Button("▷ 处理视频", variant="primary", size="lg", elem_classes=["cv-button"])
                with gr.Column(scale=5, elem_classes=["cv-card"]):
                    gr.HTML(
                        """
                        <div class="cv-section-title">
                            <span class="cv-title-left">
                                <svg class="cv-icon" viewBox="0 0 24 24"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 8h10"/><path d="M7 12h7"/><path d="M7 16h4"/></svg>
                                2. 处理状态
                            </span>
                        </div>
                        """
                    )
                    process_output = gr.HTML(_status_html())
                    detections_file = gr.DownloadButton(
                        label="下载检测结果 detections.csv",
                        value=None,
                        elem_classes=["download-button"],
                    )

            with gr.Row(equal_height=True):
                with gr.Column(scale=6, elem_classes=["cv-card-compact"]):
                    gr.HTML("<div class='panel-head'><svg class='cv-icon' viewBox='0 0 24 24' style='vertical-align:-4px;margin-right:8px'><path d='M15 10l4.5-3v10L15 14z'/><rect x='3' y='6' width='12' height='12' rx='2'/></svg>3. YOLO 检测可视化 <span class='cv-chip' style='float:right'>检测预览</span></div>")
                    with gr.Column(elem_classes=["panel-body"]):
                        annotated_video = gr.Video(label="检测后视频 / 可视化占位", show_label=False, elem_classes=["cv-visual"])
                with gr.Column(scale=6, elem_classes=["cv-card-compact"]):
                    gr.HTML("<div class='panel-head'><svg class='cv-icon' viewBox='0 0 24 24' style='vertical-align:-4px;margin-right:8px'><path d='M3 5h18'/><path d='M3 12h18'/><path d='M3 19h18'/><path d='M8 5v14'/><path d='M16 5v14'/></svg>4. 检测结果表</div>")
                    detections_table = gr.Dataframe(
                        headers=DETECTION_HEADERS,
                        datatype=["str", "str", "str", "str"],
                        row_count=(3, "dynamic"),
                        col_count=(4, "fixed"),
                        label="检测目标列表",
                        interactive=False,
                        elem_classes=["cv-dataframe"],
                    )

            with gr.Row(equal_height=True):
                with gr.Column(scale=5, elem_classes=["cv-card"]):
                    gr.HTML(
                        """
                        <div class="cv-section-title">
                            <span class="cv-title-left">
                                <svg class="cv-icon" viewBox="0 0 24 24"><path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4z"/></svg>
                                5. 视频内容问答
                            </span>
                        </div>
                        """
                    )
                    gr.HTML("<span class='field-label'>请输入问题</span>")
                    query_input = gr.Textbox(
                        label="请输入问题",
                        show_label=False,
                        placeholder="例如：视频中有没有人？是否出现自行车？有哪些车辆？",
                        lines=3,
                        elem_classes=["query-box"],
                    )
                    gr.HTML("<div class='slider-caption'><span>检索返回数量 top-k</span><strong>3</strong></div>")
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="",
                        show_label=False,
                        elem_classes=["cv-slider"],
                    )
                    query_button = gr.Button("◎ 检索并回答", variant="primary", size="lg", elem_classes=["cv-button"])
                with gr.Column(scale=7, elem_classes=["cv-card-compact"]):
                    gr.HTML("<div class='panel-head'><svg class='cv-icon' viewBox='0 0 24 24' style='vertical-align:-4px;margin-right:8px'><rect x='4' y='7' width='16' height='12' rx='2'/><path d='M8 7V4h8v3'/><path d='M8 13h.01'/><path d='M16 13h.01'/><path d='M9 17h6'/></svg>6. LLM 回答</div>")
                    with gr.Column(elem_classes=["answer-shell"]):
                        query_output = gr.HTML(_answer_html())

            with gr.Column(elem_classes=["cv-card-compact"]):
                gr.HTML("<div class='panel-head'><svg class='cv-icon' viewBox='0 0 24 24' style='vertical-align:-4px;margin-right:8px'><path d='M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z'/><path d='M9 12l2 2 4-4'/></svg>7. 匹配目标证据</div>")
                with gr.Row(equal_height=True):
                    with gr.Column(scale=4, elem_classes=["panel-body"]):
                        gr.HTML("<div class='cv-section-title' style='font-size:14px'><span>匹配目标图片 Gallery</span><span class='cv-chip'>证据图</span></div>")
                        match_gallery = gr.Gallery(
                            label="匹配目标图片 Gallery",
                            show_label=False,
                            columns=2,
                            height=330,
                            object_fit="cover",
                            elem_classes=["cv-gallery"],
                        )
                    with gr.Column(scale=8, elem_classes=["panel-body"]):
                        gr.HTML("<div class='cv-section-title' style='font-size:14px'><span>检索结果列表</span><span class='cv-chip'>排序方式：相似度</span></div>")
                        retrieval_table = gr.Dataframe(
                            headers=RETRIEVAL_HEADERS,
                            datatype=["number", "str", "str", "str", "str"],
                            row_count=(3, "dynamic"),
                            col_count=(5, "fixed"),
                            label="检索结果",
                            interactive=False,
                            elem_classes=["cv-dataframe"],
                        )

        process_button.click(
            fn=frontend_process_video,
            inputs=[video_upload, sample_interval, conf_threshold],
            outputs=[
                process_output,
                annotated_video,
                detections_table,
                current_run_dir,
                detections_file,
            ],
        )

        query_button.click(
            fn=frontend_query,
            inputs=[query_input, current_run_dir, top_k],
            outputs=[query_output, match_gallery, retrieval_table],
        )

    return interface

if __name__ == "__main__":
    interface = gradio_interface()
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    interface.launch(server_port=server_port)  # Set GRADIO_SERVER_PORT to override.

# =============================================================================
# 智瞳 CampusVision - LLM 问答模块接入点说明 (由 LLM 模块负责人添加, 不影响原逻辑)
# =============================================================================
#
# 改造目标: 把上面 `query_system()` 中硬编码的 `f"Found {len(results)} relevant
# segments. ..."` 替换为基于 LLM 的自然语言回答, 同时把 `generate_clip()` 替换为
# 直接展示 CLIP B 检索返回的目标裁剪图 (crop_path) Gallery。
#
# 推荐接入位置: `on_query()` 函数 (位于本文件约 294 行)。
#
# 示例替换代码 (Gradio 同学可按需改造) ------------------------------
#
#     from pathlib import Path
#     from llm_qa.qa_engine import answer_question
#     from clip_search import search_by_text
#
#     def on_query(query, video_state):
#         # 1) CLIP B 检索: 用户问题 -> top-k 目标元数据
#         results = search_by_text(
#             query=query,
#             index_dir="processed/clip_index",   # CLIP A 产物目录
#             top_k=5,
#         )
#         # 2) 组装 video_meta (默认从 video_state 推断)
#         if isinstance(video_state, dict):
#             video_name = Path(video_state.get("video_path", "uploaded.mp4")).name
#             duration = video_state.get("duration", "-")
#         else:
#             video_name, duration = "uploaded.mp4", "-"
#         # 3) LLM 问答: 基于检索证据生成自然语言回答
#         result = answer_question(
#             question=query,
#             retrieval_results=results,
#             video_meta={"video_name": video_name,
#                         "duration": duration,
#                         "detection_count": len(results)},
#         )
#         # 4) 返回: 文本 + 目标裁剪图路径列表 (Gradio 用 gr.Gallery 渲染)
#         image_paths = [r["crop_path"] for r in results]
#         return result, image_paths
#
# 配套要求:
#   1. CLIP B 同学已实现 `clip_search/search_clip.py` 并提供
#      `search_by_text()` 接口 (见 llm_qa/contract_with_clip_b.md)。
#   2. YOLO 同学确保 `detections.csv` / `clip_metadata.json` 中含
#      `timestamp` / `class_name` / `crop_path` 字段 (已是项目基础要求)。
#   3. 在项目根 `requirements.txt` 中已追加:
#         openai>=1.0.0
#         python-dotenv>=1.0.0
#   4. 在项目根创建 `.env` (或复制 `llm_qa/.env.example` 为
#      `llm_qa/.env`), 配置 `DEEPSEEK_API_KEY=sk-xxx`。
#   5. CLIP A 同学已生成 `processed/clip_index/{faiss.index,
#      clip_metadata.json, image_embeddings.npy}`。
#
# 完整文档与测试: 见 `llm_qa/README.md` 与 `llm_qa/tests/test_qa_engine.py`。
# =============================================================================

