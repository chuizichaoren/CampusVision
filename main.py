import gradio as gr
import os
import cv2
import whisper
import torch
from ultralytics import YOLO
import pytesseract
# Comment out the path setting to use system PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from PIL import Image
import faiss
import numpy as np
from moviepy.editor import VideoFileClip
from moviepy.config import change_settings
# Comment out FFMPEG path to use default
# change_settings({"FFMPEG_BINARY": r"C:\Program Files\FFmpeg\bin\ffmpeg.exe"})
from transformers import CLIPProcessor, CLIPModel
import tempfile
import shutil
from sentence_transformers import SentenceTransformer
import logging

# Add logging for device detection and imports
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    # Your existing model initializations with device and logging
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

except ImportError as e:
    logging.error(f"Import error: {e}. Ensure all dependencies are installed correctly.")
    raise

# Directories
UPLOAD_DIR = "uploads"
PROCESSED_DIR = "processed"
CLIPS_DIR = "clips"

# Ensure directories exist
for dir_path in [UPLOAD_DIR, PROCESSED_DIR, CLIPS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

def process_video(video_path):
    """
    Process uploaded video: Extract audio, detect objects, OCR, generate embeddings.
    Returns processed data for indexing.
    """
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
    with gr.Blocks(title="Video RAG System") as interface:
        gr.Markdown("# Video RAG System\nUpload a video, process it, and query for events!")

        with gr.Row():
            video_upload = gr.Video(label="Upload Video (.mp4 or .mov)")
            process_button = gr.Button("Process Video")

        process_output = gr.Textbox(label="Processing Status")

        with gr.Row():
            query_input = gr.Textbox(label="Enter Query (e.g., 'When does Kohli hit a six?')")
            query_button = gr.Button("Submit Query")

        query_output = gr.Textbox(label="Answer")
        video_output = gr.Video(label="Retrieved Clips")
        download_button = gr.Button("Download Clips")

        def on_process(video_path):
            result = process_video(video_path)
            if result["status"] == "Processed successfully":
                index_data(result["data"])
                return f"Video processed and indexed. {len(result['data'])} frames analyzed."
            return result["status"]

        def on_query(query):
            result = query_system(query)
            clips = result.get("clips", [])
            if clips:
                return result["answer"], clips[0] if clips else None  # Show first clip
            return result["answer"], None

        process_button.click(fn=on_process, inputs=video_upload, outputs=process_output)
        query_button.click(fn=on_query, inputs=query_input, outputs=[query_output, video_output])

        # Simple download functionality (for demo, could be improved)
        def download_clips():
            return "Download functionality - clips saved in 'clips' folder"

        download_button.click(fn=download_clips, outputs=query_output)

    return interface

if __name__ == "__main__":
    interface = gradio_interface()
    interface.launch()  # Use localhost; if blocked, run as admin or allow in Windows Defender
