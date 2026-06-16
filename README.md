# Video RAG (Retrieval-Augmented Generation) System

A powerful video analysis and retrieval system that processes videos to extract meaningful information and enables natural language queries to find relevant video segments.

## Features

- **Video Processing**: Extract frames, audio, and generate video clips
- **Object Detection**: Identify objects in video frames using YOLOv8
- **Audio Transcription**: Convert speech to text using OpenAI's Whisper
- **OCR (Optical Character Recognition)**: Extract text from video frames using Tesseract
- **Semantic Search**: Find relevant video segments using natural language queries
- **Web Interface**: User-friendly Gradio interface for easy interaction

## Prerequisites

- Python 3.8+
- FFmpeg (for audio/video processing)
- Tesseract OCR (for text recognition)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ANN-Video-RAG
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On Linux/Mac
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Install system dependencies:
   - **FFmpeg**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
   - **Tesseract OCR**: Download from [GitHub - tesseract-ocr/tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and add to PATH

## Usage

1. Run the application:
   ```bash
   python main.py
   ```

2. Open your web browser and go to: `http://127.0.0.1:7860`

3. Upload a video file and click "Process Video"

4. Once processed, enter natural language queries to find relevant video segments

## Project Structure

- `main.py`: Main application code
- `requirements.txt`: Python dependencies
- `uploads/`: Directory for uploaded videos
- `processed/`: Directory for processed data and FAISS index
- `clips/`: Directory for generated video clips

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- YOLOv8 for object detection
- OpenAI Whisper for speech recognition
- Tesseract OCR for text recognition
- FAISS for efficient similarity search
- Gradio for the web interface
