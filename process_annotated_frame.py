import cv2
import os
import csv
import json
import numpy as np
from ultralytics import YOLO

def extract_timestamp_from_filename(filename):
    """从文件名中提取时间戳信息"""
    parts = filename.split('_')
    if len(parts) >= 3:
        time_part = parts[2].replace('.jpg', '').replace('.png', '')
        return time_part
    return "00-00-00-000"

def parse_timecode(timecode_str):
    """解析时间码字符串为秒数"""
    parts = timecode_str.split('-')
    if len(parts) >= 4:
        h, m, s, ms = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        return h * 3600 + m * 60 + s + ms / 1000.0
    return 0.0

def process_single_frame(image_path, output_dir, model):
    """处理单帧图像，使用YOLO进行目标检测"""
    try:
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"无法读取图像: {image_path}, 错误: {e}")
        return None
    
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None
    
    height, width = image.shape[:2]
    filename = os.path.basename(image_path)
    timestamp_str = extract_timestamp_from_filename(filename)
    timestamp_sec = parse_timecode(timestamp_str)
    
    # 使用YOLO进行推理（降低置信度阈值以捕获更多目标）
    results = model(image, conf=0.15, iou=0.45, verbose=False)
    
    # 收集所有原始检测结果
    raw_detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id] if model.names else 'unknown'
            
            raw_detections.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'class': class_name, 'class_id': class_id,
                'confidence': confidence
            })
    
    # 按置信度从高到低排序
    raw_detections.sort(key=lambda d: -d['confidence'])
    
    # 去重：如果两个框的 IoU > 0.5，只保留置信度更高的
    def compute_iou(a, b):
        """计算两个框的 IoU"""
        inter_x1 = max(a['x1'], b['x1'])
        inter_y1 = max(a['y1'], b['y1'])
        inter_x2 = min(a['x2'], b['x2'])
        inter_y2 = min(a['y2'], b['y2'])
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        
        area_a = (a['x2'] - a['x1']) * (a['y2'] - a['y1'])
        area_b = (b['x2'] - b['x1']) * (b['y2'] - b['y1'])
        union = area_a + area_b - inter_area
        
        return inter_area / union if union > 0 else 0
    
    kept = []
    for det in raw_detections:
        duplicate = False
        for k in kept:
            if compute_iou(det, k) > 0.5:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    
    print(f"YOLO检测: {len(raw_detections)} 个, 去重后: {len(kept)} 个")
    
    # 保存去重后的裁剪图像
    detections = []
    expand_x, expand_y = 5, 5
    
    for idx, det in enumerate(kept):
        x1, y1 = det['x1'], det['y1']
        x2, y2 = det['x2'], det['y2']
        
        x1_exp = max(0, x1 - expand_x)
        y1_exp = max(0, y1 - expand_y)
        x2_exp = min(width, x2 + expand_x)
        y2_exp = min(height, y2 + expand_y)
        
        crop = image[y1_exp:y2_exp, x1_exp:x2_exp]
        
        crop_filename = f"{os.path.splitext(filename)[0]}_bbox_{idx:02d}.jpg"
        crop_path = os.path.join(output_dir, crop_filename)
        cv2.imencode('.jpg', crop)[1].tofile(crop_path)
        
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        detections.append({
            'object_id': idx,
            'class': det['class'],
            'class_id': det['class_id'],
            'confidence': det['confidence'],
            'bbox': [x1, y1, x2, y2],
            'bbox_expanded': [x1_exp, y1_exp, x2_exp, y2_exp],
            'crop_path': crop_path,
            'bbox_width': bbox_width,
            'bbox_height': bbox_height
        })
    
    frame_info = {
        'filename': filename,
        'image_path': image_path,
        'timestamp': timestamp_str,
        'timestamp_sec': timestamp_sec,
        'width': width,
        'height': height,
        'object_count': len(detections),
        'detections': detections
    }
    
    return frame_info

def generate_csv(data, output_path):
    """生成CSV文件"""
    fields = [
        'image_filename', 'object_id', 'class', 'class_id', 'confidence',
        'timestamp', 'timestamp_sec', 
        'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
        'bbox_width', 'bbox_height',
        'crop_path', 'image_width', 'image_height'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        
        for frame in data:
            for det in frame['detections']:
                row = {
                    'image_filename': frame['filename'],
                    'object_id': det['object_id'],
                    'class': det['class'],
                    'class_id': det['class_id'],
                    'confidence': det['confidence'],
                    'timestamp': frame['timestamp'],
                    'timestamp_sec': frame['timestamp_sec'],
                    'bbox_x1': det['bbox'][0],
                    'bbox_y1': det['bbox'][1],
                    'bbox_x2': det['bbox'][2],
                    'bbox_y2': det['bbox'][3],
                    'bbox_width': det['bbox_width'],
                    'bbox_height': det['bbox_height'],
                    'crop_path': det['crop_path'],
                    'image_width': frame['width'],
                    'image_height': frame['height']
                }
                writer.writerow(row)

def generate_json(data, output_path):
    """生成JSON文件"""
    output_data = {
        'total_frames': len(data),
        'total_objects': sum(frame['object_count'] for frame in data),
        'frames': data
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

def main():
    input_image_path = u"E:\\CODE_File\\YOLOB\\说明\\annotated_frames\\frame_000000_00-00-00-000.jpg"
    output_crops_dir = u"E:\\CODE_File\\YOLOB\\Video-RAG\\object_crops"
    output_csv_path = u"E:\\CODE_File\\YOLOB\\Video-RAG\\annotations.csv"
    output_json_path = u"E:\\CODE_File\\YOLOB\\Video-RAG\\annotations.json"
    model_path = u"E:\\CODE_File\\YOLOB\\Video-RAG\\yolov8n.pt"
    
    os.makedirs(output_crops_dir, exist_ok=True)
    
    # 清空之前的裁剪文件
    for f in os.listdir(output_crops_dir):
        os.remove(os.path.join(output_crops_dir, f))
    
    print(f"加载YOLO模型: {model_path}")
    model = YOLO(model_path)
    
    print(f"处理图像: {input_image_path}")
    frame_info = process_single_frame(input_image_path, output_crops_dir, model)
    
    if frame_info is None:
        print("处理失败")
        return
    
    print("生成CSV文件...")
    generate_csv([frame_info], output_csv_path)
    
    print("生成JSON文件...")
    generate_json([frame_info], output_json_path)
    
    print("\n处理完成！")
    print(f"裁剪图像: {output_crops_dir}")
    print(f"CSV: {output_csv_path}")
    print(f"JSON: {output_json_path}")
    
    print("\n检测结果:")
    print(f"图像: {frame_info['filename']}")
    print(f"尺寸: {frame_info['width']} x {frame_info['height']}")
    print(f"目标数量: {frame_info['object_count']}")
    for det in frame_info['detections']:
        print(f"  - [{det['object_id']}] {det['class']} (置信度: {det['confidence']:.2f}), bbox: {det['bbox']}, 尺寸: {det['bbox_width']}x{det['bbox_height']}")

if __name__ == "__main__":
    main()