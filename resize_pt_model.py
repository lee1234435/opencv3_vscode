import torch
from ultralytics import YOLO

model = YOLO('opencv3_1\Quality.pt')  # YOLOv8 모델 로드

from torch.quantization import quantize_dynamic

# 모델 양자화
model_quantized = quantize_dynamic(model.model, {torch.nn.Linear}, dtype=torch.qint8)

torch.save(model_quantized.state_dict(), 'yolov8n_quantized.pt')
