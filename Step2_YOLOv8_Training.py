import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO
import torch

print("CUDA available:", torch.cuda.is_available())

data_path = r'C:/Users/PC/Documents/GitHub/Project 3 Data/data/data.yaml'

model = YOLO('yolov8n.pt')

try:
    results = model.train(
        data=data_path,       # Path to data configuration
        epochs=150,           # Number of training epochs (within recommended range)
        batch=16,             # Batch size (adjust based on your system)
        imgsz=928,            # Update to nearest multiple of 32 (as suggested in warning)
        name='pcb_component_detection',  # Name of the model
        device='cpu',          # Force CPU usage
        workers=0             # Set workers to 0 to avoid multiprocessing issues
    )

    model.save('best_pcb_component_model.pt')

    print("Training completed. Model saved.")

except Exception as e:
    print(f"An error occurred during training: {e}")