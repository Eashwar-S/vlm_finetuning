"""
Fire and Smoke Detection Script using YOLO
Processes video frames to detect fire and smoke, draws bounding boxes, and saves the output.
"""

import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
import numpy as np

def detect_fire_smoke_in_video(
    model_path: str,
    video_path: str,
    output_path: str,
    conf_threshold: float = 0.25,
    device: str = None
):
    """
    Detect fire and smoke in video frames using YOLO model.
    
    Args:
        model_path: Path to the YOLO model (.pt file)
        video_path: Path to input video file
        output_path: Path to save output video
        conf_threshold: Confidence threshold for detections
        device: Device to run inference on ('cuda' or 'cpu'). Auto-detects if None.
    """
    
    # Auto-detect device if not specified
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Load YOLO model
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    model.to(device)
    
    # Open video
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise ValueError(f"Could not create output video file: {output_path}")
    
    print(f"Output video will be saved to: {output_path}")
    
    # Define colors for bounding boxes (BGR format)
    colors = {
        'fire': (0, 0, 255),    # Red
        'smoke': (128, 128, 128)  # Gray
    }
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Run YOLO inference
            results = model.predict(
                frame,
                conf=conf_threshold,
                device=device,
                verbose=False
            )
            
            # Process detections
            if len(results) > 0:
                result = results[0]
                
                # Get boxes, classes, and confidences
                boxes = result.boxes
                
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        
                        # Get class and confidence
                        cls_id = int(box.cls[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        
                        # Get class name
                        class_name = model.names[cls_id].lower()
                        
                        # Determine color based on class
                        color = colors.get(class_name, (0, 255, 0))  # Default to green
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Create label
                        label = f"{class_name}: {conf:.2f}"
                        
                        # Get label size for background
                        (label_width, label_height), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                        )
                        
                        # Draw label background
                        cv2.rectangle(
                            frame,
                            (x1, y1 - label_height - baseline - 5),
                            (x1 + label_width, y1),
                            color,
                            -1
                        )
                        
                        # Draw label text
                        cv2.putText(
                            frame,
                            label,
                            (x1, y1 - baseline - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2
                        )
            
            # Write frame to output video
            out.write(frame)
            
            # Print progress
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
    
    finally:
        # Release resources
        cap.release()
        out.release()
        print(f"\nProcessing complete! Processed {frame_count} frames.")
        print(f"Output saved to: {output_path}")


def main():
    """Main function to run fire and smoke detection."""
    
    # Define paths
    model_path = "10-40K-100e-l.pt"
    video_path = "testing/DJI_20241008145319_0002_W.MP4"
    output_path = "testing/DJI_20241008145319_0002_W_detected.mp4"
    
    # Verify paths exist
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Run detection
    detect_fire_smoke_in_video(
        model_path=model_path,
        video_path=video_path,
        output_path=output_path,
        conf_threshold=0.25,  # Adjust confidence threshold as needed
        device='cuda'  # Force GPU usage
    )


if __name__ == "__main__":
    main()
