import os
import json
import tinker
import cv2
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
from dotenv import load_dotenv
from tqdm import tqdm
from io import BytesIO

# Load environment variables
load_dotenv()


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
SAMPLER_PATH = "tinker://201b90b4-3773-5092-b9d0-1b640e673bcb:train:0/sampler_weights/004000"
BASE_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"

# Video processing config
VIDEO_INPUT_PATH = "testing/DJI_20241008145319_0002_W.MP4"
VIDEO_OUTPUT_PATH = "testing/output_visualization.mp4"
FRAME_SKIP = 10  # Process 1 in 10 frames


PROMPT = """
Analyze the aerial image from the UAV's camera and detect potential forest fires.
Answer the following questions based on the image analysis in a JSON schema.
If a question cannot be determined from the image answer with 'Cannot be determined'.

forest_fire_smoke_visible: ['Yes', 'No']
forest_fire_flames_visible: ['Yes', 'No']
confirm_uncontrolled_forest_fire: ['Yes', 'Closer investigation required', 'No forest fire visible']
fire_state: ['Ignition Phase', 'Growth Phase', 'Fully Developed Phase', 'Decay Phase', 'Cannot be determined', 'No forest fire visible']
fire_type: ['Ground Fire', 'Surface Fire', 'Crown Fire', 'Cannot be determined', 'No forest fire visible']
fire_intensity: ['Low', 'Moderate', 'High', 'Cannot be determined', 'No forest fire visible']
fire_size: ['Small', 'Medium', 'Large', 'Cannot be determined', 'No forest fire visible']
fire_hotspots: ['Multiple hotspots', 'One hotspot', 'Cannot be determined', 'No forest fire visible']
infrastructure_nearby: ['Yes', 'No', 'Cannot be determined', 'No forest fire visible']
people_nearby: ['Yes', 'No', 'Cannot be determined', 'No forest fire visible']
tree_vitality: ['Vital', 'Moderate Vitality', 'Declining', 'Dead', 'Cannot be determined', 'No forest fire visible']
"""


# ------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------
def run_inference(image_bytes: bytes, sampling_client, tokenizer) -> dict:
    """Run VLM inference on image bytes and return parsed JSON"""
    
    # Build multimodal prompt (image + text)
    model_input = tinker.ModelInput(
        chunks=[
            tinker.types.EncodedTextChunk(
                tokens=tokenizer.encode("<|im_start|>user\n<|vision_start|>")
            ),
            tinker.types.ImageChunk(data=image_bytes, format="jpeg"),
            tinker.types.EncodedTextChunk(
                tokens=tokenizer.encode(
                    f"<|vision_end|>{PROMPT}<|im_end|>\n<|im_start|>assistant\n"
                )
            ),
        ]
    )

    # Run sampling
    result_future = sampling_client.sample(
        prompt=model_input,
        num_samples=1,
        sampling_params=tinker.types.SamplingParams(
            max_tokens=512,
            temperature=0.0,   # deterministic JSON
        ),
    )

    # Wait for the future to complete and get the result
    result = result_future.result()

    # Decode model output
    output_text = tokenizer.decode(result.sequences[0].tokens)

    # Parse JSON
    try:
        # Remove markdown code fences if present
        cleaned_text = output_text
        if "```json" in cleaned_text:
            cleaned_text = cleaned_text.split("```json")[1].split("```")[0]
        elif "```" in cleaned_text:
            cleaned_text = cleaned_text.split("```")[1].split("```")[0]
        
        # Remove special tokens like <|im_end|>
        cleaned_text = cleaned_text.replace("<|im_end|>", "").strip()
        
        # Find JSON object
        json_start = cleaned_text.find("{")
        json_end = cleaned_text.rfind("}") + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = cleaned_text[json_start:json_end]
            json_data = json.loads(json_str)
            return json_data
        else:
            return {"error": "Could not find JSON object in output"}
    except Exception as e:
        return {"error": f"JSON parsing failed: {e}"}


def create_visualization_frame(rgb_frame: np.ndarray, json_data: dict) -> np.ndarray:
    """Create side-by-side visualization with JSON on left and RGB frame on right"""
    
    height, width = rgb_frame.shape[:2]
    
    # Create left panel (dark background for JSON text)
    left_panel = np.zeros((height, width, 3), dtype=np.uint8)
    left_panel[:] = (30, 30, 30)  # Dark gray background
    
    # Render JSON data as text
    y_offset = 50
    line_height = 45
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    text_color = (255, 255, 255)  # White text
    
    # Title
    cv2.putText(left_panel, "Forest Fire Analysis", (10, y_offset), 
                font, 1.0, (100, 200, 255), 3)
    y_offset += line_height + 10
    
    # Render each JSON field
    for key, value in json_data.items():
        # Format key (replace underscores with spaces, capitalize)
        display_key = key.replace("_", " ").title()
        
        # Truncate long values
        value_str = str(value)
        if len(value_str) > 30:
            value_str = value_str[:27] + "..."
        
        # Render key-value pair
        text = f"{display_key}:"
        cv2.putText(left_panel, text, (10, y_offset), 
                    font, font_scale, (150, 150, 150), font_thickness)
        
        # Render value with color coding
        value_color = text_color
        if value == "Yes" or "visible" in value_str.lower():
            value_color = (100, 255, 100)  # Green
        elif value == "No":
            value_color = (200, 200, 200)  # Light gray
        elif "Cannot be determined" in value_str:
            value_color = (100, 150, 255)  # Light blue
        
        cv2.putText(left_panel, value_str, (20, y_offset + 20), 
                    font, font_scale, value_color, font_thickness)
        
        y_offset += line_height + 15
        
        # Break if we run out of space
        if y_offset > height - 40:
            break
    
    # Convert RGB frame to BGR for OpenCV
    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    
    # Combine panels horizontally
    combined = np.hstack([left_panel, bgr_frame])
    
    return combined


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    
    # Check if video exists
    if not os.path.exists(VIDEO_INPUT_PATH):
        print(f"❌ Error: Video file not found at {VIDEO_INPUT_PATH}")
        return
    
    print("=" * 80)
    print("Forest Fire Detection - Video Processing")
    print("=" * 80)
    print(f"Input Video: {VIDEO_INPUT_PATH}")
    print(f"Output Video: {VIDEO_OUTPUT_PATH}")
    print(f"Processing: 1 in {FRAME_SKIP} frames")
    print("=" * 80)
    
    # Load tokenizer for correct special tokens
    print("\n📦 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Create Tinker service + sampling client from checkpoint
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(
        model_path=SAMPLER_PATH,
        base_model=BASE_MODEL,
    )
    print("✅ Model loaded successfully")

    # Open video file
    print(f"\n🎥 Opening video: {VIDEO_INPUT_PATH}")
    cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
    
    if not cap.isOpened():
        print("❌ Error: Could not open video file")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📊 Video Info:")
    print(f"   - Resolution: {frame_width}x{frame_height}")
    print(f"   - FPS: {fps}")
    print(f"   - Total Frames: {total_frames}")
    print(f"   - Frames to Process: {total_frames // FRAME_SKIP}")
    
    # Create video writer (double width for side-by-side)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_OUTPUT_PATH, fourcc, fps, 
                          (frame_width * 2, frame_height))
    
    if not out.isOpened():
        print("❌ Error: Could not create output video writer")
        cap.release()
        return
    
    print(f"\n🚀 Starting video processing...")
    
    # Process frames with progress bar
    frame_count = 0
    processed_count = 0
    last_json_data = {"status": "Initializing..."}
    
    with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Process every FRAME_SKIP frames
            if frame_count % FRAME_SKIP == 0:
                try:
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert frame to JPEG bytes
                    pil_image = Image.fromarray(rgb_frame)
                    buffer = BytesIO()
                    pil_image.save(buffer, format="JPEG", quality=95)
                    image_bytes = buffer.getvalue()
                    
                    # Run inference
                    json_data = run_inference(image_bytes, sampling_client, tokenizer)
                    last_json_data = json_data
                    processed_count += 1
                    
                except Exception as e:
                    print(f"\n⚠️ Warning: Failed to process frame {frame_count}: {e}")
                    json_data = last_json_data  # Use last successful result
            else:
                # Use last JSON data for skipped frames
                json_data = last_json_data
            
            # Create visualization
            vis_frame = create_visualization_frame(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                json_data
            )
            
            # Write frame
            out.write(vis_frame)
            
            frame_count += 1
            pbar.update(1)
    
    # Release resources
    cap.release()
    out.release()
    
    print(f"\n✅ Processing complete!")
    print(f"📊 Statistics:")
    print(f"   - Total frames: {frame_count}")
    print(f"   - Frames analyzed: {processed_count}")
    print(f"   - Output saved to: {VIDEO_OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
