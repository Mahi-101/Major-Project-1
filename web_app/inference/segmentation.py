from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

MODEL_PATH = Path(r"D:\B.Tech\CE-AI\Sem 7\MP\Code work\Yolo_seg\runs\segment\breed_segmentation\yolov8n_seg_optimized-2\weights\best.pt")
GATE_MODEL_PATH = Path(r"D:\B.Tech\CE-AI\Sem 7\MP\Code work\yolov8m.pt")
DEVICE = 0 if torch.cuda.is_available() else "cpu"
COW_CLASS_ID = 19
GATE_CONFIDENCE = 0.30


class SegmentationService:
    def __init__(self):
        self.model = YOLO(str(MODEL_PATH))
        self.gate_model = YOLO(str(GATE_MODEL_PATH))

    def _process_frame(self, frame):
        gate_result = self.gate_model.predict(
            frame,
            classes=[COW_CLASS_ID],
            conf=GATE_CONFIDENCE,
            device=DEVICE,
            verbose=False,
        )[0]
        if gate_result.boxes is None or len(gate_result.boxes) == 0:
            annotated = frame.copy()
            cv2.putText(
                annotated,
                "No cattle/buffalo detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )
            return annotated

        result = self.model.predict(frame, conf=0.35, iou=0.45, device=DEVICE, verbose=False)[0]
        return result.plot()

    def process_image(self, input_path: Path, output_path: Path):
        frame = cv2.imread(str(input_path))
        if frame is None:
            raise ValueError("Could not read the uploaded image.")
        if not cv2.imwrite(str(output_path), self._process_frame(frame)):
            raise ValueError("Could not save the annotated image.")

    def process_video(self, input_path: Path, output_path: Path, progress_callback):
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise ValueError("Could not open the uploaded video.")

        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Use temporary directory to store raw frames
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            processed = 0
            
            try:
                while True:
                    success, frame = capture.read()
                    if not success:
                        break
                    
                    annotated_frame = self._process_frame(frame)
                    
                    # Convert RGB to BGR
                    if len(annotated_frame.shape) == 3 and annotated_frame.shape[2] == 3:
                        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                    
                    # Ensure uint8
                    if annotated_frame.dtype != np.uint8:
                        annotated_frame = annotated_frame.astype(np.uint8)
                    
                    # Save individual frame
                    frame_path = temp_path / f"frame_{processed:06d}.png"
                    cv2.imwrite(str(frame_path), annotated_frame)
                    
                    processed += 1
                    percent = int(processed * 100 / total) if total else 0
                    progress_callback(percent, f"Annotated frame {processed} of {total or '?'}")
                
                capture.release()
                
                # Use FFmpeg to create video from frames
                progress_callback(95, "Encoding video...")
                input_pattern = str(temp_path / "frame_%06d.png")
                cmd = [
                    "ffmpeg",
                    "-framerate", str(fps),
                    "-i", input_pattern,
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-y",
                    str(output_path)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise ValueError(f"FFmpeg encoding failed: {result.stderr}")
                
                progress_callback(100, "Video encoding complete!")
                
            except Exception as e:
                capture.release()
                raise e
