from __future__ import annotations

import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from inference.classification import ClassificationService
from inference.segmentation import SegmentationService
from inference.segmentation_classification import SegmentationClassificationService

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

ALLOWED_IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
ALLOWED_VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.config["UPLOAD_DIR"] = UPLOAD_DIR
app.config["RESULT_DIR"] = RESULT_DIR

classification_service = ClassificationService()
segmentation_service = SegmentationService()
segmentation_classification_service = SegmentationClassificationService()
video_jobs: dict[str, dict] = {}
video_jobs_lock = threading.Lock()


def extension_allowed(filename: str, allowed: set[str]) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


def save_upload(file_storage):
    original_name = secure_filename(file_storage.filename or "")
    if not original_name:
        raise ValueError("Please choose a file.")
    unique_name = f"{uuid.uuid4().hex}_{original_name}"
    path = UPLOAD_DIR / unique_name
    file_storage.save(path)
    return path


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/segment-then-classify")
def segment_then_classify_page():
    return render_template("segmentation_classification_upload.html")


@app.post("/segment-then-classify")
def segment_then_classify():
    uploaded = request.files.get("file")
    if uploaded is None or not extension_allowed(uploaded.filename or "", ALLOWED_IMAGE_EXTENSIONS):
        return render_template("segmentation_classification_upload.html", error="Please upload a supported image file."), 400

    try:
        input_path = save_upload(uploaded)
        result_id = uuid.uuid4().hex
        annotated_path = RESULT_DIR / f"{result_id}_seg_classified.jpg"
        crop_dir = RESULT_DIR / f"{result_id}_crops"
        result = segmentation_classification_service.process_image(input_path, annotated_path, crop_dir)
        return render_template("segmentation_classification.html", result=result,
                               annotated_name=annotated_path.name, crop_dir=crop_dir.name)
    except Exception as exc:
        return render_template("segmentation_classification_upload.html", error=f"Pipeline failed: {exc}"), 500


@app.post("/classify")
def classify():
    uploaded = request.files.get("file")
    if uploaded is None or not extension_allowed(uploaded.filename or "", ALLOWED_IMAGE_EXTENSIONS):
        return render_template("index.html", error="Please upload a supported image file.", selected_task="classification"), 400

    try:
        input_path = save_upload(uploaded)
        result = classification_service.predict(input_path)
        return render_template("classification_result.html", result=result)
    except Exception as exc:
        return render_template("index.html", error=f"Classification failed: {exc}", selected_task="classification"), 500


@app.post("/segment")
def segment():
    uploaded = request.files.get("file")
    filename = uploaded.filename or "" if uploaded else ""
    if uploaded is None or not (extension_allowed(filename, ALLOWED_IMAGE_EXTENSIONS) or extension_allowed(filename, ALLOWED_VIDEO_EXTENSIONS)):
        return render_template("index.html", error="Please upload a supported image or video file.", selected_task="segmentation"), 400

    try:
        input_path = save_upload(uploaded)
        if extension_allowed(filename, ALLOWED_IMAGE_EXTENSIONS):
            output_path = RESULT_DIR / f"{input_path.stem}_segmented.jpg"
            segmentation_service.process_image(input_path, output_path)
            return render_template("segmentation_result.html", media_type="image", media_name=output_path.name)

        job_id = uuid.uuid4().hex
        with video_jobs_lock:
            video_jobs[job_id] = {"status": "queued", "progress": 0, "message": "Waiting to start..."}
        output_path = RESULT_DIR / f"{job_id}_segmented.mp4"
        thread = threading.Thread(target=run_video_job, args=(job_id, input_path, output_path), daemon=True)
        thread.start()
        return render_template("video_progress.html", job_id=job_id)
    except Exception as exc:
        return render_template("index.html", error=f"Segmentation failed: {exc}", selected_task="segmentation"), 500


def run_video_job(job_id: str, input_path: Path, output_path: Path):
    def progress_callback(percent: int, message: str):
        with video_jobs_lock:
            video_jobs[job_id].update(progress=percent, message=message)

    try:
        with video_jobs_lock:
            video_jobs[job_id]["status"] = "processing"
        segmentation_service.process_video(input_path, output_path, progress_callback)
        with video_jobs_lock:
            video_jobs[job_id].update(status="complete", progress=100, message="Video processing complete.", result=output_path.name)
    except Exception as exc:
        with video_jobs_lock:
            video_jobs[job_id].update(status="error", message=str(exc))


@app.get("/progress/<job_id>")
def progress(job_id: str):
    with video_jobs_lock:
        job = video_jobs.get(job_id)
    if job is None:
        return jsonify({"status": "error", "message": "Job not found."}), 404
    return jsonify(job)


@app.get("/results/<path:filename>")
def result_file(filename: str):
    return send_from_directory(RESULT_DIR, filename)


@app.get("/done/<job_id>")
def video_done(job_id: str):
    with video_jobs_lock:
        job = video_jobs.get(job_id)
    if not job or job.get("status") != "complete":
        return redirect(url_for("index"))
    return render_template("segmentation_result.html", media_type="video", media_name=job["result"])


@app.errorhandler(413)
def too_large(_error):
    return render_template("index.html", error="The file is too large. Maximum size is 500 MB."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
