import os
from dotenv import load_dotenv
load_dotenv()

PG_HOST = os.getenv("PG_HOST", "127.0.0.1")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DB   = os.getenv("PG_DB", "scoutis")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASS = os.getenv("PG_PASS", "postgres")

SECTORS = os.getenv("SCOUTIS_SECTORS", "S11,S12,S21,S22").split(",")

CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")  # 0 webcam, ou rtsp/http
TICK_SECONDS = float(os.getenv("TICK_SECONDS", "1.0"))

# treino
CALIBRATION_SECONDS = int(os.getenv("CALIBRATION_SECONDS", "60"))
SAMPLE_EVERY_N_FRAMES = int(os.getenv("SAMPLE_EVERY_N_FRAMES", "5"))

# modelo
MODEL_DIR = os.getenv("MODEL_DIR", "models")
MODEL_VERSION = os.getenv("MODEL_VERSION", "scoutis-v1")
DEVICE = os.getenv("DEVICE", "cpu")   # cpu/cuda
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
