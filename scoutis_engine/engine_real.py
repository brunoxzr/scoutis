from __future__ import annotations

import os
import json
import time
import numpy as np
import cv2
import torch

from ultralytics import YOLO

from config import (
    SECTORS, CAMERA_SOURCE, TICK_SECONDS,
    MODEL_DIR, MODEL_VERSION, DEVICE, YOLO_MODEL
)
from db import get_conn, apply_schema, seed_sectors, insert_metric
from features import FeatureExtractor
from ml_model import EmbeddingAutoEncoder


# ============================================================
# CONFIG: ROIs (1 câmera -> 4 setores)
# Ajuste depois (pode virar calibração por clique)
# ROI = (x, y, w, h) em percentual 0..1
# ============================================================
ROIS = {
    "S11": (0.00, 0.00, 0.50, 0.50),
    "S12": (0.50, 0.00, 0.50, 0.50),
    "S21": (0.00, 0.50, 0.50, 0.50),
    "S22": (0.50, 0.50, 0.50, 0.50),
}


# ============================================================
# CÂMERA IP: open + robust read + reconnection
# ============================================================
def open_cam(src: str):
    # IP cam: usa FFMPEG; buffersize baixo reduz atraso
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Não abriu câmera IP: {src}")
    return cap

def read_frame(cap, src: str, retries: int = 3, wait_s: float = 1.5):
    """
    Se falhar, reconecta.
    Retorna (cap, ok, frame)
    """
    ok, frame = cap.read()
    if ok and frame is not None:
        return cap, True, frame

    # tenta reconectar
    for _ in range(retries):
        try:
            cap.release()
        except:
            pass
        time.sleep(wait_s)
        try:
            cap = open_cam(src)
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap, True, frame
        except:
            continue

    return cap, False, None


# ============================================================
# ROI crop (percentual)
# ============================================================
def crop_roi(frame, roi):
    h, w = frame.shape[:2]
    x, y, rw, rh = roi
    x1 = max(0, int(x * w))
    y1 = max(0, int(y * h))
    x2 = min(w, int((x + rw) * w))
    y2 = min(h, int((y + rh) * h))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


# ============================================================
# LOAD ML MODEL (AE) + threshold learned
# ============================================================
def load_ae_and_threshold():
    meta_path = os.path.join(MODEL_DIR, "meta.json")
    ae_path = os.path.join(MODEL_DIR, "ae.pt")
    if not os.path.exists(meta_path) or not os.path.exists(ae_path):
        raise RuntimeError(
            f"Modelo não encontrado em {MODEL_DIR}. Rode train_normal.py antes "
            "para gerar ae.pt e meta.json."
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    dim = int(meta["dim"])
    threshold = float(meta["threshold"])

    ae = EmbeddingAutoEncoder(dim).to(DEVICE)
    ae.load_state_dict(torch.load(ae_path, map_location=DEVICE))
    ae.eval()

    return ae, threshold, meta


def reconstruction_error(ae: EmbeddingAutoEncoder, emb: np.ndarray) -> float:
    x = torch.tensor(emb[None, :], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        recon = ae(x)
        err = torch.mean((recon - x) ** 2).item()
    return float(err)


# ============================================================
# YOLO counts (objetivo e real sem dataset custom)
# -> plantas/falhas perfeitos exigem treino próprio do seu cenário.
# -> aqui: detecta "objetos relevantes" (pessoas/veículos/animais)
# ============================================================
COCO_OBJ_IDS = {
    # person, car, motorcycle, bus, truck
    0, 2, 3, 5, 7,
    # bird, cat, dog, horse, sheep, cow
    14, 15, 16, 17, 18, 19
}

def yolo_counts(result) -> dict:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return {"plantas": 0, "falhas": 0, "objetos": 0}

    cls = boxes.cls.detach().cpu().numpy().astype(int)
    objetos = int(np.sum(np.isin(cls, list(COCO_OBJ_IDS))))
    return {"plantas": 0, "falhas": 0, "objetos": objetos}


# ============================================================
# DECISÃO + "possível causa" (sem vibe IA)
# ============================================================
def decide(anom: float, thr: float, counts: dict):
    # estabilidade (0..1): quanto maior, mais dentro do padrão
    # mapeamento robusto (não explode com valores)
    ratio = anom / (thr + 1e-9)
    stability = float(np.clip(1.0 - (ratio - 1.0) * 0.55, 0.0, 1.0))
    health = int(np.clip(stability * 100.0, 0, 100))

    objetos = int(counts.get("objetos", 0))

    if anom >= thr * 1.35 and objetos >= 2:
        status = "ANOMALIA"
        cause = "Mudança persistente fora do padrão com interferência detectada. Recomenda-se inspeção no setor."
    elif anom >= thr * 1.35:
        status = "ANOMALIA"
        cause = "Mudança persistente fora do padrão. Recomenda-se inspeção no setor."
    elif anom >= thr and objetos >= 1:
        status = "ATENCAO"
        cause = "Mudança leve de padrão com interferência pontual detectada. Verifique rapidamente o setor."
    elif anom >= thr:
        status = "ATENCAO"
        cause = "Mudança leve de padrão. Pode estar ligada a variação do ambiente ou manejo recente."
    else:
        status = "NORMAL"
        cause = "Dentro do padrão esperado para a área."

    return health, stability, status, cause


# ============================================================
# MAIN ENGINE
# ============================================================
def main():
    # --- DB ---
    conn = get_conn()
    apply_schema(conn, "schema.sql")
    seed_sectors(conn, SECTORS)

    # garante que os setores do ROI existam no DB (se estiverem em ROIS)
    # (SECTORS pode ser diferente; aqui a gente prioriza ROIS)
    for sid in ROIS.keys():
        if sid not in SECTORS:
            # se quiser, mantenha SECTORS = S11,S12,S21,S22 pra ficar consistente
            pass

    # --- Models ---
    ae, thr, meta = load_ae_and_threshold()
    fx = FeatureExtractor(device=DEVICE)
    yolo = YOLO(YOLO_MODEL)

    # --- Camera ---
    cap = open_cam(str(CAMERA_SOURCE))

    print("\n[SCOUTIS] Engine REAL (IP Camera) ON")
    print(f"camera: {CAMERA_SOURCE}")
    print(f"device: {DEVICE} | yolo: {YOLO_MODEL}")
    print(f"AE threshold: {thr:.8f} | model_version: {MODEL_VERSION}")
    print("Pressione Q para sair.\n")

    # debug window (você pode desligar no pitch)
    show_debug = True

    try:
        while True:
            cap, ok, frame = read_frame(cap, str(CAMERA_SOURCE))
            if not ok:
                print("[WARN] Falha ao ler frame. Tentando reconectar...")
                continue

            # processa ROIs (cada ROI = setor)
            for sector_id, roi in ROIS.items():
                crop = crop_roi(frame, roi)
                if crop is None or crop.size == 0:
                    continue

                # 1) ML real (embedding -> AE -> erro)
                emb = fx.embed(crop)
                anom = reconstruction_error(ae, emb)

                # 2) YOLO real
                res = yolo.predict(source=crop, verbose=False)[0]
                counts = yolo_counts(res)

                # 3) decisão
                health, stability, status, cause = decide(anom, thr, counts)

                # 4) grava no PG
                insert_metric(conn, {
                    "sector_code": sector_id,
                    "health": health,
                    "stability_score": stability,
                    "anomaly_score": anom,
                    "anomaly_threshold": thr,
                    "yolo_counts": counts,
                    "status": status,
                    "cause": cause,
                    "model_version": MODEL_VERSION,
                    "camera_source": str(CAMERA_SOURCE),
                })

                # debug overlay (opcional)
                if show_debug:
                    h, w = frame.shape[:2]
                    x, y, rw, rh = roi
                    x1 = int(x * w); y1 = int(y * h)
                    x2 = int((x + rw) * w); y2 = int((y + rh) * h)

                    color = (34,197,94) if status == "NORMAL" else (251,191,36) if status == "ATENCAO" else (239,68,68)
                    color = (int(color[2]), int(color[1]), int(color[0]))  # RGB->BGR

                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(frame, f"{sector_id} {status} H{health}% A{anom:.4g}",
                                (x1+8, y1+22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

            if show_debug:
                cv2.imshow("SCOUTIS | Engine (debug)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q')):
                    break

            time.sleep(TICK_SECONDS)

    finally:
        try:
            cap.release()
        except:
            pass
        try:
            cv2.destroyAllWindows()
        except:
            pass
        conn.close()
        print("\n[SCOUTIS] Engine OFF")


if __name__ == "__main__":
    main()
