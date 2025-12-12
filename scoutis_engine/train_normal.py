from __future__ import annotations

import os
import time
import json
import cv2
import numpy as np
import torch

from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from config import (
    CAMERA_SOURCE,
    CALIBRATION_SECONDS,
    SAMPLE_EVERY_N_FRAMES,
    MODEL_DIR,
    MODEL_VERSION,
    DEVICE
)

from features import FeatureExtractor
from ml_model import EmbeddingAutoEncoder


# ============================================================
# CAMERA IP ROBUSTA
# ============================================================
def open_cam(src: str):
    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"[TRAIN] Não abriu câmera: {src}")
    return cap


def read_frame(cap, src: str, retries: int = 3, wait_s: float = 1.5):
    ok, frame = cap.read()
    if ok and frame is not None:
        return cap, True, frame

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
# MAIN TRAIN
# ============================================================
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\n[SCOUTIS] TRAIN — Calibração do NORMAL")
    print(f"Camera: {CAMERA_SOURCE}")
    print(f"Tempo: {CALIBRATION_SECONDS}s | Sample every: {SAMPLE_EVERY_N_FRAMES} frames")
    print(f"Device: {DEVICE}")
    print("⚠️  Garanta que o ambiente esteja NORMAL durante o treino.")
    print("Pressione Q para encerrar antecipadamente.\n")

    fx = FeatureExtractor(device=DEVICE)
    cap = open_cam(str(CAMERA_SOURCE))

    embeddings = []
    frame_count = 0
    start = time.time()

    try:
        while time.time() - start < CALIBRATION_SECONDS:
            cap, ok, frame = read_frame(cap, str(CAMERA_SOURCE))
            if not ok:
                print("[WARN] Falha ao ler frame durante treino.")
                continue

            frame_count += 1
            if frame_count % SAMPLE_EVERY_N_FRAMES != 0:
                continue

            emb = fx.embed(frame)
            embeddings.append(emb)

            # overlay de feedback (não afeta treino)
            cv2.putText(
                frame,
                f"Calibrando NORMAL | amostras: {len(embeddings)}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2
            )
            cv2.imshow("SCOUTIS | Train (Normal)", frame)

            if (cv2.waitKey(1) & 0xFF) in (ord('q'), ord('Q')):
                print("[TRAIN] Encerrado manualmente.")
                break

    finally:
        try:
            cap.release()
        except:
            pass
        try:
            cv2.destroyAllWindows()
        except:
            pass

    # ========================================================
    # VALIDAÇÃO DE AMOSTRAS
    # ========================================================
    if len(embeddings) < 80:
        raise RuntimeError(
            f"[ERRO] Poucas amostras ({len(embeddings)}). "
            "Aumente CALIBRATION_SECONDS ou reduza SAMPLE_EVERY_N_FRAMES."
        )

    X = torch.tensor(np.stack(embeddings), dtype=torch.float32)
    dim = X.shape[1]

    print(f"[TRAIN] Amostras coletadas: {len(embeddings)} | Dimensão: {dim}")

    # ========================================================
    # TREINO DO AUTOENCODER (REAL)
    # ========================================================
    model = EmbeddingAutoEncoder(dim).to(DEVICE)
    opt = Adam(model.parameters(), lr=1e-3)

    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model.train()
    epochs = 16
    print("[TRAIN] Treinando AutoEncoder...")

    for ep in range(epochs):
        losses = []
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            recon = model(xb)
            loss = torch.mean((recon - xb) ** 2)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

        print(f"Epoch {ep+1}/{epochs} | loss={np.mean(losses):.6f}")

    # ========================================================
    # THRESHOLD REAL (estatístico)
    # ========================================================
    model.eval()
    with torch.no_grad():
        recon = model(X.to(DEVICE)).cpu()
        errors = torch.mean((recon - X) ** 2, dim=1).numpy()

    # threshold robusto (percentil alto do NORMAL)
    threshold = float(np.percentile(errors, 99.5))

    # ========================================================
    # SALVAMENTO
    # ========================================================
    meta = {
        "model_version": MODEL_VERSION,
        "dim": int(dim),
        "threshold": threshold,
        "calibration_samples": int(len(embeddings)),
        "calibration_seconds": int(CALIBRATION_SECONDS),
        "sample_every_n_frames": int(SAMPLE_EVERY_N_FRAMES),
        "device": DEVICE,
        "created_at": int(time.time()),
    }

    ae_path = os.path.join(MODEL_DIR, "ae.pt")
    meta_path = os.path.join(MODEL_DIR, "meta.json")

    torch.save(model.state_dict(), ae_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n[SCOUTIS] TRAIN FINALIZADO COM SUCESSO")
    print(f"→ Modelo salvo em: {ae_path}")
    print(f"→ Meta salvo em: {meta_path}")
    print(f"→ Threshold aprendido: {threshold:.8f}")
    print("Agora você pode rodar: python engine_real.py\n")


if __name__ == "__main__":
    main()
