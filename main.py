import cv2
import time
import sys
from ultralytics import YOLO

# ─────────────────────────────────────
# ⚙️ CONFIGURACIÓN
# ─────────────────────────────────────

UMBRAL_CONFIANZA = 0.5

# ─────────────────────────────────────
# 🤖 CARGA DEL MODELO
# ─────────────────────────────────────

# Modelo general (se descarga automáticamente)
model = YOLO('yolov8n.pt')
print('[OK] Modelo cargado correctamente')

# ─────────────────────────────────────
# 🎨 FUNCIONES DE RENDER
# ─────────────────────────────────────

def dibujar_bbox(frame, x1, y1, x2, y2, label, conf):
    """Dibuja bounding box con etiqueta"""
    color = (0, 255, 0)
    texto = f"{label} {conf:.0%}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, texto, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def dibujar_hud(frame, tiempo_inf, fps):
    """Overlay tipo interfaz"""
    y = 25

    cv2.putText(frame, f"Inferencia: {tiempo_inf:.1f} ms",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2)

    y += 25

    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 2)


# ─────────────────────────────────────
# 🔍 PROCESAMIENTO
# ─────────────────────────────────────

def procesar_frame(frame):
    t_inicio = time.time()

    resultados = model(frame, conf=UMBRAL_CONFIANZA, verbose=False)

    for r in resultados:
        if r.boxes is None:
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            label = model.names[cls_id]

            dibujar_bbox(frame, x1, y1, x2, y2, label, conf)

    tiempo_inf = (time.time() - t_inicio) * 1000
    fps = 1000 / tiempo_inf if tiempo_inf > 0 else 0

    return frame, tiempo_inf, fps


# ─────────────────────────────────────
# 🖼️ MODO IMAGEN
# ─────────────────────────────────────

def modo_imagen(ruta):
    frame = cv2.imread(ruta)

    if frame is None:
        print(f"[ERROR] No se encontró la imagen: {ruta}")
        return

    frame, tiempo_inf, fps = procesar_frame(frame)
    dibujar_hud(frame, tiempo_inf, fps)

    cv2.imshow("YOLO - Imagen", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────
# 🎥 MODO CÁMARA
# ─────────────────────────────────────

def modo_camara():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara")
        return

    print("[INFO] Presiona 'q' para salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, tiempo_inf, fps = procesar_frame(frame)
        dibujar_hud(frame, tiempo_inf, fps)

        cv2.imshow("YOLO - Camara", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────
# 🚀 MAIN
# ─────────────────────────────────────

if __name__ == "__main__":

    if len(sys.argv) > 1:
        modo_imagen(sys.argv[1])
    else:
        modo_camara()