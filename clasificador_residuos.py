import cv2
import time
import sys
from ultralytics import YOLO

UMBRAL_CONFIANZA = 0.70

COLORES_CLASE = {
    'plastic':    (255, 100,   0),
    'metal':      (150, 150, 150),
    'glass':      (  0, 200, 150),
    'cardboard':  (  0, 165, 255),
    'paper':      (  0, 165, 255),
    'organic':    (  0, 200,   0),
    'trash':      ( 80,  80,  80),
}

COLOR_DEFAULT = (0, 255, 0)

model = YOLO('yolov8n.pt')


def obtener_color(label):
    label_lower = label.lower()
    for clase, color in COLORES_CLASE.items():
        if clase in label_lower:
            return color
    return COLOR_DEFAULT


def dibujar_bbox(frame, x1, y1, x2, y2, label, conf):
    color = obtener_color(label)
    texto = f"{label} {conf:.0%}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    (ancho_txt, alto_txt), _ = cv2.getTextSize(
        texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )

    cv2.rectangle(
        frame,
        (x1, y1 - alto_txt - 14),
        (x1 + ancho_txt + 4, y1),
        color, -1
    )

    cv2.putText(
        frame, texto,
        (x1 + 2, y1 - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (255, 255, 255), 2
    )


def dibujar_hud(frame, tiempo_inf, fps, n_detecciones):
    y = 25

    cv2.putText(frame, f"Inferencia: {tiempo_inf:.1f} ms",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2)
    y += 25

    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 2)
    y += 25

    cv2.putText(frame, f"Detecciones: {n_detecciones}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 0), 2)
    y += 25

    cv2.putText(frame, f"Umbral: {UMBRAL_CONFIANZA:.0%}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 2)


def procesar_frame(frame):
    t_inicio = time.time()

    resultados = model(frame, conf=UMBRAL_CONFIANZA, verbose=False)

    n_detecciones = 0

    for r in resultados:
        if r.boxes is None:
            continue

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            label  = model.names[cls_id]

            if conf >= UMBRAL_CONFIANZA:
                dibujar_bbox(frame, x1, y1, x2, y2, label, conf)
                n_detecciones += 1

    tiempo_inf = (time.time() - t_inicio) * 1000
    fps = 1000 / tiempo_inf if tiempo_inf > 0 else 0

    return frame, tiempo_inf, fps, n_detecciones


def modo_imagen(ruta):
    frame = cv2.imread(ruta)

    if frame is None:
        print("[ERROR] Imagen no encontrada")
        return

    frame, t, fps, n = procesar_frame(frame)
    dibujar_hud(frame, t, fps, n)

    cv2.imshow("Resultado", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def modo_camara():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cámara no disponible")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, t, fps, n = procesar_frame(frame)
        dibujar_hud(frame, t, fps, n)

        cv2.imshow("Tiempo real", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        modo_imagen(sys.argv[1])
    else:
        modo_camara()