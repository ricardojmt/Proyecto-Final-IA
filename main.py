import cv2
import time
import sys
from ultralytics import YOLO

# ─────────────────────────────────────
# ⚙️ CONFIGURACIÓN
# ─────────────────────────────────────

# AVANCE 1: umbral ajustado según diseño del agente
# (anteriormente 0.5 - corregido a 0.70)
UMBRAL_CONFIANZA = 0.70

# Mapeo de colores por clase de residuo (BGR)
# En Avance 1 el modelo general no detecta estas clases,
# pero el mapeo queda listo para cuando se use TrashNet
COLORES_CLASE = {
    'plastic':    (255, 100,   0),   # Azul
    'metal':      (150, 150, 150),   # Gris
    'glass':      (  0, 200, 150),   # Verde azulado
    'cardboard':  (  0, 165, 255),   # Naranja
    'paper':      (  0, 165, 255),   # Naranja (igual que cartón)
    'organic':    (  0, 200,   0),   # Verde
    'trash':      ( 80,  80,  80),   # Gris oscuro (no reciclable)
}

# Color por defecto si la clase no está en el mapeo
COLOR_DEFAULT = (0, 255, 0)

# ─────────────────────────────────────
# 🤖 CARGA DEL MODELO
# ─────────────────────────────────────

# AVANCE 1: modelo general como prueba de concepto.
# En fases posteriores se reemplaza por modelo fine-tuneado
# con TrashNet: model = YOLO('modelo_residuos.pt')
model = YOLO('yolov8n.pt')
print('[OK] Modelo cargado correctamente')
print('[INFO] Avance 1 - modelo general (prueba de concepto)')
print('[INFO] Clases disponibles:', list(model.names.values())[:10], '...')

# ─────────────────────────────────────
# 🎨 FUNCIONES DE RENDER
# ─────────────────────────────────────

def obtener_color(label):
    """Retorna el color BGR según la clase detectada"""
    label_lower = label.lower()
    for clase, color in COLORES_CLASE.items():
        if clase in label_lower:
            return color
    return COLOR_DEFAULT


def dibujar_bbox(frame, x1, y1, x2, y2, label, conf):
    """Dibuja bounding box con etiqueta y color por clase"""
    color = obtener_color(label)
    texto = f"{label} {conf:.0%}"

    # Caja del objeto
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Fondo de la etiqueta para legibilidad
    (ancho_txt, alto_txt), _ = cv2.getTextSize(
        texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    cv2.rectangle(
        frame,
        (x1, y1 - alto_txt - 14),
        (x1 + ancho_txt + 4, y1),
        color, -1
    )

    # Texto de la etiqueta
    cv2.putText(
        frame, texto,
        (x1 + 2, y1 - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        (255, 255, 255), 2
    )


def dibujar_hud(frame, tiempo_inf, fps, n_detecciones):
    """Overlay tipo interfaz con métricas del sistema"""
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

    # Indicador de umbral de confianza activo
    cv2.putText(frame, f"Umbral conf.: {UMBRAL_CONFIANZA:.0%}",
                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 2)


# ─────────────────────────────────────
# 🔍 PROCESAMIENTO
# ─────────────────────────────────────

def procesar_frame(frame):
    """Ejecuta inferencia YOLO y dibuja resultados en el frame"""
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

            # Solo mostrar si supera el umbral de confianza
            # (YOLO ya filtra internamente, pero lo validamos
            # explícitamente para coherencia con el diseño del agente)
            if conf >= UMBRAL_CONFIANZA:
                dibujar_bbox(frame, x1, y1, x2, y2, label, conf)
                n_detecciones += 1
                print(f"  → Clase: {label:<12} | Confianza: {conf:.2f}")
            else:
                # Detección de baja confianza → revisión manual
                print(f"  [BAJA CONF] {label} ({conf:.2f}) → no identificado")

    tiempo_inf = (time.time() - t_inicio) * 1000
    fps = 1000 / tiempo_inf if tiempo_inf > 0 else 0

    return frame, tiempo_inf, fps, n_detecciones


# ─────────────────────────────────────
# 🖼️ MODO IMAGEN
# ─────────────────────────────────────

def modo_imagen(ruta):
    """Procesa una imagen estática y muestra el resultado"""
    frame = cv2.imread(ruta)

    if frame is None:
        print(f"[ERROR] No se encontró la imagen: {ruta}")
        return

    print(f"\n[INFO] Procesando imagen: {ruta}")
    frame, tiempo_inf, fps, n_det = procesar_frame(frame)
    dibujar_hud(frame, tiempo_inf, fps, n_det)

    print(f"[RESULTADO] Detecciones: {n_det} | Inferencia: {tiempo_inf:.1f} ms")

    cv2.imshow("Clasificador de Residuos - Imagen", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────
# 🎥 MODO CÁMARA
# ─────────────────────────────────────

def modo_camara():
    """Detección en tiempo real desde cámara web"""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara")
        return

    print("[INFO] Cámara activa. Presiona 'q' para salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] No se pudo leer el frame")
            break

        frame, tiempo_inf, fps, n_det = procesar_frame(frame)
        dibujar_hud(frame, tiempo_inf, fps, n_det)

        cv2.imshow("Clasificador de Residuos - Tiempo Real", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Saliendo...")
            break

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────
# 🚀 MAIN
# ─────────────────────────────────────

if __name__ == "__main__":
    print("=" * 45)
    print("  Clasificador de Residuos - Avance 1")
    print("  Método: YOLOv8 | Umbral:", UMBRAL_CONFIANZA)
    print("=" * 45)

    if len(sys.argv) > 1:
        # Uso: python clasificador_residuos.py imagen.jpg
        modo_imagen(sys.argv[1])
    else:
        # Sin argumentos → modo cámara en tiempo real
        modo_camara()