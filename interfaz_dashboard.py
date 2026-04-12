import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import clasificador_residuos as core

# 🎨 COLORES
BG = "#f1f5f9"
CARD = "#ffffff"
VIDEO_BG = "#0f172a"
TEXT = "#0f172a"
GREEN = "#22c55e"

ORANGE = "#f97316"
BLUE = "#3b82f6"
GRAY = "#9ca3af"
RED = "#ef4444"

VIDEO_W = 640
VIDEO_H = 400


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificador Inteligente de Residuos")
        self.root.geometry("1100x700")
        self.root.configure(bg=BG)

        self.cap = None
        self.running = False

        self.reset_contadores()
        self.crear_ui()

    # ─────────────────────────
    def reset_contadores(self):
        self.plastico = 0
        self.metal = 0
        self.papel = 0

    # ───────────────────────── UI
    def crear_ui(self):

        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=20, pady=20)

        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=1)

        # IZQUIERDA
        left = tk.Frame(main, bg=BG)
        left.grid(row=0, column=0, sticky="n")

        self.crear_video(left)
        self.crear_botones(left)

        # DERECHA
        right = tk.Frame(main, bg=BG)
        right.grid(row=0, column=1, sticky="n")

        self.crear_categorias(right)
        self.crear_contador(right)   # 👈 AQUÍ DEBAJO DE CATEGORÍAS
        self.crear_tiempo(right)

    # ───────────────────────── VIDEO
    def crear_video(self, parent):

        card = tk.Frame(parent, bg=CARD)
        card.pack(pady=10)

        frame_video = tk.Frame(card, width=VIDEO_W, height=VIDEO_H, bg=VIDEO_BG)
        frame_video.pack(padx=10, pady=10)
        frame_video.pack_propagate(False)

        self.video_label = tk.Label(
            frame_video,
            bg=VIDEO_BG,
            fg="white",
            text="Selecciona cámara o imagen",
            font=("Arial", 14)
        )
        self.video_label.pack(fill="both", expand=True)

    # ───────────────────────── BOTONES
    def crear_botones(self, parent):

        frame = tk.Frame(parent, bg=BG)
        frame.pack(pady=10)

        self.btn(frame, "📷 Cámara", "#2563eb", self.iniciar_camara)
        self.btn(frame, "🖼 Imagen", ORANGE, self.cargar_imagen)
        self.btn(frame, "⛔ Stop", RED, self.detener)

    def btn(self, parent, txt, color, cmd):
        tk.Button(
            parent,
            text=txt,
            bg=color,
            fg="white",
            width=15,
            height=2,
            command=cmd
        ).pack(side="left", padx=5)

    # ───────────────────────── PANEL DERECHO
    def crear_categorias(self, parent):

        card = tk.Frame(parent, bg=CARD)
        card.pack(fill="x", pady=10)

        tk.Label(card, text="Categorías", bg=CARD, fg=TEXT).pack(pady=5)

        self.barra(card, "Plástico", ORANGE)
        self.barra(card, "Metal", GRAY)
        self.barra(card, "Papel", BLUE)

    def crear_contador(self, parent):

        card = tk.Frame(parent, bg=CARD)
        card.pack(fill="x", pady=10)

        tk.Label(
            card,
            text="Contador por Categoría",
            bg=CARD,
            fg=TEXT,
            font=("Arial", 12, "bold")
        ).pack(pady=10)

        self.lbl_p = self.barra(card, "Plástico", ORANGE, "0")
        self.lbl_m = self.barra(card, "Metal", GRAY, "0")
        self.lbl_pa = self.barra(card, "Papel", BLUE, "0")

    def crear_tiempo(self, parent):

        card = tk.Frame(parent, bg=GREEN)
        card.pack(fill="x", pady=10)

        self.lbl_tiempo = tk.Label(
            card,
            text="Tiempo por frame\n0 ms",
            bg=GREEN,
            fg="white",
            font=("Arial", 12, "bold")
        )
        self.lbl_tiempo.pack(pady=15)

    def barra(self, parent, texto, color, valor=""):
        f = tk.Frame(parent, bg=color)
        f.pack(fill="x", padx=10, pady=5)

        lbl = tk.Label(f, text=f"{texto} {valor}", bg=color, fg="white")
        lbl.pack(side="left", padx=10)

        return lbl

    # ───────────────────────── FUNCIONES
    def iniciar_camara(self):
        self.detener()
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.reset_contadores()
        self.loop()

    def cargar_imagen(self):
        self.detener()

        ruta = filedialog.askopenfilename(
            filetypes=[("Imágenes", "*.jpg *.png *.jpeg")]
        )
        if not ruta:
            return

        frame = cv2.imread(ruta)
        frame, t, fps, n = core.procesar_frame(frame)
        self.actualizar_ui(frame, t)

    def detener(self):
        self.running = False
        if self.cap:
            self.cap.release()

    # ───────────────────────── LOOP
    def loop(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame, t, fps, n = core.procesar_frame(frame)
        self.actualizar_ui(frame, t)

        self.root.after(10, self.loop)

    # ───────────────────────── UPDATE
    def actualizar_ui(self, frame, tiempo):

        self.lbl_tiempo.config(text=f"Tiempo por frame\n{tiempo:.1f} ms")

        frame = cv2.resize(frame, (VIDEO_W, VIDEO_H))

        # simulación
        if tiempo < 50:
            self.plastico += 1
        elif tiempo < 80:
            self.metal += 1
        else:
            self.papel += 1

        self.lbl_p.config(text=f"Plástico {self.plastico}")
        self.lbl_m.config(text=f"Metal {self.metal}")
        self.lbl_pa.config(text=f"Papel {self.papel}")

        self.mostrar(frame)

    def mostrar(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk, text="")


# MAIN
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()