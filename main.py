import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import tkinter as tk
import pyautogui
import pyttsx3
from datetime import datetime, timedelta
from tkinter import ttk, Label, Button, Frame
from PIL import Image, ImageTk

# Configuración de MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Clases del modelo
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
           'K', 'L', 'M', 'N', 'O', 'Q', 'R', 'U', 'V', 'W',
           'X', 'Y', 'Z', 'nothing', 'space']

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Intérprete de Señas en Tiem Real")
        self.root.geometry("1280x720")
        
        # Cargar modelo
        try:
            self.model = tf.keras.models.load_model("sign_language_modelMejorado3-7.keras")
            print("Modelo cargado exitosamente")
        except Exception as e:
            print(f"Error cargando modelo: {str(e)}")
            raise
        
        # Configuración de cámara
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Error: No se detectó cámara")
        
        # Estados
        self.is_processing = False
        self.is_recording = False
        
        # Configuración TTS
        self.tts_engine = pyttsx3.init()
        self.last_spoken = None
        self.last_spoken_time = datetime.now()
        self.configure_tts()
        
        # Interfaz
        self.create_widgets()
        self.setup_controls()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def configure_tts(self):
        """Configura el motor de voz con voz en español"""
        try:
            voices = self.tts_engine.getProperty('voices')
            # Buscar voz en español (ajustar según sistema)
            for voice in voices:
                if 'spanish' in voice.name.lower() or 'español' in voice.name.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
            self.tts_engine.setProperty('rate', 150)
        except Exception as e:
            print(f"Error en TTS: {str(e)}")

    def create_widgets(self):
        """Crea la interfaz gráfica"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Panel de video dual
        video_frame = ttk.Frame(main_frame)
        video_frame.pack(fill=tk.BOTH, expand=True)
        
        # Cámara
        self.camera_label = ttk.Label(video_frame)
        self.camera_label.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)
        
        # Grabación de pantalla
        self.screen_label = ttk.Label(video_frame)
        self.screen_label.pack(side=tk.RIGHT, padx=10, fill=tk.BOTH, expand=True)
        
        # Etiqueta de predicción
        self.prediction_label = Label(
            main_frame,
            text="Presiona 'Iniciar Cámara' para comenzar",
            font=("Arial", 24, "bold"),
            bg="#2c3e50",
            fg="white",
            padx=20,
            pady=10
        )
        self.prediction_label.pack(pady=20)

    def setup_controls(self):
        """Configura los botones de control"""
        control_frame = Frame(self.root, bg="#2c3e50")
        control_frame.pack(pady=10)
        
        # Botones
        self.btn_camera = ttk.Button(
            control_frame,
            text="Iniciar Cámara",
            command=self.toggle_camera,
            width=15
        )
        self.btn_camera.pack(side=tk.LEFT, padx=10)
        
        self.btn_record = ttk.Button(
            control_frame,
            text="Grabar Pantalla",
            command=self.toggle_recording,
            width=15
        )
        self.btn_record.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(
            control_frame,
            text="Salir",
            command=self.on_close,
            width=15
        ).pack(side=tk.RIGHT, padx=10)

    def toggle_camera(self):
        """Alterna el procesamiento de la cámara"""
        self.is_processing = not self.is_processing
        self.btn_camera.config(
            text="Detener Cámara" if self.is_processing else "Iniciar Cámara"
        )
        if self.is_processing:
            self.process_camera()
        else:
            self.camera_label.config(image='')

    def toggle_recording(self):
        """Alterna la grabación de pantalla"""
        self.is_recording = not self.is_recording
        self.btn_record.config(
            text="Detener Grabación" if self.is_recording else "Grabar Pantalla"
        )
        if self.is_recording:
            self.process_screen()
        else:
            self.screen_label.config(image='')

    def process_camera(self):
        """Procesa los frames de la cámara"""
        if self.is_processing and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)  # Espejo para vista natural
                processed_frame = self.detect_hands(frame)
                self.update_frame(self.camera_label, processed_frame)
            self.root.after(10, self.process_camera)

    def process_screen(self):
        """Procesa la pantalla y detecta señas"""
        if self.is_recording:
            # Capturar pantalla completa
            screen = pyautogui.screenshot()
            frame = np.array(screen)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Detectar manos y actualizar
            processed_frame = self.detect_hands(frame)
            self.update_frame(self.screen_label, processed_frame)
            self.root.after(50, self.process_screen)

    def detect_hands(self, frame):
        """Detecta manos y realiza predicciones"""
        # Detección con MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            h, w = frame.shape[:2]
            
            # Calcular región de interés (ROI)
            x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            margin = int(0.2 * max(x_max - x_min, y_max - y_min))
            
            # Ajustar coordenadas con margen
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)
            
            # Recortar y preprocesar imagen
            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size > 0:
                hand_img = cv2.resize(hand_img, (224, 224))
                hand_img = hand_img.astype(np.float32) / 255.0
                
                # Predicción
                predictions = self.model.predict(np.expand_dims(hand_img, axis=0))
                class_id = np.argmax(predictions[0])
                confidence = np.max(predictions[0])
                
                # Lógica de voz
                if confidence > 0.6:
                    letter = CLASSES[class_id]
                    self.handle_tts(letter)
                    
                    # Actualizar UI
                    self.prediction_label.config(
                        text=f"{letter} ({confidence:.2f})",
                        fg="#2ecc71" if letter not in ['nothing', 'space'] else "#e74c3c"
                    )
                
                # Dibujar cuadro de detección
                cv2.rectangle(frame, 
                    (x_min, y_min), (x_max, y_max),
                    (0, 255, 0), 2
                )

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def handle_tts(self, letter):
        """Maneja la reproducción de voz"""
        current_time = datetime.now()
        # Filtrar 'nothing' y manejar 'space'
        if letter == 'nothing':
            return
        elif letter == 'space':
            letter = ' '
        
        # Evitar repeticiones rápidas
        if (self.last_spoken != letter or 
            (current_time - self.last_spoken_time) > timedelta(seconds=1)):
            try:
                self.tts_engine.say(letter)
                self.tts_engine.runAndWait()
                self.last_spoken = letter
                self.last_spoken_time = current_time
            except Exception as e:
                print(f"Error en TTS: {str(e)}")

    def update_frame(self, label_widget, frame):
        """Actualiza un widget de imagen con el frame procesado"""
        frame = cv2.resize(frame, (640, 480))
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label_widget.imgtk = imgtk
        label_widget.config(image=imgtk)

    def on_close(self):
        """Maneja el cierre seguro de la aplicación"""
        self.is_processing = False
        self.is_recording = False
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()