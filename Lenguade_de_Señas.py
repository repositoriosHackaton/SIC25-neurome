import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# 1. Cargar el modelo entrenado
model = tf.keras.models.load_model("sign_language_modelMejorado.keras")
CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'Q', 'R', 'U', 'V', 'W',
    'X', 'Y', 'Z', 'nothing', 'space'  # 25 clases
]

# 2. Configurar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Detectar solo una mano
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 3. Configurar la cámara
cap = cv2.VideoCapture(0)  # Usar cámara 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 4. Convertir BGR a RGB (MediaPipe requiere RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 5. Detectar la mano
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        # 6. Obtener coordenadas del bounding box de la mano
        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, _ = frame.shape
        x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
        y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # 7. Añadir margen al bounding box (20% del tamaño)
        margin = int(0.2 * max(x_max - x_min, y_max - y_min))
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(w, x_max + margin)
        y_max = min(h, y_max + margin)
        
        # 8. Recortar y redimensionar la mano
        hand_img = frame[y_min:y_max, x_min:x_max]
        hand_img = cv2.resize(hand_img, (224, 224))
        
        # 9. Preprocesamiento para el modelo
        processed_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
        processed_img = tf.keras.applications.mobilenet_v2.preprocess_input(processed_img)
        input_tensor = tf.expand_dims(processed_img, axis=0)
        
        # 10. Predecir la seña
        predictions = model.predict(input_tensor, verbose=0)
        class_id = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        label = f"{CLASSES[class_id]} ({confidence:.2f})"
        
        # 11. Dibujar resultados en el frame original
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, label, (x_min, y_min - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 12. Mostrar el frame
    cv2.imshow("Detección de Señas", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 13. Liberar recursos
cap.release()
cv2.destroyAllWindows()