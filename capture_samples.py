import cv2
import mediapipe as mp
import os
import time

def detectar_manos():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    contador_muestras = 0
    nombre_carpeta_principal = input("Ingrese el nombre de la carpeta principal: ")
    ruta_carpeta_principal = os.path.join(os.getcwd(), nombre_carpeta_principal) # Obtener la ruta completa

    try:
        os.makedirs(ruta_carpeta_principal, exist_ok=True)
        print(f"Carpeta principal creada en: {os.path.abspath(ruta_carpeta_principal)}")
    except OSError as e:
        print(f"Error al crear la carpeta principal: {e}")
        return  # Salir si no se puede crear la carpeta principal

    grabando = False
    frames_grabados = 0
    nombre_carpeta_video = ""
    tiempo_inicio_grabacion = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            if not grabando:
                grabando = True
                contador_muestras += 1
                nombre_carpeta_video = os.path.join(ruta_carpeta_principal, f"video_{contador_muestras}") #usar la ruta completa
                try:
                    os.makedirs(nombre_carpeta_video, exist_ok=True)
                    print(f"Carpeta de video creada en: {os.path.abspath(nombre_carpeta_video)}")
                except OSError as e:
                    print(f"Error al crear la carpeta de video: {e}")
                    grabando = False  # Detener la grabación si no se puede crear la carpeta
                    continue  # Saltar al siguiente fotograma
                print(f"Iniciando grabación... Muestra {contador_muestras}")
                frames_grabados = 0
                tiempo_inicio_grabacion = time.time()  # Guarda el tiempo de inicio

            frames_grabados += 1
            try:
                ruta_frame = os.path.join(nombre_carpeta_video, f"frame_{frames_grabados:04d}.jpg")
                cv2.imwrite(ruta_frame, frame)
                #print(f"Frame guardado en: {os.path.abspath(ruta_frame)}") #Imprime la ruta donde se guarda cada frame.
            except OSError as e:
                print(f"Error al guardar el frame: {e}")
            time.sleep(1/20)  # 20 fps

            tiempo_transcurrido = int(time.time() - tiempo_inicio_grabacion)
            cv2.putText(frame, f"Tiempo: {tiempo_transcurrido}s | Muestra: {contador_muestras}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif grabando:
            grabando = False
            print(f"Deteniendo grabación... Muestra {contador_muestras}")
            tiempo_inicio_grabacion = 0  # Reinicia el tiempo de inicio

        if grabando:
            cv2.putText(frame, "Grabando...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Detección de Manos', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detectar_manos()