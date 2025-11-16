import cv2
import os
import time

dataPath = r'C:/Users/jesus/Desktop/Reconocimiento Emociones/Data'
emotionsList = os.listdir(dataPath)
print('emotionsList=', emotionsList)

#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Leyendo el modelo entrenado de emociones
#face_recognizer.read('modeloEigenFaceEmociones.xml')
#face_recognizer.read('modeloFisherFaceEmociones.xml')
face_recognizer.read('modeloLBPHFaceEmociones.xml')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cv2.namedWindow('Reconocimiento de Emociones', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Reconocimiento de Emociones', 800, 600)

start_time = time.time()

while True:
    ret, frame = cap.read()
    if ret == False: 
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)

        cv2.putText(frame, str(result), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        
        # LBPHFace - Misma lógica que tu código original
        if result[1] < 70:  # Puedes ajustar este threshold
            cv2.putText(frame, '{}'.format(emotionsList[result[0]]), (x, y - 45), 
                        2, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Desconocido', (x, y - 20), 
                        2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
    cv2.imshow('Reconocimiento de Emociones', frame)
    k = cv2.waitKey(1)
    
    if k == 27 or cv2.getWindowProperty('Reconocimiento de Emociones', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()

end_time = time.time()
elapsed = end_time - start_time
print(f"[Finished in {elapsed:.2f} s]")