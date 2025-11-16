import cv2
import os
import numpy as np

dataPath = r'C:/Users/jesus/Desktop/Reconocimiento Emociones/Data'
emotionsList = os.listdir(dataPath)
print('Lista de emociones: ', emotionsList)

labels = []
facesData = []
label = 0

for nameDir in emotionsList:
    emotionPath = dataPath + '/' + nameDir
    print('Leyendo las imágenes de:', nameDir)

    for fileName in os.listdir(emotionPath):
        print('Rostros: ', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(emotionPath + '/' + fileName, 0))
        image = cv2.imread(emotionPath + '/' + fileName, 0)
        # cv2.imshow('image', image)
        # cv2.waitKey(10)
    label = label + 1

#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))

# Almacenando el modelo
#face_recognizer.write('modeloEigenFaceEmociones.xml')
#face_recognizer.write('modeloFisherFaceEmociones.xml')
face_recognizer.write('modeloLBPHFaceEmociones.xml')
print("Modelo de emociones almacenado..")