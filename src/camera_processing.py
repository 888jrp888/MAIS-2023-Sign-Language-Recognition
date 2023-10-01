import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = '../models/gesture_recognizer.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
#detector = vision.HandLandmarker.create_from_options(options)

#Affiche les FPS de la caméra en haut a droite
def fpsTracker(cv2, tempsInitial, tempsPresent):

    FPS = 1/(tempsPresent - tempsInitial)
    return cv2.putText(frame, f"FPS: {str(int(FPS))}", (7,70), cv2.FONT_ITALIC, 1, (200,200,200), 3)



mpMain = mp.solutions.hands
mains = mpMain.Hands()

mpDessin = mp.solutions.drawing_utils

#####################   CAMERA   ############################
cv2.namedWindow("Camera Window")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

tempsInitial = 0
while rval:
    cv2.imshow("Camera Window", frame)
    rval, frame = vc.read()

    resultat = mains.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if resultat.multi_hand_landmarks:
        for pointsMain in resultat.multi_hand_landmarks:
            mpDessin.draw_landmarks(frame,
                                    pointsMain,
                                    mpMain.HAND_CONNECTIONS)

    #print(resultat.multi_hand_landmarks) #imprime la position des mains


    tempsPresent = time.time()
    fpsTracker(cv2, tempsInitial, tempsPresent)
    tempsInitial = tempsPresent

    #Reader afin de savoir quand quitter
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break