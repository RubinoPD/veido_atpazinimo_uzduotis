import cv2
import numpy as np

# Inicijuojame vaizdo įrašą iš kameros
cap = cv2.VideoCapture(0)

# Inicijuojame veido detektorių
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Mėlynos spalvos diapazonas HSV erdvėje
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])

while True:
    # Nuskaityti kiekvieną kadrą
    ret, frame = cap.read()
    if not ret:
        break

    # Konvertuoti kadrą į pilką atspalvį ir HSV spalvų erdvę
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Aptikti veidus
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Sukurti mėlynos spalvos kaukę
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    for (x, y, w, h) in faces:
        # Apibrėžti veido plotą
        face_roi = mask[y:y+h, x:x+w]  # Kaukę pritaikome tik veido sričiai
        blue_pixels = cv2.countNonZero(face_roi)  # Suskaičiuojame mėlynus pikselius

        # Patikriname, ar veido srityje yra mėlynų pikselių (pvz., mėlyna kaukė)
        if blue_pixels > 500:  # Slenkstis (gali būti koreguojamas)
            color = (255, 0, 0)  # Mėlyna spalva
            label = "Blue Mask Detected"
        else:
            color = (0, 255, 0)  # Žalia spalva
            label = "No Mask"

        # Atvaizduojame rezultatą
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Rodome rezultatus
    cv2.imshow('Frame', frame)
    cv2.imshow('Mask', mask)

    # Nutraukimo mygtukas (ESC)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Atlaisviname resursus
cap.release()
cv2.destroyAllWindows()
