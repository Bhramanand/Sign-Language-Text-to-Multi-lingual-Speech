import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Initialize camera and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Constants
offset = 20
imgSize = 300

# Variables
current_sign = 'A'  # Start with 'A'
folder = f"Data/{current_sign}"
os.makedirs(folder, exist_ok=True)
counter = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from camera. Exiting...")
        break

    # Detect hands
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure cropping is within image boundaries
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(img.shape[1], x + w + offset)
        y2 = min(img.shape[0], y + h + offset)
        imgCrop = img[y1:y2, x1:x2]

        # Create a white canvas
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Aspect ratio adjustment
        aspectRatio = h / w
        try:
            if aspectRatio > 1:  # Tall image
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:  # Wide image
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
        except Exception as e:
            print(f"Error during resizing or cropping: {e}")
            continue

        # Display processed images
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # Display the main camera feed with the current sign
    cv2.putText(img, f"Sign: {current_sign}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Image", img)

    # Keyboard interactions
    key = cv2.waitKey(1)
    if key == ord("s"):  # Save image
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(f"Saved {current_sign} image: {counter}")
    elif key >= ord('A') and key <= ord('Z'):  # Change sign
        current_sign = chr(key)
        folder = f"Data/{current_sign}"
        os.makedirs(folder, exist_ok=True)
        counter = 0
        print(f"Switched to sign: {current_sign}")
    elif key == ord("q"):  # Quit
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
