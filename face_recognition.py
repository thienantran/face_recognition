import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load an image
img = cv2.imread('476C8EA2-253B-4733-BFA8-46E4BB6C6401.jpg')

# Convert color image to grayscale for Viola-Jones
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Perform face detection
faces = face_cascade.detectMultiScale(grayscale_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangle around the faces in the original image
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display the output
cv2.imshow('img', img)
cv2.waitKey()
