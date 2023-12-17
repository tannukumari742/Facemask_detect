import cv2
#from tensorflow.keras.preprocessing import image
from keras.preprocessing import image
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def maskdetect():

    # Build a simple CNN model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Load the trained model
    model.load_weights(r'C:\Users\HP\Desktop\face_mask\dataset.h5')

    # Open camera
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        img = cv2.resize(frame, (150, 150))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255

        prediction = model.predict(img)
        mask_status = "Mask" if prediction[0][0] > 0.5 else "No Mask"

        # Detect face using a face detection algorithm (e.g., Haar Cascade)
        # Assuming you have a trained Haar Cascade XML file for face detection named 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Draw a red rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            # Write text below the rectangle
            cv2.putText(frame, mask_status, (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Mask Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    #To call the function
if __name__ == "__main__":
    maskdetect()
