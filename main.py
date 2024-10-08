import datetime
import cv2
import face_recognition

#load sample image
# sample_image = face_recognition.load_image_file("./images/messi.jpg")
sample_image = face_recognition.load_image_file("./images/ronaldo.jpg")
sample_image_encoded = face_recognition.face_encodings(sample_image)[0]

# Load input image
image = cv2.imread("./input.jpg")

# Convert to RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Find faces
face_locations = face_recognition.face_locations(rgb_image)
face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

# Loop through faces
for face_encoding in face_encodings :
    # Compare faces
    results = face_recognition.compare_faces([sample_image_encoded], face_encoding)

    # Print results
    if results[0]:
        # Display image
        for (top, right, bottom, left), name in zip(face_locations, "DODO".split()):
            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

            cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Save image
            cv2.imwrite("./output.jpg", image)

        print("Found face")
    else:
        print("Not found")