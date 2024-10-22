import cv2
import face_recognition

# Memuat gambar contoh dan meng-encode-nya
# sample_image = face_recognition.load_image_file("./images/messi.jpg")
sample_image = face_recognition.load_image_file("./images/ronaldo.jpg")
sample_image_encoded = face_recognition.face_encodings(sample_image)[0]

# Memuat gambar input
image = cv2.imread("./input.jpg")

# Mengonversi ke RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Menemukan wajah dalam gambar input
face_locations = face_recognition.face_locations(rgb_image)
face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

# Melakukan loop untuk setiap wajah yang ditemukan
for face_encoding in face_encodings:
    # Membandingkan wajah dengan gambar contoh
    results = face_recognition.compare_faces([sample_image_encoded], face_encoding)

    if results[0]:
        # Jika ditemukan kecocokan, gambarkan persegi di sekitar wajah
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

            # Menambahkan label
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image, "Ronaldo", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Menyimpan gambar hasil
        cv2.imwrite("./output.jpg", image)

        print("Wajah ditemukan")
    else:
        print("Wajah tidak ditemukan")
