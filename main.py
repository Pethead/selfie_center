import dlib
import glob
import cv2
import numpy as np

# Load the pre-trained shape predictor model
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Get a list of all image files in the current folder
image_files = glob.glob("*.jpg") + glob.glob("*.jpeg") + glob.glob("*.png")

for image_file in image_files:
    # Read the image file
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = dlib.get_frontal_face_detector()(gray)

    for face in faces:
        # Detect facial landmarks including the nose
        landmarks = predictor(gray, face)
        nose_landmark = landmarks.part(30)  # Index 30 corresponds to the tip of the nose

        # Calculate the offset for centering the image around the nose
        dx = int(img.shape[1] / 2 - nose_landmark.x)
        dy = int(img.shape[0] / 2 - nose_landmark.y)

        # Create a translation matrix to shift the image
        translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])

        # Apply the translation to center the image around the nose
        centered_img = cv2.warpAffine(img, translation_matrix, (img.shape[1], img.shape[0]))

        # Save the centered image with a new filename
        centered_filename = "centered_" + image_file
        cv2.imwrite(centered_filename, centered_img)

        print(f"Image {image_file} centered around the nose and saved as {centered_filename}.")