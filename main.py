import os
import argparse
import dlib
import glob
import cv2
import numpy as np

def center_images(image_files, input_folder, output_folder):
    # Load the pre-trained shape predictor model
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    total_images = len(image_files)
    current_image = 1

    for image_file in image_files:
        # Read the image file
        img = cv2.imread(image_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = dlib.get_frontal_face_detector()(gray)

        # Take first face only as the first detected face is
        #  most likely the person who is taking the selfie
        if (1 == len(faces)):
            face = faces[0]
        else:
            # TODO: if zero or multiple faces, write filename to stderr so it can be analyzed.
            continue

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
        centered_filename = "centered_" + str(image_file).replace(input_folder + "/", "")
        cv2.imwrite(output_folder + "/" + centered_filename, centered_img)

        print(f"{current_image}/{total_images} - './{image_file}' centered around the nose & saved as './{output_folder}/{centered_filename}'")
        current_image += 1


def create_video(image_folder):
    video_name = "centered_video.mp4"

    image_files = glob.glob(image_folder + "/*.jpg")

    # Create video resolution based on height and width of first image
    frame = cv2.imread(image_files[0])
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"mp4v"), 2, (width, height))

    for image in image_files:
        frame = cv2.imread(image)
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()


def main(args):
    input_folder = str(args.input)
    image_files = []

    if (False == os.path.exists(input_folder)):
        print("Input folder doesn't exist!")
    else:
        # Get a list of all image files in the current folder
        image_files = glob.glob(input_folder + "/*.jpg")

        if (0 == len(image_files)):
            print("No images found in '{input_folder}'")
        else:
            output_folder = str(args.output)
            # Create output folder if it doesn't exist
            if (False == os.path.isdir(output_folder)):
                os.mkdir(output_folder)

            center_images(image_files, input_folder, output_folder)
            # Create video based on all centered images
            create_video(output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        help="Input folder that contains all the images",
                        action="store")
    parser.add_argument("-o", "--output", type=str,
                    help="Output folder to store output in",
                    action="store")
    args = parser.parse_args()
    main(args)
