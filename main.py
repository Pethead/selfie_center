import os
import sys
import argparse
import dlib
import glob
import cv2
import numpy as np


def check_input_args(args):
    if (os.path.exists(args.input) is True and
            os.path.isdir(args.input) is True):

        image_files = glob.glob(args.input + "/*.jpg")

        if (0 == len(image_files)):
            print(f"No *.jpg files found in '{args.input}'!", file=sys.stderr)
        else:
            # Create output directory if it doesn't exist
            if (os.path.isdir(args.output) is False):
                os.mkdir(args.output)
            return True
    else:
        print(f"{args.input} directory doesn't exist!", file=sys.stderr)

    return False


def center_images(image_files, input_folder, output_folder):
    # Load the pre-trained shape predictor model
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    total_images = len(image_files)
    current_image = 1

    for image_file in image_files:
        # Read the image file
        image = cv2.imread(image_file)
        # Convert image to grayscale in order to detect faces
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = dlib.get_frontal_face_detector()(grayscale_image)

        # Disregard pictures with 0 or multiple faces
        if (1 == len(faces)):
            face = faces[0]
        else:
            print(f"Disregarded: {image_file} - Detected '{len(faces)}' faces",
                  file=sys.stderr)
            continue

        # Detect facial landmarks including the nose
        landmarks = predictor(grayscale_image, face)
        # Index 30 corresponds to the tip of the nose
        nose_landmark = landmarks.part(30)

        # Calculate the offset for centering the image around the nose
        dx = int(image.shape[1] / 2 - nose_landmark.x)
        dy = int(image.shape[0] / 2 - nose_landmark.y)

        # Create a translation matrix to shift the image
        translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])

        # Apply the translation to center the image around the nose
        centered_img = cv2.warpAffine(image, translation_matrix,
                                      (image.shape[1], image.shape[0]))

        image_file = str(image_file).replace(input_folder + "/", "")
        # Save the centered image with a new filename in given output directory
        centered_filename = "centered_" + str(image_file)
        cv2.imwrite(output_folder + "/" + centered_filename, centered_img)

        print(f"{current_image}/{total_images} - ./{input_folder}/{image_file}\
 centered around the nose & saved as ./{output_folder}/{centered_filename}")
        current_image += 1


def create_video(image_folder, fps):
    video_name = "centered_video.mp4"
    height = 1280
    width = 1000

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"mp4v"),
                            fps, (width, height))

    image_files = glob.glob(image_folder + "/*.jpg")
    total_frames = len(image_files)
    current_frame = 1

    for image in image_files:
        # Read image and resize to fit video resolution
        frame = cv2.imread(image)
        resized_frame = cv2.resize(frame, (width, height))

        video.write(resized_frame)

        print(f"Frame '{current_frame}/{total_frames}' written - {image}")
        current_frame += 1

    cv2.destroyAllWindows()
    video.release()

    print(f"Centered video saved as ./{video_name}")


def main(args):
    if (check_input_args(args) is True):
        image_files = glob.glob(args.input + "/*.jpg")

        center_images(image_files, args.input, args.output)
        create_video(args.output, args.fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        help="Input directory containing all the images (jpg)",
                        action="store")
    parser.add_argument("-o", "--output", type=str,
                        help="Output directory to store centered images",
                        action="store")
    parser.add_argument("-fps", "--fps", type=int,
                        help="Frames per second of output video",
                        action="store")
    args = parser.parse_args()
    main(args)
