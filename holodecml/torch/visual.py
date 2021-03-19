import os
import cv2
import logging
from PIL import Image


logger = logging.getLogger(__name__)


def generate_video(image_folder, video_name, fps=5):

    logger.info(f"Creating movie from images in {image_folder} ...")

    images = [img for img in os.listdir(image_folder)
              if img.endswith(".jpg") or
              img.endswith(".jpeg") or
              img.endswith("png")]

    images = sorted(images, key=lambda x: int(x.split("_")[-1].strip(".png")))
    # Array images should only consider
    # the image files ignoring others if any
    # print(images)
    logger.info(f"... there are {len(images)} images that will be used")

    frame = cv2.imread(os.path.join(image_folder, images[0]))

    # setting the frame width, height width
    # the width, height of first image
    height, width, layers = frame.shape

    video_name = os.path.join(image_folder, video_name)

    video = cv2.VideoWriter(video_name, 0, fps, (width, height))

    # Appending the images to the video one by one
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    # Deallocating memories taken for window creation
    cv2.destroyAllWindows()
    video.release()  # releasing the video generated

    logger.info(f"... saved movie to {video}")
