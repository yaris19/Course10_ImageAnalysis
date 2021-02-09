import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
from cv2 import CascadeClassifier, imread, imshow, rectangle, waitKey, \
    VideoCapture, cvtColor, COLOR_BGR2GRAY, destroyAllWindows, flip
from matplotlib.patches import Rectangle, Circle
from mtcnn.mtcnn import MTCNN


def detect_face_cv2(img):
    # load the photograph
    pixels = imread(img)
    # load the pre-trained model
    classifier = CascadeClassifier('haarcascade_frontalface_default.xml')
    # perform face detection
    bboxes = classifier.detectMultiScale(pixels,
                                         scaleFactor=1.05,
                                         minNeighbors=8
                                         )
    # print bounding box for each detected face
    for box in bboxes:
        x, y, width, height = box
        x2, y2 = x + width, y + height
        rectangle(pixels, (x, y), (x2, y2), (0, 0, 255), 1)

    imshow("face detection", pixels)
    waitKey(0)
    destroyAllWindows()


def detect_face_cv2_webcam():
    cap = VideoCapture(0)
    classifier = CascadeClassifier('haarcascade_frontalface_default.xml')

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        frame = flip(frame, 1)
        gray = cvtColor(frame, COLOR_BGR2GRAY)
        faces = classifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            # flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        imshow("Video", frame)
        if waitKey(1) == ord('q'):
            break

    cap.release()
    destroyAllWindows()


def find_face_mtcnn(color, result_list):
    for result in result_list:
        x, y, w, h = result["box"]
        rectangle(color, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return color


def detect_face_mtcnn_webcam():
    cap = VideoCapture(0)
    detector = MTCNN()

    while True:
        _, color = cap.read()
        frame = flip(color, 1)
        frame = cvtColor(frame, COLOR_BGR2GRAY)
        faces = detector.detect_faces(frame)
        detect_face = find_face_mtcnn(frame, faces)
        imshow("Video", detect_face)
        if waitKey(1) == ord('q'):
            break
    cap.release()
    destroyAllWindows()


def detect_face_mtcnn(img):
    pixels = plt.imread(img)
    detector = MTCNN()
    faces = detector.detect_faces(pixels)
    draw_image_with_boxes(img, faces)
    draw_faces(img, faces)


def draw_faces(img, result_list):
    data = plt.imread(img)
    for i, result in enumerate(result_list):
        x1, y1, width, height = result["box"]
        x2, y2 = x1 + width, y1 + height
        plt.subplot(1, len(result_list), i + 1)
        plt.axis("off")
        plt.imshow(data[y1:y2, x1:x2])
    plt.show()


def draw_image_with_boxes(img, result_list):
    data = plt.imread(img)
    plt.imshow(data, interpolation="nearest")
    ax = plt.gca()
    for result in result_list:
        x, y, width, height = result["box"]
        rect = Rectangle((x, y), width, height, fill=False, color="red")
        ax.add_patch(rect)

        for key, value in result["keypoints"].items():
            dot = Circle(value, radius=2, color="red")
            ax.add_patch(dot)

    plt.axis("off")
    plt.show()


# detect_face_cv2("images/test1.jpg")
# detect_face_cv2("images/test2.jpg")
# detect_face_cv2("images/yaris.jpg")

# detect_face_mtcnn("images/test1.jpg")
# detect_face_mtcnn("images/test2.jpg")
# detect_face_mtcnn("images/yaris.jpg")

detect_face_cv2_webcam()
# detect_face_mtcnn_webcam()
