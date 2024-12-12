import cv2
import time
import os
import tensorflow as tf
import numpy as np
import pyttsx3

from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(123)

class Detector:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Speed percent (can go over 100)
        self.engine.setProperty('volume', 0.9)  # Volume 0-1

    def readClasses(self, classesFilePath):
        with open(classesFilePath, 'r') as f:
            self.classesList = f.read().splitlines()

        # Color list
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))

    def downloadModel(self, modelURL):
        fileName = os.path.basename(modelURL)
        self.modelName = fileName[:fileName.index('.')]
        
        self.cacheDir = "./pretrained_models"
        os.makedirs(self.cacheDir, exist_ok=True)

        get_file(fname=fileName, origin=modelURL, cache_dir=self.cacheDir, cache_subdir="checkpoints", extract=True)
        
    def loadModel(self):
        print("Loading Model " + self.modelName)
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkpoints", self.modelName, "saved_model"))

        print("Model " + self.modelName + " loaded successfully...")

    def estimateDistance(self, bbox, image_shape):
        ymin, xmin, ymax, xmax = bbox
        obj_height = ymax - ymin

        # Placeholder values
        focal_length = 800  # Placeholder focal length (in pixels)
        real_object_height = 1.7  # Placeholder height of the object in meters

        distance = (focal_length * real_object_height) / obj_height
        return distance

    def createBoundingBox(self, image, threshold=0.5):
        if image is None:
            print("Error: Image not loaded correctly.")
            return None

        inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
        inputTensor = inputTensor[tf.newaxis, ...]

        detections = self.model(inputTensor)

        bboxs = detections['detection_boxes'][0].numpy()
        classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
        classScores = detections['detection_scores'][0].numpy()

        imH, imW, imC = image.shape

        bboxIdx = tf.image.non_max_suppression(
            bboxs, classScores, max_output_size=50,
            iou_threshold=threshold, score_threshold=threshold
        )

        detected_classes = []
        
        if len(bboxIdx) != 0:
            for i in bboxIdx:
                bbox = tuple(bboxs[i].tolist())
                classConfidence = round(100 * classScores[i])
                classIndex = classIndexes[i]

                classLabelText = self.classesList[classIndex].upper()
                classColor = self.colorList[classIndex]

                displayText = '{}: {}%'.format(classLabelText, classConfidence)

                ymin, xmin, ymax, xmax = bbox

                xmin, xmax, ymin, ymax = (xmin * imW, xmax * imW, ymin * imH, ymax * imH)
                xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=classColor, thickness=1)
                cv2.putText(image, displayText, (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

                distance = self.estimateDistance((ymin, xmin, ymax, xmax), image.shape)
                distance_text = f'Distance: {distance:.2f} m'
                cv2.putText(image, distance_text, (xmin, ymin - 30), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

                lineWidth = min(int((xmax - xmin) * 0.2), int((ymax - ymin) * 0.2))
                                                                 
                cv2.line(image, (xmin, ymin), (xmin + lineWidth, ymin), classColor, thickness=5)
                cv2.line(image, (xmin, ymin), (xmin, ymin + lineWidth), classColor, thickness=5)

                cv2.line(image, (xmax, ymin), (xmax - lineWidth, ymin), classColor, thickness=5)
                cv2.line(image, (xmax, ymin), (xmax, ymin + lineWidth), classColor, thickness=5)

                cv2.line(image, (xmin, ymax), (xmin + lineWidth, ymax), classColor, thickness=5)
                cv2.line(image, (xmin, ymax), (xmin, ymax - lineWidth), classColor, thickness=5)

                cv2.line(image, (xmax, ymax), (xmax - lineWidth, ymax), classColor, thickness=5)
                cv2.line(image, (xmax, ymax), (xmax, ymax - lineWidth), classColor, thickness=5)

                detected_classes.append(classLabelText)
        
        if detected_classes:
            self.speak(detected_classes)
                
        return image

    def speak(self, detected_classes):
        unique_classes = set(detected_classes)
        message = "Detected objects: " + ", ".join(unique_classes)
        self.engine.say(message)
        self.engine.runAndWait()

    def predictImage(self, imagePath, threshold=0.5):
        image = cv2.imread(imagePath)
        if image is None:
            print(f"Error: Unable to load image at path {imagePath}")
            return

        bboxImage = self.createBoundingBox(image, threshold)
        if bboxImage is not None:
            cv2.imwrite(self.modelName + ".jpg", bboxImage)
            cv2.imshow("Result", bboxImage)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def predictVideo(self, videoPath, threshold=0.5):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error opening video file.")
            return

        success, image = cap.read()
        startTime = 0

        while success:
            currentTime = time.time()
            fps = 1 / (currentTime - startTime)
            startTime = currentTime

            bboxImage = self.createBoundingBox(image, threshold)
            if bboxImage is not None:
                cv2.putText(bboxImage, "FPS: " + str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                cv2.imshow("Result", bboxImage)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            success, image = cap.read()

        cap.release()
        cv2.destroyAllWindows()
