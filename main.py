import cv2
import numpy as np

net = cv2.dnn.readNetFromDarknet("Resources/yolov4-tiny.cfg", "Resources/yolov4-tiny.weights")
with open("Resources/coco.names.txt") as file:
    classes = file.read().split("\n")
while True:
    videoCameraCapture = cv2.VideoCapture("1.mp4")
    while videoCameraCapture.isOpened():
        ret, frame = videoCameraCapture.read()
        if not ret:
            break
        height, width, _ = frame.shape
        net.setInput(cv2.dnn.blobFromImage(frame, 1 / 255, (608, 608), (0, 0, 0), swapRB=True, crop=False))
        classIndexes, classScores, boxes = ([] for i in range(3))
        for out in net.forward([net.getLayerNames()[index - 1] for index in net.getUnconnectedOutLayers()]):
            for i in out:
                scores = i[5:]
                if scores[np.argmax(scores)] > 0:
                    boxes.append([int(i[0] * width) - int(i[2] * width) // 2, int(i[1] * height) - int(i[3] * height) // 2, int(i[2] * width), int(i[3] * height)])
                    classIndexes.append(np.argmax(scores))
                    classScores.append(float(scores[np.argmax(scores)]))
        for boxIndex in cv2.dnn.NMSBoxes(boxes, classScores, 0.0, 1):
            if classes[classIndexes[boxIndex]] in ["sports ball"]:
                x, y, w, h = boxes[boxIndex]
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
        frame = cv2.resize(frame, (1920 // 3, 1080 // 3))
        cv2.imshow("Video Capture", frame)
        cv2.waitKey(1)
    videoCameraCapture.release()
    cv2.destroyAllWindows()