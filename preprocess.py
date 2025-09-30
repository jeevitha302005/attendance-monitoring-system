from imutils import paths
import numpy as np
import imutils
import pickle  # used to save the embeddings data
import cv2
import os

# Paths for dataset and output files
dataset = "dataset"
embeddingFile = "output/embeddings.pickle"  # output embedding file
embeddingModel = "openface_nn4.small2.v1.t7"  # embedding model (Pytorch)

# Initialization of the Caffe model for face detection (pre-trained)
prototxt = "model/deploy.prototxt"
model = "model/res10_300x300_ssd_iter_140000.caffemodel"

# Load the face detector and the embedding model
detector = cv2.dnn.readNetFromCaffe(prototxt, model)
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

# Load Haar Cascade for eye detection – update this path accordingly
eyeCascadePath = r"haarcascade_eye.xml"
eyeDetector = cv2.CascadeClassifier(eyeCascadePath)

# Directories to save outputs (annotated images and processed face crops)
annotatedDir = "annotated_images"
processedDir = "processed_faces"
os.makedirs(annotatedDir, exist_ok=True)
os.makedirs(processedDir, exist_ok=True)

# Grab the paths to the input images in our dataset
imagePaths = list(paths.list_images(dataset))
knownEmbeddings = []
knownNames = []
total = 0
conf_threshold = 0.95  # confidence threshold for face detections

for (i, imagePath) in enumerate(imagePaths):
    print("Processing image {}/{}".format(i + 1, len(imagePaths)))
    # Get the student name from the folder structure
    name = imagePath.split(os.path.sep)[-2]
    
    # Read and resize the image
    orig_image = cv2.imread(imagePath)
    image = imutils.resize(orig_image, width=600)
    (h, w) = image.shape[:2]

    # Prepare the image for face detection
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False
    )
    detector.setInput(imageBlob)
    detections = detector.forward()

    best_conf = 0
    best_embedding = None
    best_face_box = None
    best_eye_boxes = None  # store eye detections (relative to face crop)

    fallback_conf = 0
    fallback_embedding = None
    fallback_face_box = None
    fallback_eye_boxes = None

    # Iterate over the detections
    for j in range(0, detections.shape[2]):
        confidence = detections[0, 0, j, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # Ensure coordinates are within image boundaries
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)
            
            # Crop the face region (full face crop)
            face = image[startY:endY, startX:endX]
            if face.size == 0 or face.shape[0] < 20 or face.shape[1] < 20:
                continue

            # Prepare the face crop for embedding extraction
            faceBlob = cv2.dnn.blobFromImage(
                face, 1.0 / 255, (96, 96),
                (0, 0, 0), swapRB=True, crop=False
            )
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # Save this candidate as fallback (using full face crop) if it has the highest confidence so far.
            if confidence > fallback_conf:
                fallback_conf = confidence
                fallback_embedding = vec.flatten()
                fallback_face_box = (startX, startY, endX, endY)
                fallback_eye_boxes = []  # default empty list

            # Now detect eyes within the face crop (convert to grayscale)
            grayFace = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            eyes = eyeDetector.detectMultiScale(
                grayFace, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15)
            )

            if len(eyes) > 0:
                # If eyes are detected and this candidate has higher confidence, update best candidate.
                if confidence > best_conf:
                    best_conf = confidence
                    best_embedding = vec.flatten()
                    best_face_box = (startX, startY, endX, endY)
                    best_eye_boxes = eyes.copy()

    # Choose the best candidate (with eyes) if available; otherwise, use the fallback candidate.
    if best_embedding is not None:
        embedding = best_embedding
        face_box = best_face_box
        eye_boxes = best_eye_boxes
    elif fallback_embedding is not None:
        print("No valid eye detection for image: {}. Using fallback face crop.".format(imagePath))
        embedding = fallback_embedding
        face_box = fallback_face_box
        eye_boxes = fallback_eye_boxes
    else:
        print("No valid face detected for image:", imagePath)
        continue

    # Create an annotated copy of the image.
    annotated = image.copy()
    if face_box is not None:
        (fx, fy, fex, fey) = face_box
        cv2.rectangle(annotated, (fx, fy), (fex, fey), (0, 255, 0), 2)
        if eye_boxes is not None:
            for (ex, ey, ew, eh) in eye_boxes:
                # The eye boxes are relative to the face crop.
                cv2.rectangle(annotated, (fx + ex, fy + ey), (fx + ex + ew, fy + ey + eh), (255, 0, 0), 2)

    # Use the full face crop from the best candidate for processed face saving.
    final_crop = image[face_box[1]:face_box[3], face_box[0]:face_box[2]]

    # Create unique filenames using the student name and the original image basename
    baseName = os.path.splitext(os.path.basename(imagePath))[0]
    uniqueName = f"{name}_{baseName}.png"
    annotatedPath = os.path.join(annotatedDir, uniqueName)
    processedPath = os.path.join(processedDir, uniqueName)

    # Save the annotated image (with face and eye boxes) and the processed face crop
    cv2.imwrite(annotatedPath, annotated)
    cv2.imwrite(processedPath, final_crop)

    knownNames.append(name)
    knownEmbeddings.append(embedding)
    total += 1

print("Embeddings: {}".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
with open(embeddingFile, "wb") as f:
    f.write(pickle.dumps(data))
print("Process Completed")
