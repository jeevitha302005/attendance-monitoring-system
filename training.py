import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import numpy as np

# Path to your embeddings file
embeddingsPath = "output/embeddings.pickle"

# Load the embeddings and names
data = pickle.loads(open(embeddingsPath, "rb").read())
embeddings = np.array(data["embeddings"])
names = data["names"]

# Encode the labels (student names)
le = LabelEncoder()
labels = le.fit_transform(names)

# Train the classifier (using a linear SVM here)
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(embeddings, labels)

# Save the trained face recognition model and label encoder to disk
modelPath = "output/recognizer.pickle"
lePath = "output/le.pickle"

with open(modelPath, "wb") as f:
    f.write(pickle.dumps(recognizer))
    
with open(lePath, "wb") as f:
    f.write(pickle.dumps(le))

print("Face recognition model and label encoder have been trained and saved.")
