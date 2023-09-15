import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Data Collection (prepare a dataset with labeled audio clips)

# Feature Extraction (using MFCCs)
def extract_features(audio_path):
    y, sr = librosa.load(audio_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)  # Calculate the mean of MFCCs as features

# Example data structure for your dataset:
# data = [
#     {"path": "dog_barking_1.wav", "label": "dog_barking"},
#     {"path": "engine_noise_1.wav", "label": "engine_noise"},
#     ...
# ]

X = []
y = []

# Extract features and labels
for item in data:
    features = extract_features(item["path"])
    X.append(features)
    y.append(item["label"])

# Step 3: Data Preprocessing
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Selection (SVM classifier)
clf = SVC(kernel='linear')

# Model Training
clf.fit(X_train, y_train)

# Model Evaluation
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# You can use the trained model to classify new audio clips.
# For example, load a new audio clip, extract features, and use clf.predict() to get the prediction.
