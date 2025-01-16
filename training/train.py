import os
import argparse
import joblib
import numpy as np
from PIL import Image
from torchvision import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from face_recognition import FaceFeaturesExtractor, FaceRecogniser


MODEL_DIR_PATH = 'model'


def parse_args():
    parser = argparse.ArgumentParser(description='Script for training Face Recognition model.')
    parser.add_argument('-d', '--dataset-path', help='Path to folder with subfolders of images.', required=True)
    return parser.parse_args()


def dataset_to_embeddings(dataset, features_extractor):
    embeddings = []
    labels = []
    for img_path, label in dataset.samples:
        print(f"Processing: {img_path}")
        _, embedding = features_extractor(Image.open(img_path).convert('RGB'))
        if embedding is None:
            print(f"Could not find face in {img_path}")
            continue
        embeddings.append(embedding.flatten())
        labels.append(label)

    return np.stack(embeddings), labels


def main():
    args = parse_args()

    # Initialize face feature extractor
    features_extractor = FaceFeaturesExtractor()

    # Load dataset and compute embeddings
    dataset = datasets.ImageFolder(args.dataset_path)
    embeddings, labels = dataset_to_embeddings(dataset, features_extractor)

    # Perform train-test split
    embeddings_train, embeddings_test, labels_train, labels_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Train the model
    clf = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=10000)
    clf.fit(embeddings_train, labels_train)

    # Evaluate the model
    labels_pred = clf.predict(embeddings_test)
    print("Test Set Classification Report:")
    print(metrics.classification_report(labels_test, labels_pred))

    # Save model
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    if not os.path.isdir(MODEL_DIR_PATH):
        os.mkdir(MODEL_DIR_PATH)
    model_path = os.path.join(MODEL_DIR_PATH, 'face_recogniser.pkl')
    joblib.dump(FaceRecogniser(features_extractor, clf, idx_to_class), model_path)
    print(f"Model saved to {model_path}")


if __name__ == '__main__':
    main()
