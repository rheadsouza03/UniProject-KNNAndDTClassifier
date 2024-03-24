# Name: Rhea D'Souza
# Student ID: 300627012
# Username: dsouzrhea

"""
Scikit library was restricted for this project.
"""

# Imports
import numpy as np
import pandas as pd
import argparse as ap


# KNN Classifier
class KNNClassifier:
    def __init__(self, k):
        self.k = k

    def euclidean(self, point, data):
        return np.sqrt(np.sum((data - point) ** 2, axis=1))

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        return self

    def predict(self, X_test):
        X_test = np.array(X_test)
        predictions = []
        distances = []
        for x in X_test:
            dist = self.euclidean(x, self.X_train)
            # Sort distances and get indices of k nearest neighbors
            nearest_indices = np.argsort(dist)[:self.k]
            # Retrieve labels of k nearest neighbors
            nearest_labels = self.y_train[nearest_indices]
            # Predict the class based on the majority vote
            prediction = np.argmax(np.bincount(nearest_labels))
            predictions.append(prediction)
            # Append distances of k nearest neighbors
            distances.append(dist[nearest_indices])
        return predictions, distances


# Min-Max Normalisation
def min_max_normalisation(train_data, test_data):
    minimum = train_data.min(axis=0)
    maximum = train_data.max(axis=0)

    X_train = (train_data - minimum) / (maximum - minimum)
    X_test = (test_data - minimum) / (maximum - minimum)
    return X_train, X_test


def parse_arguments():
    parser = ap.ArgumentParser(description='KNN Classification')

    # Creating arguments types to access the information that is
    parser.add_argument('train_file', type=str, help='File name for training data')
    parser.add_argument('test_file', type=str, help='File name for test data')
    parser.add_argument('out_file', type=str, help='File name for the data output')
    parser.add_argument('k', type=int, help='Number of neighbours')

    return parser.parse_args()


def accuracy_calculation(y_test, predictions, num_classes=None):
    # Calculate overall accuracy
    accuracy = np.mean(y_test == predictions)

    # If num_classes is not provided, infer it from y_test and predictions
    if num_classes is None:
        num_classes = max(np.max(y_test), np.max(predictions))

    # Initialize dictionaries to store counts of true positives, and total samples for each class
    class_counts = {class_label: {"true_positives": 0, "total": 0} for class_label in range(1, num_classes + 1)}

    # Loop through the test data and update counts for each class
    for true_label, pred_label in zip(y_test, predictions):
        if true_label == pred_label:
            class_counts[true_label]["true_positives"] += 1
        class_counts[true_label]["total"] += 1

    class_accuracy = {}
    # Calculate and print accuracy for each class
    for class_label, counts in class_counts.items():
        accuracy_for_class = counts["true_positives"] / counts["total"] if counts["total"] > 0 else 0
        class_accuracy[class_label] = accuracy_for_class

    return accuracy, class_accuracy


def main():
    # Parse commandline arguments
    args = parse_arguments()

    # Initialising Train Data into pandas DataFrame
    train = pd.read_csv('data_part1/' + args.train_file)
    X_train = train.drop(columns='class')
    y_train = train['class']

    # Initialising Test Data into pandas DataFrame
    test = pd.read_csv('data_part1/' + args.test_file)
    X_test = test.drop(columns='class')
    y_test = test['class']

    # Performing Min-max normalisation on the test and training data
    X_train_scaled, X_test_scaled = min_max_normalisation(X_train, X_test)

    # Perform KNN Classification
    knn = KNNClassifier(args.k).fit(X_train_scaled, y_train)
    y_pred, distances = knn.predict(X_test_scaled)

    distance_dict = {'y': y_test, 'y_pred': y_pred}
    for distance in distances:
        for i in range(args.k):
            key = "distance" + str(i + 1)
            if distance_dict.get(key) is None:
                distance_dict[key] = []
            distance_dict[key].append(distance[i])

    # Converting data collected into a csv output file with the given label
    df = pd.DataFrame(distance_dict)
    print(df)
    df.to_csv('data_part1/' + args.out_file, index=False)

    # Get accuracy
    total_accuracy_test, class_accuracy_test = accuracy_calculation(y_test, y_pred)
    print(f'Overall Accuracy: {total_accuracy_test * 100:.2f}%\nClass Accuracies: {class_accuracy_test}')


if __name__ == "__main__":
    main()
