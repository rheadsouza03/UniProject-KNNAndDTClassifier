# Course: COMP307
# Name: Rhea D'Souza
# Student ID: 300627012
# Username: dsouzrhea

"""
Decision Tree Classification Algorithm
Scikit library was restricted for this project.
"""

# Imports
import argparse as ap
import math
import pandas as pd
import numpy as np


# Decision Tree Classification
class DecisionTree:
    # TreeNode private inner class
    class _TreeNode:
        def __init__(self, class_count, parent_node=None,
                    split_feature_value=None, split_feature: str = None,
                     child_nodes: list = None, information_gain=0, entropy=0.0):
            self.class_count = class_count
            self.parent_node = parent_node
            self.child_nodes = child_nodes
            self.split_feature = split_feature
            self.split_feature_value = split_feature_value
            self.information_gain = information_gain
            self.entropy = entropy

    def __init__(self):
        # Initialising DT node fields
        self.dataset = None
        self.features = None
        self.root = None
        self._build_script = ""

    # Splitting methods (Entropy and Information Gain)
    def _entropy(self, class_count: dict):
        """
        Calculates the Entropy of the given node.
        Entropy Formula:
        Entropy = - sum_{i=1}^c(P(i) * log_2(P(i)))
        ## Note: 'P(i)' is the proportion of instances in the node that belongs to class 'i'

        :return: Entropy of the given node
        """
        total_instances = sum(class_count.values())
        entropy = 0.0
        for (key, value) in class_count.items():
            if value == 0:
                continue
            probability = value / total_instances
            entropy -= probability * math.log2(probability)
        return entropy

    def _info_gain(self, entropy_parent, entropy_children: list, child_weights: list):
        """
        Information Gained Formula (Binary Version):
        IG(F) = H(P) - [((N_L)/(N_P)) * H(L) + ((N_R)/(N_P)) * H(R)]

        Information Gained Formula (General Version):
        IG(F) = H(P) - sum_{l=1}^k(((N_l)/(N_P)) * H(l))
        ## Note: 'P' is the parent/current node and 'H' represents the weighted entropy

        :return: Informational gained for the given node.
        """
        total_instances = sum(child_weights)
        weighted_child_entropy = sum((child_weight / total_instances) * child_entropy
                                     for child_weight, child_entropy in zip(child_weights, entropy_children))
        return entropy_parent - weighted_child_entropy

    def _get_child_weights(self, parent_instances, child_instances) -> list:
        """
        Calculates the weights of the children instances against the parent instances.
        :param parent_instances: Dataframe containing only the features of the parent instances.
        :param child_instances: Dataframe containing only the features of the child instances.
        :return: Weight of the children instances in a list.
        """
        return [instances / parent_instances for instances in child_instances]

    def _calculate_class_counts(self, class_subset) -> dict:
        """
        Calculates the class counts for the given class subset.
        :param class_subset: The classes of the instance within the current node.
        :return: Count of all the classes as an integer.
        """
        labels = set(class_subset)
        class_counts = {0: 0, 1: 0}
        for label in class_subset:
            class_counts[label] += 1
        return class_counts

    def _split_data(self, dataset: pd.DataFrame, feature_name, split_value):
        """
        The split that is performed is a binary split. In which the data is split
        according to the split value into 2 subsets.
        :param dataset: Whole dataset of the node (including classes)
        :param feature_values: The column of the feature that is used for the split
        :param split_value: Value to split the dataset into
        :return: left and right subset of the binary feature values as pandas DataFrames
        """
        left_subset = dataset[dataset[feature_name] == split_value]
        right_subset = dataset[dataset[feature_name] != split_value]

        return left_subset, right_subset

    # Building the train datas decision tree
    def _find_best_split(self, class_counts, dataset, features):
        """
        Finds the best split in the dataset.
        :param class_counts: Current node's class counts
        :param dataset: Data stored in the node
        :param features: Unused features for the current branch
        :return: entropy, information gain, the best split feature and best split value.
        """
        parent_entropy = self._entropy(class_counts)
        best_information_gain = -float('inf')
        best_feature = ''
        for feature in features:
            # Split data based on feature
            left_subset, right_subset = self._split_data(dataset, feature, 0)

            # Calculate Entropy of subsets
            left_entropy = self._entropy(self._calculate_class_counts(left_subset['class']))
            right_entropy = self._entropy(self._calculate_class_counts(right_subset['class']))

            # Calculate Information Gain (IG) and Entropy for the split
            gain = self._info_gain(parent_entropy, [left_entropy, right_entropy],
                                   [len(left_subset), len(right_subset)])

            # Updates IG and Entropy if new IG is larger and new Entropy is smaller
            if gain > best_information_gain:
                best_information_gain = gain
                best_feature = feature

        return parent_entropy, best_information_gain, best_feature, 0

    def _build_tree(self, data, features, node: _TreeNode, current_tree_depth=0):
        #
        # for each leaf_node:
        #   Compute if the set of instances is as pure as possible
        # if set is not pure:
        #   Select the best (unused in that path) feature as the next node (lowest impurity)
        #   Split training data into subsets according to the chosen feature's possible values
        #   Recurse on each of the subsets
        #

        indentation = current_tree_depth * "\t"
        node.class_count = self._calculate_class_counts(data['class'])

        if not ((node.class_count[0] == 0) | (node.class_count[1] == 0)):
            # Splitting the data at the best feature
            entropy, information_gain, split_feature, split_value = (
                self._find_best_split(node.class_count, data, features))
            left_subset, right_subset = self._split_data(data, split_feature, 0)

            self._build_script += f"{indentation}{split_feature} (IG: {information_gain:.4f}, Entropy: {entropy:.4f})\n"

            # Update info
            features.remove(split_feature)
            node.split_feature = split_feature
            node.entropy = entropy
            node.information_gain = information_gain

            # Creating left node
            left_node = self._TreeNode(self._calculate_class_counts(left_subset['class']), parent_node=node, split_feature_value=0)
            self._build_script += f"{indentation}--{split_feature} == {left_node.split_feature_value}--\n"
            left_node = self._build_tree(left_subset, features.copy(), left_node, current_tree_depth + 1)

            # Creating right node
            right_node = self._TreeNode(self._calculate_class_counts(right_subset['class']), parent_node=node, split_feature_value=1)
            self._build_script += f"{indentation}--{split_feature} == {right_node.split_feature_value}--\n"
            right_node = self._build_tree(right_subset, features.copy(), right_node, current_tree_depth + 1)

            # Update node and features
            node.child_nodes = [left_node, right_node]
        if (current_tree_depth > 0) & (node.child_nodes is None):
            self._build_script += f"{indentation}leaf{node.class_count}\n"
        return node

    def get_script(self):
        '''
        Returns a string representation of the decision tree traversed through a depth-first search.
        :return: String of the decision tree.
        '''
        return self._build_script

    def fit(self, data):
        """
        Fits the decision tree by building and initializing all the key fields
        :param data: training data to be used for the decision tree
        :return: the current instance of the object
        """
        self.dataset = data
        self.features = data.columns.tolist()
        self.features.remove("class")
        self.root = self._build_tree(data, self.features.copy(), self._TreeNode(self._calculate_class_counts(data['class'])))
        return self

    def _traverse_tree(self, instance, features, tree_node):
        if tree_node.split_feature is None:
            return tree_node.class_count

        feature_index = features.index(tree_node.split_feature)
        for child in tree_node.child_nodes:
            if child.split_feature_value == instance[feature_index]:
                features.remove(tree_node.split_feature)
                return self._traverse_tree(instance, features, child)

    def predict(self, X) -> pd.DataFrame:
        predictions = []
        X_test = np.array(X)
        for x in X_test:
            class_count = self._traverse_tree(x, self.features.copy(), self.root)
            for (key, value) in class_count.items():
                if value != 0:
                    predictions.append(key)
                    break

        return predictions


# Commandline arguments
def parse_arguments():
    parser = ap.ArgumentParser(description='Decision Tree Classification')

    # Creating arguments types to access the information that is
    parser.add_argument('train_file', type=str, help='File name for training data')
    parser.add_argument('out_file', type=str, help='File name for the data output')

    return parser.parse_args()


# Calculating the accuracy
def accuracy_calculation(y_test, predictions, num_classes=None):
    # Overall accuracy
    accuracy = np.mean(y_test == predictions)

    # If num_classes is not provided, infer it from y_test and predictions
    if num_classes is None:
        num_classes = max(np.max(y_test), np.max(predictions)) + 1

    # Initialize dictionaries to store counts of true positives, and total samples for each class
    class_counts = {class_label: {"true_positives": 0, "total": 0} for class_label in range(0, num_classes)}

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


def save_tree(string, filename):
    with open('data_part2/' + filename, 'w') as file:
        file.write(string)
    print("File created successfully. Can be found in: data_part2/" + filename)


# main() method
def main():
    """
    This method will print the structure of the output file before creation
    and will print the accuracy of the Decision Tree Classification.

    Ensure the command line arguments are present.
    Expected structure:
        % python k_nearest_neighbour.py <train-data>.csv <output-file>.txt
    For Example:
        % python decision_tree.py rtg_A.csv DT_A.txt
        OR
        % python decision_tree.py rtg_B.csv DT_B.txt

    ## Note: The given files are expected to be in the 'data_part2/' directory
    """
    # Gets the expected parameters from the terminal
    args = parse_arguments()

    """
    In this assignment train_data == test_data, so we will just copy the data to test
    """
    train = pd.read_csv("data_part2/" + args.train_file)
    X_test = train.drop(columns=["class"])
    y_test = train['class']

    dt = DecisionTree().fit(train)
    # Gets the depth-first search string of the decision tree and saves it to the given txt filename
    tree_string = dt.get_script()
    save_tree(tree_string, args.out_file)

    y_pred = dt.predict(X_test)
    accuracy, class_accuracy = accuracy_calculation(y_test, y_pred)
    print(f'Accuracy Information:\nOverall Accuracy - {accuracy * 100:.2f}%\nClass Accuracies - {class_accuracy}')


if __name__ == "__main__":
    main()
