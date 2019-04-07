## general imports
import random
import itertools 
from pprint import pprint  
import numpy as np
import pandas as pd  
from sklearn.model_selection import train_test_split 
import sklearn.metrics 

## project supplied imports
import exercise_data_builder
from solution3 import Submission

classifier_under_test = Submission()

# get the corpus data
print("getting the corpus data ...")
data = exercise_data_builder.build_verbs_data()
all_tokens = set(data['non_verbs'].keys()) | set(data['verbs'].keys()) | data['dual']

baseline_accuracy = len(data['non_verbs']) / len(all_tokens)

accuracies = []
for cross_validation_pass in range(4):
    print()
    print("starting cross-validation pass {}".format(cross_validation_pass))
    print("rebuilding the model under test ...")

    # building X, y for and getting a train-test split
    X_negative = list(data['non_verbs'])
    X_positive = list(data['verbs'])
    X_dual     = list(data['dual'])

    y_negative = [0] * len(X_negative)
    y_positive = [1] * len(X_positive)
    y_dual     = [2] * len(X_dual)

    X = X_negative + X_positive + X_dual
    y = y_negative + y_positive + y_dual

    assert len(X) == len(y)

    # get a train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)

    model = classifier_under_test.train(X_train, y_train)

    # test on the test split
    print("using the model under test's vectorization function to vectorize test data ...")
    y_pred = model.predict(list(map(classifier_under_test.vectorize, X_test)))
    accuracy_score = sklearn.metrics.accuracy_score(y_test, y_pred)
    margin_from_baseline = accuracy_score - baseline_accuracy
    accuracy_score = sklearn.metrics.accuracy_score(y_test, y_pred)
    accuracies.append(accuracy_score)
    print("baseline_accuracy: {:.3f}".format(baseline_accuracy))
    print("model under tests's accuracy: {:.3f}".format(accuracy_score))
    print("positive margin from baseline: {:.3f}".format(margin_from_baseline))


accuracy_score = np.average(accuracies)
accuracy_score_std = np.std(accuracies)

print()
print('-----------')
print('final score')
print("baseline_accuracy: {:.3f}".format(baseline_accuracy))
print("model under tests's cross-validation accuracy: {:.3f} (std: {:.5f})".format(accuracy_score, accuracy_score_std))
print("positive margin from baseline: {:.3f}".format(margin_from_baseline))
