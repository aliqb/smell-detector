import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load

from SmellDetector import SmellDetector

# def read_data(dataset_name):
#     data = pd.read_csv(f'datasets/{dataset_name}.csv')
#     X = data.drop(['IDType', 'project', 'package', 'complextype', 'is_smell'],
#                   axis=1)  # Features (remove the target column)
#     y = data['is_smell']  # Target variable
#     X.replace('?', -1, inplace=True)
#     for col in X.columns:
#         try:
#             X[col] = pd.to_numeric(X[col], errors='coerce')
#
#         except Exception as e:
#             print(f"Error in column {col}: {e}")
#     return X, y
#
#
# def balance_dataset(X, y):
#     # Step 1: Initialize your classifier
#     classifier = RandomForestClassifier()
#
#     # Step 2: Perform 3-fold cross-validation to get confidence scores
#     confidence_scores = cross_val_predict(classifier, X, y, cv=3, method='predict_proba', n_jobs=-1)
#
#     # Assuming the classifier outputs probabilities and you're interested in the second class (smelly)
#     confidences = confidence_scores[:, 0]  # Confidence of being smelly
#
#     # Step 3: Filter instances based on confidence threshold
#     threshold = 0.95
#
#     # Create a mask for non-smelly instances with high confidence
#     high_confidence_non_smelly_mask = (y == 0) & (confidences > threshold)
#
#     # Use the mask to filter out high-confidence non-smelly instances from the DataFrame
#     retain_mask = (y != 0) | (confidences <= threshold)
#     # Filter the dataset based on the mask
#     if isinstance(X, pd.DataFrame):
#         filtered_X = X[retain_mask].reset_index(drop=True)
#     else:  # Assuming numpy array
#         filtered_X = X[retain_mask]
#
#     if isinstance(y, pd.Series):
#         filtered_y = y[retain_mask].reset_index(drop=True)
#     else:  # Assuming numpy array
#         filtered_y = y[retain_mask]
#
#     return filtered_X, filtered_y
#     # filtered_df now contains your filtered dataset
#
#
# def train_classifier(X, y, model, scaling=False):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     scoring = ('accuracy', 'precision', 'recall', 'f1', 'roc_auc')
#     if scaling:
#         scaler = MinMaxScaler()
#         X_train = scaler.fit_transform(X_train)
#         X_test = scaler.transform(X_test)
#     X_train_balanced, y_train_balnced = balance_dataset(X_train, y_train)
#     cv_results = cross_validate(model, X_train_balanced, y_train_balnced, cv=10, scoring=scoring)
#     model_name = model.__class__.__name__
#     print("##############################################")
#     print(f"Model: {model_name}")
#     print("cross validation result")
#     print(f"Accuracy: {cv_results['test_accuracy'].mean():.2f}")
#     print(f"Precision: {cv_results['test_precision'].mean():.2f}")
#     print(f"Recall: {cv_results['test_recall'].mean():.2f}")
#     print(f"F1-Score: {cv_results['test_f1'].mean():.2f}")
#     print(f"roc_auc: {cv_results['test_roc_auc'].mean():.2f}")
#
#     model.fit(X_train_balanced, y_train_balnced)
#     print("-------------------------------------")
#     print("test result")
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     roc_auc = roc_auc_score(y_test, y_pred)
#
#     print(f'Accuracy: {accuracy:.2f}')
#     print(f'precision: {precision:.2f}')
#     print(f'recall: {recall:.2f}')
#     print(f'f1_score: {f1:.2f}')
#     print(f'roc_auc: {roc_auc:.2f}')


if __name__ == '__main__':
    datasets = ['data-class', 'god-class', 'long-method', 'feature-envy']
    print("Choose smell dataset")
    print("1: data class")
    print("2: god class")
    print("3: long method")
    print("4: feature envy")
    selected_dataset = int(input())
    if 1 <= selected_dataset <= 4:
        dataset_name = datasets[selected_dataset - 1]
        print(f"dataset:{dataset_name}")
        classifiers = [(DecisionTreeClassifier(), {'criterion': ['log_loss', 'gini'],
                                                   'max_depth': list(range(1, 101)) + [None]}),
                       (RandomForestClassifier(),
                        {'criterion': ['log_loss', 'gini'], 'max_depth': list(range(1, 101, 10)),
                         'n_estimators': list(range(1, 200, 20))}),
                       (HistGradientBoostingClassifier(), {
                           'max_depth': list(range(1, 101, 10)), 'max_iter': list(range(1, 200, 20))
                       }),
                       (SVC(kernel='rbf'), {
                           'C': [float(x) for x in range(200, 2001, 200)],
                           'gamma': [float(i) for i in range(0, 100, 10)] + ['scale', 'auto']
                       })]
        print('Choose your classifier:')
        for index in range(len(classifiers)):
            print(f"{index + 1}: {classifiers[index][0].__class__.__name__}")
        selected_classifier = int(input())
        if 1 <= selected_classifier <= len(classifiers) + 1:
            (model, param_grid) = classifiers[selected_classifier - 1]
            scaling = str(model.__class__.__name__).startswith('SVC')
            smell_detector = SmellDetector(dataset_name, model, param_grid=param_grid, scaling=scaling)
            commands = [smell_detector.train_classifier, smell_detector.test_classifier]
            print("1: train a model")
            print("2: test  model")
            selected_command = int(input())
            if 1 <= selected_command <= len(commands):
                commands[selected_command-1]()
        else:
            print("invalid input")
    else:
        print("invalid input")
