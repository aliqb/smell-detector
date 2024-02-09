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
from sklearn.model_selection import GridSearchCV


class SmellDetector:
    def __init__(self, dataset_name, model, param_grid=None):
        self.X_train_balanced = None
        self.y_train_balanced = None
        self.model = model
        self.dataset_name = dataset_name
        self.X, self.y = self.read_data()
        self.param_grid = param_grid

    def read_data(self):
        data = pd.read_csv(f'datasets/{self.dataset_name}.csv')
        none_data_cols = ['project', 'package', 'complextype', 'is_smell']
        if 'IDType' in data.columns:
            none_data_cols.append('IDType')
        if 'IDMethod' in data.columns:
            none_data_cols.append('IDMethod')
        X = data.drop(none_data_cols,
                      axis=1)  # Features (remove the target column)
        y = data['is_smell']  # Target variable
        X.replace('?', -1, inplace=True)
        for col in X.columns:
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')

            except Exception as e:
                print(f"Error in column {col}: {e}")
        return X, y

    def balance_dataset(self, X, y):
        # Step 1: Initialize your classifier
        classifier = RandomForestClassifier()

        # Step 2: Perform 3-fold cross-validation to get confidence scores
        confidence_scores = cross_val_predict(classifier, X, y, cv=3, method='predict_proba', n_jobs=-1)

        # Assuming the classifier outputs probabilities and you're interested in the second class (smelly)
        confidences = confidence_scores[:, 0]  # Confidence of being smelly

        # Step 3: Filter instances based on confidence threshold
        threshold = 0.95

        # Create a mask for non-smelly instances with high confidence
        high_confidence_non_smelly_mask = (y == 0) & (confidences > threshold)

        # Use the mask to filter out high-confidence non-smelly instances from the DataFrame
        retain_mask = (y != 0) | (confidences <= threshold)
        # Filter the dataset based on the mask
        if isinstance(X, pd.DataFrame):
            filtered_X = X[retain_mask].reset_index(drop=True)
        else:  # Assuming numpy array
            filtered_X = X[retain_mask]

        if isinstance(y, pd.Series):
            filtered_y = y[retain_mask].reset_index(drop=True)
        else:  # Assuming numpy array
            filtered_y = y[retain_mask]

        return filtered_X, filtered_y
        # filtered_df now contains your filtered dataset

    def save_model(self):
        dump(self.model, f'{self.dataset_name}-{self.model.__class__.__name__}')

    def find_best_model(self):
        grid_search = GridSearchCV(estimator=self.model, param_grid=self.param_grid, cv=10, scoring='accuracy',
                                   verbose=1,
                                   n_jobs=-1)
        grid_search.fit(self.X_train_balanced, self.y_train_balanced)
        return grid_search.best_estimator_, grid_search.best_params_

    def train_classifier(self, scaling=False):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        scoring = ('accuracy', 'precision', 'recall', 'f1', 'roc_auc')
        if scaling:
            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        self.X_train_balanced, self.y_train_balanced = self.balance_dataset(X_train, y_train)

        if self.param_grid is not None:
            best_model, best_params = self.find_best_model()
            print(f"best params: {best_params}")
        else:
            best_model = self.model
        cv_results = cross_validate(best_model, self.X_train_balanced, self.y_train_balanced, cv=10, scoring=scoring)
        model_name = self.model.__class__.__name__
        print("##############################################")
        print(f"Model: {model_name}")
        print("cross validation result")
        print(f"Accuracy: {cv_results['test_accuracy'].mean():.2f}")
        print(f"Precision: {cv_results['test_precision'].mean():.2f}")
        print(f"Recall: {cv_results['test_recall'].mean():.2f}")
        print(f"F1-Score: {cv_results['test_f1'].mean():.2f}")
        print(f"roc_auc: {cv_results['test_roc_auc'].mean():.2f}")

        # self.model.fit(self.X_train_balanced, self.y_train_balanced)
        print("-------------------------------------")
        print("test result")
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)

        print(f'Accuracy: {accuracy:.2f}')
        print(f'precision: {precision:.2f}')
        print(f'recall: {recall:.2f}')
        print(f'f1_score: {f1:.2f}')
        print(f'roc_auc: {roc_auc:.2f}')
