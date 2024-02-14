import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load
from sklearn.model_selection import GridSearchCV
from lime import lime_tabular
import matplotlib
import matplotlib.pyplot as plt
import re

matplotlib.use('TkAgg')


class SmellDetector:
    def __init__(self, dataset_name, model, param_grid=None, scaling=False):
        self.X_test = None
        self.X_train = None
        self.y_test = None
        self.y_train = None
        self.X_train_balanced = None
        self.y_train_balanced = None
        self.last_model = None
        # self.trained_model = None
        # if isinstance(model, SVC):
        #     model.probability = True  # Ensure probability is enabled for SVC
        self.model = model
        self.dataset_name = dataset_name
        self.scaling = scaling
        self.is_fitted = False
        self.X, self.y = self.read_data()
        self.make_train_test_sets()
        self.param_grid = param_grid
        self.set_last_model()

    def read_data(self):
        data = pd.read_csv(f'datasets/{self.dataset_name}.csv')
        none_data_cols = ['project', 'package', 'complextype', 'is_smell']
        if 'IDType' in data.columns:
            none_data_cols.append('IDType')
        if 'IDMethod' in data.columns:
            none_data_cols.append('IDMethod')
        if 'method' in data.columns:
            none_data_cols.append('method')
        X = data.drop(none_data_cols,
                      axis=1)  # Features (remove the target column)
        y = data['is_smell']  # Target variable
        X.replace('?', -1, inplace=True)
        for col in X.columns:
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')

            except Exception as e:
                print(f"Error in column {col}: {e}")
        nan_locs_X = np.where(X.isnull())
        # nan_indices_X = list(zip(nan_locs_X[0], nan_locs_X[1]))

        # Print out the indices of NaN values in X
        # for row, col in nan_indices_X:
        #     print(f"NaN in X at row {row}, column {X.columns[col]}")
        return X, y

    def make_train_test_sets(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=42)
        if self.scaling:
            scaler = MinMaxScaler()
            self.X_train = pd.DataFrame(scaler.fit_transform(self.X_train), columns=self.X_train.columns,
                                        index=self.X_train.index)

            # Transform the test data and convert back to DataFrame
            self.X_test = pd.DataFrame(scaler.transform(self.X_test), columns=self.X_test.columns,
                                       index=self.X_test.index)
        self.X_train_balanced, self.y_train_balanced = self.balance_dataset(self.X_train, self.y_train)

    def set_last_model(self):
        try:
            self.model = load(f"./models/{self.dataset_name}-{self.model.__class__.__name__}")
            self.is_fitted = True
            print("trained model exist")
        except Exception as e:
            self.is_fitted = False
            print("trained model does not exist")

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
        dump(self.model, f'./models/{self.dataset_name}-{self.model.__class__.__name__}')

    def find_best_model(self):
        grid_search = GridSearchCV(estimator=self.model, param_grid=self.param_grid, cv=10, scoring='accuracy',
                                   verbose=1,
                                   n_jobs=-1)
        grid_search.fit(self.X_train_balanced, self.y_train_balanced)
        return grid_search.best_estimator_, grid_search.best_params_

    def train_classifier(self):
        scoring = ('accuracy', 'precision', 'recall', 'f1', 'roc_auc')
        # if self.param_grid is not None:
        self.model, best_params = self.find_best_model()
        if isinstance(self.model, SVC):
            self.model.probability = True  # Ensure probability is enabled for SVC
        print(f"best params: {best_params}")
        # else:
        #     best_model = self.model
        cv_results = cross_validate(self.model, self.X_train_balanced, self.y_train_balanced, cv=10, scoring=scoring)
        self.is_fitted = True
        self.save_model()
        print("##############################################")
        print(f"Model: {self.model.__class__.__name__}")
        print("cross validation result")
        print(f"Accuracy: {(cv_results['test_accuracy'].mean()*100):.2f}")
        print(f"roc_auc: {(cv_results['test_roc_auc'].mean()*100):.2f}")
        print(f"F1-Score: {(cv_results['test_f1'].mean()*100):.2f}")
        print(f"Recall: {(cv_results['test_recall'].mean()*100):.2f}")
        print(f"Precision: {(cv_results['test_precision'].mean()*100):.2f}")

        # self.model.fit(self.X_train_balanced, self.y_train_balanced)
        # self.test_classifier()

    def test_classifier(self):
        print("-------------------------------------")
        print("test result")
        # self.set_last_model()
        if not self.is_fitted:
            print("there is no trained model, training start:")
            self.train_classifier()
            return
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred)

        print(f'Accuracy: {(accuracy*100):.2f}')
        print(f'recall: {(recall*100):.2f}')
        print(f'precision: {(precision*100):.2f}')
        print(f'f1_score: {(f1*100):.2f}')
        print(f'roc_auc: {(roc_auc*100):.2f}')

    def explain_instance(self, instance_index, num_features=10):
        if not self.is_fitted:
            print("there is no trained model, training start:")
            self.train_classifier()
            return
        # Create a Lime explainer object
        # Note: There are different explainers in LIME for different types of data (e.g., text, tabular, and images).
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=np.array(self.X_train_balanced),
            feature_names=self.X_train_balanced.columns.tolist(),
            class_names=['Non-smelly', 'Smelly'],
            mode='classification'
        )

        # Wrap the model's predict_proba to include feature names
        def model_predict_proba(data):
            # Convert the numpy array data back to DataFrame with feature names
            df = pd.DataFrame(data, columns=self.X_train_balanced.columns)
            # Use the original model's predict_proba on the DataFrame
            return self.model.predict_proba(df)

        # Explain a prediction
        exp = explainer.explain_instance(
            data_row=np.array(self.X_test.iloc[instance_index]),
            predict_fn=model_predict_proba,  # Use the wrapped predict function
            num_features=num_features
        )

        # Display the explanation
        # exp.show_in_notebook(show_table=True)
        # fig = exp.as_pyplot_figure()
        # For non-notebook environments, you can use exp.as_pyplot_figure()
        # plt.show(block=True)
        # print(exp.as_list())
        return exp.as_list()

    def get_global_importance(self, num_features=10):
        # Select a sample of instances to explain
        np.random.seed(42)  # For reproducibility
        # sample_indices = np.random.choice(self.X_test.shape[0], num_samples, replace=False)
        feature_importances = {}

        for idx in range(self.X_test.shape[0]):
            exp_list = self.explain_instance(idx, num_features)
            for feature, importance in exp_list:
                match = re.search('(\d*\.?\d*\s*<=?)?\s*([A-Za-z_]*)\s*(\d*\.?\d*\s*[><]=?)?', feature)
                main_feature_name = match.group(2)
                if main_feature_name in feature_importances:
                    feature_importances[main_feature_name].append(importance)
                else:
                    feature_importances[main_feature_name] = [importance]

        # Aggregate the feature importances
        aggregated_importances = {feature: np.mean(importances) for feature, importances in feature_importances.items()}
        sorted_importances = sorted(aggregated_importances.items(), key=lambda x: x[1], reverse=True)
        print(sorted_importances)
        return sorted_importances
