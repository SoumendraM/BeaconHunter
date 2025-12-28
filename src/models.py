import pandas as pd
import numpy as np
from features import create_derived_features
from utils import process_features 
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.metrics import make_scorer, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import os
from pathlib import Path


class RandomForestModel:
    def __init__(self, 
                 n_estimators=50, 
                 max_depth=10, 
                 min_samples_split=5, 
                 random_state=42, 
                 model_path=None):
        self.model_path = model_path
        if model_path == None:
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=random_state
            )
        elif model_path != None:
            file_path = Path(model_path)
            if file_path.is_file():
                self.load(model_path)
            else:
                self.model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=random_state
            )

    def load(self, path: str):
        self.model = joblib.load(path)
        self.model_path = path

        return self      

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        return self.model.predict_proba(X)

    def risk_scores(self, X: pd.DataFrame):
        probs = self.predict_proba(X)[:, 1]
        return MinMaxScaler().fit_transform(probs.reshape(-1, 1)).ravel()

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series):
        y_pred = self.predict(X_test)
        print("Random Forest Classifier Performance:")
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        roc = roc_auc_score(y_test, self.predict_proba(X_test)[:, 1])
        print(f"Random Forest ROC-AUC: {roc}")
        precision, recall, _ = precision_recall_curve(y_test, self.predict_proba(X_test)[:, 1])
        pr_auc = auc(recall, precision)
        print(f"Random Forest PR-AUC: {pr_auc}")
        return {"roc_auc": roc, "pr_auc": pr_auc}

    def save(self, path: str = None):
        path = path or self.model_path
        if not path:
            raise ValueError("No path provided to save the model.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

   
    # Get tuned classifier threshhold
    def evaluate_threshhold(self, default_threshold: float = 0.5):
        y_probs = self.predict_proba(self.X_test)[:, 1]

        #default_threshold = 0.8
        y_pred_default = (y_probs >= default_threshold).astype(int)
        cm = confusion_matrix(self.y_test, y_pred_default)
        print(f"Confusion Matrix (Threshold={default_threshold:0.2f}):\n{cm}")

        # Precision/Recall at default threshold
        precision = precision_score(self.y_test, y_pred_default)
        recall = recall_score(self.y_test, y_pred_default)
        print(f"Precision (Threshold={default_threshold:0.2f}): {precision:.3f}")
        print(f"Recall (Threshold={default_threshold:0.2f}): {recall:.3f}")

        #ROC-AUC Score (threshold independent)
        roc_auc = roc_auc_score(self.y_test, y_probs)
        print(f"\nROC-AUC Score: {roc_auc:.3f}")

    def evaluate_threshold_test(self, X_test: pd.DataFrame, y_test: pd.Series, default_threshold: float = 0.5):
        y_probs = self.predict_proba(X_test)[:, 1]

        #default_threshold = 0.8
        y_pred_default = (y_probs >= default_threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred_default)
        print(f"Confusion Matrix (Threshold={default_threshold:0.2f}):\n{cm}")

        # Precision/Recall at default threshold
        precision = precision_score(y_test, y_pred_default)
        recall = recall_score(y_test, y_pred_default)
        print(f"Precision (Threshold={default_threshold:0.2f}): {precision:.3f}")
        print(f"Recall (Threshold={default_threshold:0.2f}): {recall:.3f}")

        #ROC-AUC Score (threshold independent)
        roc_auc = roc_auc_score(y_test, y_pred_default)
        print(f"\nROC-AUC Score: {roc_auc:.3f}")

    r'''    
    def get_optimal_threshhold_for_test(self, X_test: pd.DataFrame, y_test: pd.Series) -> np.float64:
        # Finding an Optimal Threshold (e.g., balancing Precision and Recall)
        # Get all thresholds for Precision-Recall curve
        y_probs = self.predict_proba(X_test)[:, 1]
        precision_points, recall_points, thresholds_pr = precision_recall_curve(y_test, y_probs)

        # Find the threshold that balances precision and recall (e.g., closest to precision=recall)\
        fscore = (2 * precision_points * recall_points) / (precision_points + recall_points)

        # find the optimal threshold for the highest f-score
        ix = np.argmax(fscore)
        optimal_threshold_pr = thresholds_pr[ix]
        print(f"\nOptimal Threshold (Max F1 Score): {optimal_threshold_pr:.3f}")
        print(f"Precision: {precision_points[ix]:.3f}, Recall: {recall_points[ix]:.3f}")

        y_pred_optimal = (y_probs >= optimal_threshold_pr).astype(int)
        print(f"Confusion Matrix (Optimal Threshold):\n{confusion_matrix(y_test, y_pred_optimal)}")
        print(type(optimal_threshold_pr))
        return optimal_threshold_pr
    '''
    def get_optimal_threshhold(self) -> np.float64:
        # Finding an Optimal Threshold (e.g., balancing Precision and Recall)
        # Get all thresholds for Precision-Recall curve
        y_probs = self.predict_proba(self.X_test)[:, 1]
        precision_points, recall_points, thresholds_pr = precision_recall_curve(self.y_test, y_probs)

        # Find the threshold that balances precision and recall (e.g., closest to precision=recall)\
        fscore = (2 * precision_points * recall_points) / (precision_points + recall_points)

        # find the optimal threshold for the highest f-score
        ix = np.argmax(fscore)
        optimal_threshold_pr = thresholds_pr[ix]
        print(f"\nOptimal Threshold (Max F1 Score): {optimal_threshold_pr:.3f}")
        print(f"Precision: {precision_points[ix]:.3f}, Recall: {recall_points[ix]:.3f}")

        y_pred_optimal = (y_probs >= optimal_threshold_pr).astype(int)
        print(f"Confusion Matrix (Optimal Threshold):\n{confusion_matrix(self.y_test, y_pred_optimal)}")
        print(type(optimal_threshold_pr))
        return optimal_threshold_pr

def get_optimal_threshhold_for_test(rf_model: RandomForestModel , X_test: pd.DataFrame, y_test: pd.Series) -> np.float64:
    # Finding an Optimal Threshold (e.g., balancing Precision and Recall)
    # Get all thresholds for Precision-Recall curve
    y_probs = rf_model.predict_proba(X_test)[:, 1]
    precision_points, recall_points, thresholds_pr = precision_recall_curve(y_test, y_probs)

    # Find the threshold that balances precision and recall (e.g., closest to precision=recall)\
    fscore = (2 * precision_points * recall_points) / (precision_points + recall_points)

    # find the optimal threshold for the highest f-score
    ix = np.argmax(fscore)
    optimal_threshold_pr = thresholds_pr[ix]
    print(f"\nOptimal Threshold (Max F1 Score): {optimal_threshold_pr:.3f}")
    print(f"Precision: {precision_points[ix]:.3f}, Recall: {recall_points[ix]:.3f}")
    y_pred_optimal = (y_probs >= optimal_threshold_pr).astype(int)
    print(f"Confusion Matrix (Optimal Threshold):\n{confusion_matrix(y_test, y_pred_optimal)}")
    print(type(optimal_threshold_pr))
    return optimal_threshold_pr


def train_supervised_models(beacon_df: pd.DataFrame):
    beacon_df = process_features(beacon_df)
    beacon_df.to_csv(r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\\artifacts\\beacon_events_train_rf_processed.csv', index=False)
   
    # Separate features and target
    X = beacon_df.drop(columns=['label'])
    y = beacon_df['label']

    # Split the dataste into training and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train.to_csv(r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\\data\\X_train.csv', index=False)
    X_test.to_csv(r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\\data\\X_test.csv', index=False)
    y_train.to_csv(r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\\data\\y_train.csv', index=False)
    y_test.to_csv(r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\\data\\y_test.csv', index=False)
    print("Training and test sets saved.")

    model_path=r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\artifacts\rf_classifier_model.joblib'
    # Use RandomForestModel class
    rf_model = RandomForestModel(n_estimators=50, 
                                 max_depth=10, 
                                 min_samples_split=5, 
                                 random_state=42,
                                 model_path=model_path)
                                 #model_path=None)
    
    rf_model.fit(X_train, y_train)
    rf_model.save()
    #rf_model.evaluate(X_test, y_test)

    # Create risk score based on predicted probabilities for full dataset
    risk_scores = rf_model.risk_scores(X)
    # (optional) attach to original dataframe
    beacon_df['risk_score'] = risk_scores

    #for threshold in np.arange(0.2, 1.0, 0.1):
    #    rf_model.evaluate_threshhold(threshold)
    #optimal_threshold_pr = rf_model.get_optimal_threshhold()
    #print(f"\nOptimal Threshold (Max F1 Score): {optimal_threshold_pr:0.3f}")

    print("EVALUATE")
    print(get_optimal_threshhold_for_test(rf_model, X_test=X_test, y_test=y_test))


    # save the trained model using joblib via class helper
    #rf_model.save()
    return rf_model

def train_unsupervised_model(beacon_df: pd.DataFrame):
    n_estimators = 100 # Number of trees in the forest 
    contamination = 0.01 # Expected proportion of outliers in the data
    random_state = 42
    sample_size = 256 # Number of samples to draw to train each base estimator
    isolation_forest = IsolationForest(n_estimators=n_estimators, 
                                       contamination=contamination,
                                        max_samples=sample_size,
                                       random_state=random_state)

    beacon_if_df = beacon_df.copy()
 
    beacon_df = process_features(beacon_if_df)
    beacon_df.to_csv(r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\\artifacts\\beacon_events_train_if_processed.csv', index=False)

    # remove label column if exists
    if 'label' in beacon_df.columns:
        beacon_df = beacon_df.drop(columns=['label'])
    
    isolation_forest.fit(beacon_df)

    #calculate anomaly scores for each record
    anomaly_scores = isolation_forest.decision_function(beacon_df)
    #beacon_if_df['anomaly_score'] = anomaly_scores

    # Get anomaly predictions
    anomaly_predictions = isolation_forest.predict(beacon_df)
    beacon_df['anomaly_prediction'] = anomaly_predictions

    # Convert predictions from -1/1 to 1/0
    beacon_df['anomaly_prediction'] = beacon_df['anomaly_prediction'].map({1: 0, -1: 1})

    # map anomaly score between 0 and 1
    beacon_df['anomaly_score'] = anomaly_scores
    scaler = MinMaxScaler()
    beacon_df['anomaly_score_scaled'] = scaler.fit_transform(beacon_df[['anomaly_score']])

    # Save the results to a CSV file
    beacon_df.to_csv(r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\artifacts\beacon_events_if_results.csv', index=False)

    # save the trained model using joblib
    joblib.dump(isolation_forest, r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\artifacts\isolation_forest_model.joblib')
    return isolation_forest
