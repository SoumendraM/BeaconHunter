import pandas as pd
from features import create_derived_features
from utils import process_features
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

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


    # Implement Random Forest Classifier
    
    #rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, random_state=42)
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)

    print("Random Forest Classifier Performance:")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # Create risk score based on predicted probabilities
    y_prob = rf_classifier.predict_proba(X)[:, 1]
    risk_scores = MinMaxScaler().fit_transform(y_prob.reshape(-1, 1))
    print("Sample Risk Scores:")
    #print(risk_scores[10:100])

    # calcolate the ROC-AUC score for both models
    from sklearn.metrics import roc_auc_score
    rf_roc_auc = roc_auc_score(y_test, rf_classifier.predict_proba(X_test)[:, 1])
    print(f"Random Forest ROC-AUC: {rf_roc_auc}")

    # calculate PR-AUC score
    from sklearn.metrics import precision_recall_curve, auc
    precision, recall, _ = precision_recall_curve(y_test, rf_classifier.predict_proba(X_test)[:, 1])
    rf_pr_auc = auc(recall, precision)  
    print(f"Random Forest PR-AUC: {rf_pr_auc}")

    # save the trained model using joblib
    import joblib
    joblib.dump(rf_classifier, r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\artifacts\rf_classifier_model.joblib')
    return rf_classifier

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
    beacon_df.to_csv(r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\\artifacts\\beacon_events_if_results.csv', index=False)

    # save the trained model using joblib
    joblib.dump(isolation_forest, r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\artifacts\isolation_forest_model.joblib')
    return isolation_forest
