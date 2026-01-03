from utils import category_cols_to_category_dtype, categorical_cols_to_dummies, process_features, scale_numerical_features
from features import create_derived_features
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from models import RandomForestModel, calculate_fusion_risk_scores
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def main():
    baecon_test_df = pd.read_csv(r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\data\beacon_events_test_labeled.csv')

    df_processed = baecon_test_df.copy()
    df_processed = process_features(df_processed)
    df_processed_X = df_processed.drop(columns=['label'])
    df_processed_y = baecon_test_df['label']

    # Load  the trained Random Forest model
    model_path=r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\artifacts\rf_classifier_model.joblib'
    rf_classifier = RandomForestModel(n_estimators=50, 
                                                                    max_depth=10, 
                                                                    min_samples_split=5, 
                                                                    random_state=42,
                                                                    model_path=model_path)
    baecon_test_df['risk_score'] = rf_classifier.risk_scores(df_processed_X)
    #rf_classifier.evaluate_threshold_test(df_processed_X, df_processed_y, 0.391)

    risk_threshold = rf_classifier.get_optimal_threshhold(X_test=df_processed_X, y_test=df_processed_y)
    print("Risk Score (threshold):", risk_threshold)

    print("Evaluate with optimal threshhold:")
    rf_classifier.evaluate_with_threshold(df_processed_X, df_processed_y, risk_threshold)

    for tr in np.arange(0.2, 1.0, 0.1):
        print("Threshold: %.2f" % tr)
        rf_classifier.evaluate_with_threshold(df_processed_X, df_processed_y, tr)

    # Load trained Isolation Forest model
    isolation_forest = joblib.load(r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\artifacts\isolation_forest_model.joblib') 
    #baecon_test_df['anomaly_prediction'], baecon_test_df['anomaly_score']

    baecon_test_df['anomaly_score'] = isolation_forest.decision_function(df_processed_X)
    scaler = MinMaxScaler()
    baecon_test_df['anomaly_score_scaled'] = scaler.fit_transform(baecon_test_df[['anomaly_score']])

    anomaly_predictions = isolation_forest.predict(df_processed_X)
    baecon_test_df['anomaly_prediction'] = anomaly_predictions

    # Convert predictions from -1/1 to 1/0
    baecon_test_df['anomaly_prediction'] = baecon_test_df['anomaly_prediction'].map({1: 0, -1: 1})
    #beacon_if_df['anomaly_score'] = anomaly_scores

    
    baecon_test_df['fusion_risk_score'] = calculate_fusion_risk_scores(beacon_df_processed=baecon_test_df)

    high_risk_low_label = baecon_test_df[(baecon_test_df['fusion_risk_score'] > 0.55) & (baecon_test_df['label'] == 0)]
    print("Events with High risk scores with low labels saved to artifacts\\high_risk_low_label_samples.csv") 
    # Save to csv
    high_risk_low_label.head(5).to_csv(r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\artifacts\high_risk_low_label_samples.csv', index=False)   



    low_risk_high_label = baecon_test_df[(baecon_test_df['fusion_risk_score'] < 0.55) & (baecon_test_df['label'] == 1)]
    print("Events with Low risk scores, high labels saved to artifacts\\low_risk_high_label_samples.csv")
    # Save to csv
    low_risk_high_label.head(5).to_csv(r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\artifacts\low_risk_high_label_samples.csv', index=False)

    ## TODO: Interpret the results
    baecon_test_df.to_csv(r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\artifacts\lbeacon_events_eval_risk_scores.csv', index=False)

if __name__ == "__main__":
    main()  