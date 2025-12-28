from utils import category_cols_to_category_dtype, categorical_cols_to_dummies, process_features, scale_numerical_features, calculate_fusion_risk_scores
from features import create_derived_features
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from models import RandomForestModel


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
    
    rf_classifier.evaluate_threshold_test(df_processed_X, df_processed_y, 0.391)

    # Load trained Isolation Forest model
    isolation_forest = joblib.load(r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\artifacts\isolation_forest_model.joblib') 
    calculate_fusion_risk_scores(beacon_df_org=baecon_test_df, rf_classifier=rf_classifier, isolation_forest=isolation_forest)

    high_risk_low_label = baecon_test_df[(baecon_test_df['fusion_risk_score'] > 0.55) & (baecon_test_df['label'] == 0)]
    print("Events with High risk scores with low labels saved to artifacts\\high_risk_low_label_samples.csv") 
    # Save to csv
    high_risk_low_label.head(5).to_csv(r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\artifacts\high_risk_low_label_samples.csv', index=False)   



    low_risk_high_label = baecon_test_df[(baecon_test_df['fusion_risk_score'] < 0.55) & (baecon_test_df['label'] == 1)]
    print("Events with Low risk scores with high labels saved to artifacts\\low_risk_high_label_samples.csv")
    # Save to csv
    low_risk_high_label.head(5).to_csv(r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\artifacts\low_risk_high_label_samples.csv', index=False)

    ## TODO: Interpret the results
    baecon_test_df.to_csv(r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\artifacts\lbeacon_events_eval_risk_scores.csv', index=False)

if __name__ == "__main__":
    main()  