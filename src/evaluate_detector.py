from utils import category_cols_to_category_dtype, categorical_cols_to_dummies, process_features, scale_numerical_features, calculate_fusion_risk_scores
from features import create_derived_features
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


def main():
    baecon_test_df = pd.read_csv(r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\data\beacon_events_test_labeled.csv')

    df_processed = baecon_test_df.copy()
    df_processed = process_features(df_processed)
    df_processed_X = df_processed.drop(columns=['label'])
    df_processed_y = baecon_test_df['label']

    # Load  the trained Random Forest model
    rf_classifier = joblib.load(r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\artifacts\rf_classifier_model.joblib')

    df_processed_y_pred = rf_classifier.predict(df_processed_X) 

    # report classification report  
    
    print(classification_report(df_processed_y, df_processed_y_pred))
    # report confusion matrix

    print(confusion_matrix(df_processed_y, df_processed_y_pred))

    # ROC AUC score
    from sklearn.metrics import roc_auc_score
    roc_auc = roc_auc_score(df_processed_y, rf_classifier.predict_proba(df_processed_X)[:, 1])
    print(f'ROC AUC Score: {roc_auc}')

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

if __name__ == "__main__":
    main()  