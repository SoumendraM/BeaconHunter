import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from features import create_derived_features
import joblib

def plot_categorial_distribution(beacon_df, feature_name: str, label: str):
    import matplotlib.pyplot as plt
    import seaborn as sns

    print(f'feature name: {feature_name}')

    aggr = beacon_df.groupby([feature_name, 'label'], as_index=False).agg(
        Count=('label', 'count')
    )
    aggr.pivot(index=feature_name, columns='label', values='Count').plot(kind='bar', stacked=True, figsize=(12,8))
    plt.title(f'Count of Records by \'{feature_name}\' and Label')
    plt.xlabel(feature_name)
    plt.ylabel('Count')
    plt.legend(title='label')
    plt.show()

def category_cols_to_category_dtype(beacon_df, categorical_cols: list):
    for col in categorical_cols:
        beacon_df[col] = beacon_df[col].astype('category')
    return beacon_df

def categorical_cols_to_dummies(beacon_df, categorical_cols: list):
    beacon_df = pd.get_dummies(beacon_df, columns=categorical_cols, drop_first=True)
    return beacon_df

def scale_numerical_features(beacon_df, numerical_features: list):
    scaler = StandardScaler()
    beacon_df[numerical_features] = scaler.fit_transform(beacon_df[numerical_features])
    return beacon_df

def calculate_fusion_risk_scores(beacon_df_org: pd.DataFrame):
    # Load the trained Random Forest model
    rf_classifier = joblib.load(r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\artifacts\rf_classifier_model.joblib')

    beacon_df = beacon_df_org.copy()
    beacon_df = create_derived_features(beacon_df)
    # Preprocess the beacon_df similar to training data
    beacon_df.drop(columns=['event_id', 'timestamp', 'src_ip', 'dst_ip', 'signed_binary', 'host_id', 'dst_port'], inplace=True)
    
    # remove label column if exists
    if 'label' in beacon_df.columns:
        beacon_df = beacon_df.drop(columns=['label'])
    categorical_cols = ['proc_name', 'wierdness', 'proc_risk']
    beacon_df = category_cols_to_category_dtype(beacon_df, categorical_cols)
    categorical_cols = ['proc_name', 'protocol', 'country_code', 'user', 'wierdness', 'proc_risk']
    beacon_df = categorical_cols_to_dummies(beacon_df, categorical_cols)
    numerical_features = ['inter_event_seconds', 'beaconness', 'bytes_in', 'bytes_out']
    beacon_df = scale_numerical_features(beacon_df, numerical_features)
    # Predict probabilities using the loaded model
    y_prob = rf_classifier.predict_proba(beacon_df)[:, 1]
    risk_scores = MinMaxScaler().fit_transform(y_prob.reshape(-1, 1))
    prediction = rf_classifier.predict(beacon_df)

    # Load trained Isolation Forest model
    isolation_forest = joblib.load(r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\artifacts\isolation_forest_model.joblib')
    # Calculate anomaly scores
    anomaly_scores = isolation_forest.decision_function(beacon_df)
    beacon_df['anomaly_score'] = anomaly_scores
    scaler = MinMaxScaler()
    beacon_df['anomaly_score_scaled'] = scaler.fit_transform(beacon_df[['anomaly_score']])
    beacon_df['risk_score'] = risk_scores
    #prediction = rf_classifier.predict(beacon_df)

    # Combine risk score and anomaly score to get final risk score using weighted average
    # The choice of weightes can be adjusted based on importance of each score
    final_risk_score = 0.6 * beacon_df['risk_score'] + 0.4 * beacon_df['anomaly_score_scaled']
    beacon_df_org['fusion_risk_score'] = final_risk_score
    beacon_df_org['anomaly_score'] = anomaly_scores
    beacon_df_org['risk_score'] = risk_scores
    beacon_df_org['prediction'] = prediction