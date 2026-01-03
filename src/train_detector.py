import sys
import os
import pandas as pd
from  src.models import train_supervised_models, train_unsupervised_model, calculate_fusion_risk_scores
#from utils import calculate_fusion_risk_scores

# Add src to path
#sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    # Fit the model to the training data
    df = pd.read_csv(r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\data\beacon_events_train.csv')

    df_processed = df.copy()
    random_forest, df_processed['risk_score'] = train_supervised_models(beacon_df=df)

    df = pd.read_csv(r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\data\beacon_events_train.csv')
    isolation_forest, df_processed['anomaly_prediction'], df_processed['anomaly_score']  = train_unsupervised_model(beacon_df=df)
    df_processed['fusion_score'] = calculate_fusion_risk_scores(beacon_df_processed=df_processed)

if __name__ == "__main__":
    main()