import sys
import os
import pandas as pd
from  models import train_supervised_models, train_unsupervised_model
from utils import calculate_fusion_risk_scores

def main():
    # Fit the model to the training data
    df = pd.read_csv(r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\data\beacon_events_train.csv')

    random_forest = train_supervised_models(beacon_df=df)

    df = pd.read_csv(r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\data\beacon_events_train.csv')
    isolation_forest = train_unsupervised_model(beacon_df=df)

    df = pd.read_csv(r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\data\beacon_events_train.csv')
    calculate_fusion_risk_scores(beacon_df_org=df, rf_classifier=random_forest, isolation_forest=isolation_forest)
    print("Risk scores calculated and added to the dataframe.")
    print(df[['event_id', 'fusion_risk_score']].head())

if __name__ == "__main__":
    main()