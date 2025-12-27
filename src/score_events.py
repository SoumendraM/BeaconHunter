import argparse
from ntpath import join
import joblib
import pandas as pd

from utils import calculate_fusion_risk_scores, process_features

def main():

    parser = argparse.ArgumentParser(description="Score events using trained models.")
    parser.add_argument("--input", type=str, required=True, help="Path to the unlabelled input CSV file.")
    parser.add_argument("--output", type=str, required=True, help="Path to the evaluation score output CSV file.")

    args = parser.parse_args()
    arg_input_file = args.input
    arg_output_file = args.output
    arg_input_file = join(r'./', arg_input_file)
    arg_output_file = join(r'./', arg_output_file)
    beacon_df = pd.read_csv(arg_input_file)

    baecon_eval_df = pd.read_csv(arg_input_file)

    df_processed = baecon_eval_df.copy()
    df_processed = process_features(df_processed)
    # Load  the trained Random Forest model
    rf_classifier = joblib.load(r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\artifacts\rf_classifier_model.joblib')
    isolation_forest = joblib.load(r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\artifacts\isolation_forest_model.joblib') 
    calculate_fusion_risk_scores(beacon_df_org=baecon_eval_df, rf_classifier=rf_classifier, isolation_forest=isolation_forest)
    baecon_eval_df['risk_label'] = baecon_eval_df['fusion_risk_score'].apply(lambda x: 'HIGH' if x >= 0.8 
                                                                             else ('MEDIUM' if x < 0.8 and x >= 0.5 
                                                                                   else 'LOW'))
    #print(baecon_eval_df[['event_id', 'fusion_risk_score']])
    print("Fusion risk scores calculated and added to the evaluation dataframe.")
    # Save 'event_id', 'host_id', 'fusion_risk_score', 'risk_label' to a CSV file
    beacon_df_eval_df = baecon_eval_df[['event_id', 'host_id', 'fusion_risk_score', 'risk_label']]
    beacon_df_eval_df.to_csv(arg_output_file, index=False)

    # Count of high risk events
    high_risk_count = baecon_eval_df[baecon_eval_df['risk_label'] == 'HIGH'].shape[0]
    print(f'Number of HIGH risk events: {high_risk_count}')


if __name__ == "__main__":
    main()  