import argparse
from ntpath import join
import joblib
import pandas as pd

from features import create_derived_features
from utils import calculate_fusion_risk_scores, categorical_cols_to_dummies, category_cols_to_category_dtype, scale_numerical_features

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


    ################
    baecon_eval_df = pd.read_csv(arg_input_file)

    df_processed = baecon_eval_df.copy()

    df_processed['inter_event_seconds'] = df_processed['inter_event_seconds'].fillna(df_processed['inter_event_seconds'].median())
    #from features import create_derived_features, plot_categorial_distribution

    df_processed = create_derived_features(df_processed)

    # Remove columns that are not needed for model training
    df_processed.drop(columns=['event_id', 'timestamp', 'src_ip', 'dst_ip', 'signed_binary', 'host_id', 'dst_port'], inplace=True)

    # Convert categorical columns to category dtype
    categorical_cols = ['proc_name', 'wierdness', 'proc_risk']
    df_processed = category_cols_to_category_dtype(df_processed, categorical_cols)

    # Convert categorical columns to numerical using dummies
    categorical_cols = ['proc_name', 'protocol','country_code', 'user', 'wierdness', 'proc_risk']
    df_processed = categorical_cols_to_dummies(df_processed, categorical_cols)

    # Apply feature scaling to numerical features
    
    numerical_features = ['inter_event_seconds', 'beaconness', 'bytes_in', 'bytes_out']
    df_processed = scale_numerical_features(df_processed, numerical_features)

    # Load  the trained Random Forest model
    rf_classifier = joblib.load(r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\artifacts\rf_classifier_model.joblib')
    isolation_forest = joblib.load(r'C:\Users\Soumendra\Documents\GitHub\BeaconHunter\artifacts\isolation_forest_model.joblib') 
    calculate_fusion_risk_scores(beacon_df_org=baecon_eval_df)
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