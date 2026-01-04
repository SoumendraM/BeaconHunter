"""
src.utils
--------
Utility helpers for preprocessing, plotting, and feature preparation used
by the BeaconHunter project.

This module contains convenience functions for:
- plotting categorical distributions (`plot_categorial_distribution`)
- converting categorical columns to `category` dtype
- encoding categorical features as dummies or target-encoded values
- scaling numerical features
- assembling a processed feature `DataFrame` for model training

Public functions
- `plot_categorial_distribution(beacon_df, feature_name, label)`
- `category_cols_to_category_dtype(beacon_df, categorical_cols)`
- `categorical_cols_to_dummies(beacon_df, categorical_cols)`
- `target_encode_categorical_columns(beacon_df, categorical_cols, target_col)`
- `scale_numerical_features(beacon_df, numerical_features)`
- `process_features(beacon_df)`
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.features import create_derived_features

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

## To be used
def target_encode_categorical_columns(beacon_df, categorical_cols: list, target_col: str):
    for col in categorical_cols:
        target_means = beacon_df.groupby(col)[target_col].mean()
        beacon_df[col] = beacon_df[col].map(target_means)
    return beacon_df    

def scale_numerical_features(beacon_df, numerical_features: list):
    scaler = StandardScaler()
    beacon_df[numerical_features] = scaler.fit_transform(beacon_df[numerical_features])
    return beacon_df

def process_features(beacon_df: pd.DataFrame):
    # Fill missing values in inter_event_seconds with median
    beacon_df['inter_event_seconds'] = beacon_df['inter_event_seconds'].fillna(beacon_df['inter_event_seconds'].median())
    # Calculate variance in inter_event_seconds
    #TODO

    # Create derived features
    beacon_df = create_derived_features(beacon_df)
    # Remove columns that are not needed for model training
    beacon_df.drop(columns=['event_id', 'timestamp', 'src_ip', 'dst_ip', 'signed_binary', 'host_id', 'dst_port', 'country_code'], inplace=True)
    # Convert categorical columns to category dtype
    categorical_cols = ['proc_name', 'wierdness', 'proc_risk']
    beacon_df = category_cols_to_category_dtype(beacon_df, categorical_cols)
    # Convert categorical columns to numerical using dummies
    #categorical_cols = ['proc_name', 'protocol', 'country_code', 'user', 'wierdness', 'proc_risk']
    categorical_cols = ['proc_name', 'protocol', 'user', 'wierdness', 'proc_risk']
    beacon_df = categorical_cols_to_dummies(beacon_df, categorical_cols)
    # Apply feature scaling to numerical features
    numerical_features = ['inter_event_seconds', 'beaconness', 'bytes_in', 'bytes_out']
    beacon_df = scale_numerical_features(beacon_df, numerical_features)
    return beacon_df

