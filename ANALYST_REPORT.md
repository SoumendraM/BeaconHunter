# Analyst Facing Security Report

## 2.1 Executive Summary

- What I did  
As goal of this exercise, the primary objective was to create a risk detector, that will train underlying model with the training data provided and output risk score with various other associated artifacts which help understand the risk model along with other error analysis.  

    **Data Exploration and Feature Engineering**  
    1. Read `\data\\beacon_events_train.csv` into dataframe for analysis.
    2. Checked if any of the column data is missing.
    3. Applied imputation with median value for the feature `inter_event_seconds`.
    4. Created histogram for key columns `inter_event_seconds`, `dst_port`, `bytes_in`, `bytes_out`, to check the corresponding data distributions.
    5. Plotted FacetGrid histogram to compare `inter_event_seconds` against `label`.
    6. For checking the outliers, plotted the `boxplot` for `inter_event_seconds`, `bytes_in`, `bytes_out`.
    7. For important categorical columns `country_code`, `proc_name`, `proptocol`, `used`, `dst_port` plotted distribution plot to compare between labels 0 and 1.
    8. Created the following derived features:
        - `baeconness`: variance of `inter_event_seconds` against `host_id` (Dues to lack of more data, could not include dst_port), as the variance could not be calculated wor host_id and dst_port group for single columns.
        - `weirdness`: From the data distribution plot for `dst_port`, the ports 80, 443, 53, 8080, 8443, 993, 995 were isolated as the port which were distnguishably used more often in the events. Thus this feature has been used to differentiate the more `common` ports from the `rare` ports.
        - `proc_risk`: The following processes were identified as high risk according to the distribution of process names against `label` 1 - 'cmd.exe', 'cscript.exe', 'meterpreter.exe', 'mshta.exe', 'powershell.exe',                     'regsvr32.exe', 'rundll32.exe', 'sliver-client.exe', 'unknown.bin', 'wscript.exe'. They have been categorised as `high` and the rest as low.

    **TRAINING the models**:
    1. Create src/train_detector.py file which is used to train the models (supervised and unsupervised).
    2. Read the training data file data/beacon_events_train.csv into a Dataframe.
    3. Created subdirectories as suggested in section 3.1 Repository Structure.
    4. Created supoorting function in the 'util' directory.
    5. Created model creation functions in the 'model' directory.
    6. Created feature processing functions in the 'features' repository.
    7. For preprocessing, analysed the training dataset for any missing values and applied imputation on 'inter_event_seconds as it was the only dataset with missing data.
    8. Altered the 'object' columns to 'category' data type column.
    9. Applied one hot encoding on the categorical column.
    10. Created derieved features - beaconness, wierdness, proc_risk and geoip
    10. Which columns were dropped and why? TODO
    11. Applied scaling and normalization to numerical columns.
    12. Saved the intermediate processed dataset to the artifacts folder for reference.
    13. Applied the processed dataset to train Random Forest Classifier for Supervised model.
    14. Generated the following statistics  
        - precision
        - recall
        - F1 score
        - ROC-AUC
        - PR-AUC
    15. Applied the processed dataset to train Isolation Forest for Unsupervised model to detect anomaly.
    16. Defined a `threshhold` of risk_score > 0.7 to mark the event as malicious. 
    15. Saved the trained models in 'artifacts' folder.

    **Evaluating the models**:
    1. Read labelled test data `data/beacon_events_test_labeled.csv` into a dataset
    2. perform the same data cleaning steps as has been done during teh training phase for the test data.
    3. Load the saved and trainied classifier `Random Forest`.
    4. Use the model for prediction on the test data.
    5. Computed the `predictions` and the `risk score`.
    6. Printed the Summary statistics - Confusion matrix, ROC-AUC, precision recall.
    7. Saved the top 10 most misclassified events (high risk but actually benign; low risk but actually malicious) into csv for analysis.

    **Scoring interface CLI**:
    1. Created a CLI that runs like:
    `python -m src.score_events --input data/beacon_events_eval_unlabeled.csv --output results/eval_scored.csv`
    2. For each of the events the followings are reported
        - `event_id`
        - `host_id`
        - `risk_score`
        - `risk_label` HIGH/MED/LOW, based on the following threshholds  
            'HIGH' if risk_score >= 0.8   
            'MEDIUM' if risk_score < 0.8 and x >= 0.5  
            'LOW' if risk_score < 0.8

- What the detector can and cannot do

- Key results on the test data (successes and limitations)


## 2.2 Methodology and Analysis

1. Feature Engineering
The features created are 
-   `baeconness` - The variance of inter_event_seconds per host_id. In standard baeconing the inter event time intervals are used as an advanced technique to evade detection. The communication pattern is disguised in the form of suddenly occuring high volume burst of transactions instead of low frequency data exchange. The `inter_event_seconds` variance per host will capture this behavior during model training in the real word scenario.

-  `wierdness` - In the real world scenario, a command and control servers (C2) destinatiion port recieves sudden bursts of data packets more frequenly from the infected host machine. This often occurs in 80/TCP/HTTP 443/HTTPS ports and is disguised to blend with normal web traffic. This feature helps to take into account the frequently visited destination ports.

-   `pro_risk` - Attackers commonly exploit vulnerable processes like powershell.exe, cmd.exe, rundll32.exe, or regsvr32.exe to execute baecons. They manipulate service hosts like svchost.exe. The processes 'cmd.exe', 'cscript.exe', 'meterpreter.exe', 'mshta.exe', 'powershell.exe', 'regsvr32.exe', 'rundll32.exe', 'sliver-client.exe', 'unknown.bin', 'wscript.exe' have been identified from the distribution plot during feature analysis as probable targets mimicing real world exploitation techniques.

Data Analysis and some observations:  
During Exploratory data analysis, the followings have been observed 
a.  Destination ports 80, 443, 53, 8080, 8443, 993, 995 have the maximum events tied to.  
b.  The host processes cmd.exe', 'cscript.exe', 'meterpreter.exe', 'mshta.exe', 'powershell.exe', regsvr32.exe', 'rundll32.exe', 'sliver-client.exe', 'unknown.bin', 'wscript.exe' have maximum label 1 ro label 0 reation making them highly suspecious.  
c.  The countries  
    "US": "High",  
    "CA": "High",  
    "GB": "High",  
    "DE": "High",  
    "FR": "High",  
    "JP": "Medium",  
    "IN": "Medium",  
    "BR": "Medium",  
    "AU": "Medium"  

and the rest of the countries have "Low" risk based on training data analysis.


2. Model choice and tradeoff  
    a.  Random Forest for Supervised model.  
    Provides great interpretability of relationships since the dataset is complex with high dimension. Also for complex data, Random Forest model provides good accuracy in prediction. The Data lysis meets that requirement compared to other more simpler models like Linear/Logistic Regression, Decision Trees, SVMs, Naive Bayes, KNN, Neural Networks.  
    Tradeoffs:  
    Random Forest Classifier is slow in training anf prediction as training 100s of decision trees takes significant resources are power. It is also space intensive as it takes a lot of memoty to store the individual trees. The vast number of trees created internally makes it difficult to trace the logic behind why it has made a certain decision during prediction.  
    
    b. Isolation Forest for Unsupervised model.  
    Based on the analysis of training data Isolation Forest has been chosen as it satisfies efficiency (fact training), it effectively handles data with high dimensionality. In the dataset there are some irrelavant feations line source and dextination ips. This model is not very sensitive to irrelevant features and outliers. this is a very popular model used in Network Intrusion Detection in cybersecurity.

3. Error Analysis

- Why you chose your supervised and unsupervised algorithms.
- How you tuned/selected thresholds.
- A brief comparison: what would have happened if you used only supervised or only unsupervised.

Fraudsters will often change their strategies and attempt new ways of committing fraud. Many of these strategies will be completely unexpected. This is why unsupervised methods are often used to detect anomalies. They work by comparing all transactions and identifying ones that have unusual feature values. Importantly, this means we do not have to label any transactions beforehand. (Modify it to explain)