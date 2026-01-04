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
    "BR": 1,
    "CN": 1,
    "HK": 1,
    "IR": 1,
    "KP": 1,
    "NG": 1,
    "RU": 1,
    "TR": 1,
    "UA": 1,
    "VN":  1

where the value 1 means high risk and the rest of the countries have "Low" risk (signified with the value 0) based on training data analysis.


2. Model choice and tradeoff  
    **The models that have been chosen** :  
    a.  Random Forest for Supervised model.  
    Provides great interpretability of relationships since the dataset is complex with high dimension. Also for complex data, Random Forest model provides good accuracy in prediction. The Data lysis meets that requirement compared to other more simpler models like Linear/Logistic Regression, Decision Trees, SVMs, Naive Bayes, KNN, Neural Networks.  
    Tradeoffs:  
    Random Forest Classifier is slow in training anf prediction as training 100s of decision trees takes significant resources are power. It is also space intensive as it takes a lot of memoty to store the individual trees. The vast number of trees created internally makes it difficult to trace the logic behind why it has made a certain decision during prediction.  
    
    b. Isolation Forest for Unsupervised model.  
    Based on the analysis of training data Isolation Forest has been chosen as it satisfies efficiency (fact training), it effectively handles data with high dimensionality. In the dataset there are some irrelavant feations line source and dextination ips. This model is not very sensitive to irrelevant features and outliers. this is a very popular model used in Network Intrusion Detection in cybersecurity.

    **How you tuned the thresholds**  
    An estimate of the optinal threshhold has been taken by calculating the probabilities of thr predictions. Then a harmonic mean has been derieved from the precicion and recall points. The maximum of the fscore has been taken as the threshold which balances the precision and recall.

    **A brief comparison: what would have happened if you used only supervised or only unsupervised.**  
    If a supervised model had only been used, ir sould have excelled at identifying only known, signature based beaconing patterns, as it would have only trained on on datasets which are labelled (benign v/s malefic). Where as an unsupervised model detects what is otherwise known as `zero-day malware` or threats which are unknown by analysing the traffic for anomalies or outliers (that are not marks with known labels).

3. Error Analysis

- Discuss patterns in false positives and false negatives from evaluate_detector.py  
The distribution of the labels in the training dataset is highly skewed with    
0 -> 7764  
1 -> 2236 

| Threshold | False Positive | False Negative |
| ------- | ----- | ------------ |
| 0.2 | 0 | 67 |
| 0.3 | 3 | 36 |
| 0.4 | 19 | 9 |
| 0.5 | 32 | 5 |
| 0.6 | 55 | 2 |
| 0.7 | 121 | 1 |
| 0.8 | 306 | 0 |
| 0.9 | 400 | 0 |

With increasing threshhold we can observe that False Positive increases and False Negative decreases.

Since the events with benign labels are 4 times more than the malicious ones there is bound to be class imbalance in terms of accuracy being more dominant towards the benign labeled ones. In C2 Beaconness scenario, a high FP will mean malicious traffic will remain undetected. 

- Suggest at least two concrete changes (data collection, rules, model) that could reduce those errors in a real deployment  
**Data Collection** : One should update the data collection pipeline to include enriched, stateful session data.  
**Rules** : Static thresholds often cause high false-positive rates when deployed across diverse network environments. Replace static rules with dynamic risk scoring based on behavioral anomalies.  
**Model** : The model should calculate a probabilistic score based on Beaconing behavior. assign a higher penalty to false negatives (missing a real attack) while using the rule-based whitelist to manage false positives (reducing alert fatigue).  


## 2.3 Prioritization of live events

| event_id | host_id | risk_score | anomaly_score | fusion_risk_score | risk_label |
| -------- | ------- | ---------- | ------------- | ----------------- | ---------- |
| EVT-126531315 | HOST-069 | 0.962647 | 0.859394 | 0.911020 | HIGH |
| EVT-349240739 | HOST-010 | 0.973784 | 0.800987 | 0.887386 | HIGH |
| EVT-663092550 | HOST-098 | 0.974792 | 0.792772 | 0.883782 | HIGH |
| EVT-952704438 | HOST-005 | 0.964654 | 0.800241 | 0.882447 | HIGH |
| EVT-391423501 | HOST-064 | 0.968113 | 0.796416 | 0.882264 | HIGH |  

The above hosts are suspecious with high fusion_risk_score and the following may be the reason for that (which triggerred the detector):

1. EVT-126531315 - Country HK (in suspicious list), process name is "unknown.bin" which is a high risk process, at high risk port 83
2. EVT-349240739 - Country BR (in suspicious list), process name is "unknown.bin" which is a high risk process, at high risk port 8080. Moreover this may quualify for 61 seconds inter event beacon to a rare country.
3. EVT-663092550 - Country CN (in suspicious list), process name is "unknown.bin" which is a high risk process, at high risk port 8080
4. EVT-952704438 - Country HK (in suspicious list), process name is "unknown.bin" which is a high risk process, at high risk port 80
5. EVT-391423501 - Country HK (in suspicious list), process name is "unknown.bin" which is a high risk process, at high risk port 8080

**For all the above the recommendation would be to isolate the host or block the destination IP**

## 2.4 Limitations and next steps
The "quality" of C2 detection refers to the signal-to-noise ratio in network traffic. The following factors such as attacckers inserting random variations into the beacon intervals, a large amout of C2 data being hidden in encrypted HTTP sessions. Also legitimate application behavior ofter is made to mimic regular small packet nature of baecons.  
The data coverage here misses detecting intermediate redirectors that mask true IP destinations.  

Potential adversarial behavior, that will evade this detector, will focus on blending into legitimate traffic to avoiid detection such as:  
- Jitter - Introduce random variations in timing
- Protocol abuse - Beaconing frequently uses common, allowed protocols like HTTP/S, DNS, and SSH to bypass firewalls
- In-memory execution - Payloads are executed directly in memory to evade leaving trace in disk.

If I stay in this role for some months, will experiment with different models and their combinations, threshold and training dataset analysing the statistics of the trained models to reduce errors. 