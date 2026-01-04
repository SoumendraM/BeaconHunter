# Chronological development log — BeaconHunter

1. Project inception (idea & research)
   - Defined problem domain and goals: locate/identify beacons and related telemetry.
   - Researched protocols, libraries constraints.
Date: 12/25/2025

2. Prototype & MVP
   - Built a minimal scanner to detect beacons and log identifiers.
   - Chose core tech stack and platform targets (CLI + optional GUI).
Date: 12/26/2025

3. Core feature implementation
   - Added parsing of beacon payloads, basic filtering, and persist/log storage.
   - Implemented configuration file and simple command-line options.
Date: 12/26/2025

4. Testing & CI
   - Introduced unit tests for parsing and filtering logic. (12/28/2025)
   - Added CI configuration (lint, tests) for automated validation. (12/28/2025)

5. Performance & refactor
   - Refactored modules for clearer separation (data handling, parser, cli).
Date: 12/28/2025
6. Documentation & examples
   - Created README, usage examples, and development notes.
   - Added sample datasets and troubleshooting tips.
Date:12/28/2025

# What I tried
1. Did EDA on the training dataset to get deeper unrstanding of the data distribution.  
   a. Histogram and histplot on inter_event_seconds to get the distribution versus label.  
   b. Histplot on `inter_event_seconds` versus protocol to get the frequency.  
   c. Boxplot to understand `dst_port`, `bytes_in`, `bytes_out` versus `label`.  
   d. Get counts for individual category columns, `country_code`, `proc_name`.  
   e. Plotted `proc_name`, `dst_port`, `protocol` distribution against label to undestand the data distributions.  
2. Created derived features `baconness`, `weirdness`, `proc_risk`, `geoip_risk`.
3. Trained the models `Random Forest` for supervised (categorical) and `Isolation Forest` on unsupervised (anomaly) on training data.
4. Calculated risk score based on threshold. The threshold has been chosen based on optimization where the precision has been balance against recall.
5. Calculated fusion risk score by choosing 50% weight on the risk score (sypervised) and anomaly score (unsupervised).
6. Created helper modules models.py for model specific, featiures.py for feature specific and utils.py for auxilliary classes, methods are data types.
7. Created train_detector.py for training the models.
8. Created evaluate.py for evalusating The trained models as per the instructions with summary and most misclassified events listed.
9. Created CLi score_events.py as per the instruction.
10. Created ANALYSI_REPORT.md, NOTES.md, INTEGRITY.pd, README.md
11. Created test files test_features.py, test_train_detectorr.py, test_score_events.py sanity testing using pytest testing framework.
12. Created ci.yml for python setup, library installation, linting and testing during push and pull request.
13. Created a minimal Dockerfile.

# Dead ends and bug
1. During github push pylint reports some errors have been left unaddressed.
2. DockerFile is untested.
3. Could not create derived feature `beaconness` by grouping `host_id` and `dst_port` adn calculatin the variance on `inter_events_seconds` due to lack of data, in the training dataset.

# What would you do next if you had more time
1. Had there been larger dataset;, whould have experimented with Histogram based Gradient Boosting for classification, as it helps in fnding complex patterns.
2. Analysed more on consistency of time deltas and packet sizes.
3. Analysed beacons that tried using common protocols on unusual ports.
4. Develop a "normal" traffic baseline for specific users, roles, and organizational units to identify true anomalies.
5. Tune thresholds by adjusting parameters for maximum jitter and beacon count variance to balance sensitivity and precision. 

