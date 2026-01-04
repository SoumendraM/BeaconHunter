# BeaconHunter

BeaconHunter is a tool designed to detect and analyze wireless/wired beacons in network environments. It provides security researchers and network administrators with insights into beacon activity, helping identify potential security threats and unauthorized devices.

## Features

- Data file based beacon detection and monitoring
- Beacon signal strength analysis
- Event identification and tracking based on event ID.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Data file available for anaysis using the score_events CI.

### Installation

```bash
git clone https://github.com/SoumendraM/BeaconHunter.git
cd BeaconHunter
pip install -r requirements.txt
```

## Usage of the code developed and how to execute 

```bash
python -m  src.train_detector.py
python -n src.evaluate_detector.py
python -m src.score_events --input data/beacon_events_eval_unlabeled.csv --output results/eval_scored.csv
```

## To run test module
```bash
python -m pytest ./tests
```

## Basic docker usage
```bash
docker build -t beaconhunter .
docker run --rm -v $(pwd)/data:/app/data beaconhunter \
python -m src.score_events --input data/beacon_events_eval_unlabeled.csv --output /app/results/eval_scored.csv
```

## Files produced as intermediate data persistence for Supervised and Unsupervised
```bash
artifacts/beacon_events_if_results.csv  
artifacts/beacon_events_test_with_risk_scores.csv  
artifacts/beacon_events_train_if_processed.csv  
artifacts/beacon_events_train_rf_processed.csv  
artifacts/beacon_events_train_with_risk_scores.csv  
artifacts/high_risk_low_label_samples.csv  
artifacts/beacon_events_eval_risk_scores.csv  
artifacts/low_risk_high_label_samples.csv  
```
