## How I approached the assignment

### Research
This was an exellent assignment with a good amount of learning curve. I did a good amount of research on the net to understand various nitty gritty of the C2 Baeconing attack and detection. The following resources were helpful:
 ```bash
https://www.extrahop.com/resources/attacks/c-c-beaconing
https://hunt.io/glossary/c2-beaconing
```

I went through some extensive research on how different parameters and clever tactics for network communication are exploited by the malicious party to exploit network traffic. Also I researched various machine learning models that are best suited for C2 Baeconing detection.

### Github Repository creation

Then I started with creating the Github repository and added the dependencies. I created the main branch and added files there as specified in the assignment sheet.

### Local environment creation
Created the loacl environment for python development. Created a virtual environment and imported all the required libraries. Then altu generated the envirnment.txt file.

### Training Data for Exploratory analysis
Created an `01_features_and_explorations.ipynb` file for EDA. Perfoemed all kinds of analysis on the data to get a general idea about the data distribution and relationships. Once the data understanding has been upto a level, went ahead with the rest of the assignment.

### Development of Code
One EDA was complete, started with writing and testing the code. Factored the code in different file like `util.py`, `features.py`, `models.py`. Wrote and tested end to end the files `evaluate_detector.py`, `train_detector.py` and `score_events.py`.

### Documentation
Based on my experience with this assignment, completed all the documentations `ANALYST_REPORT.md`, `INTEGRITY.md`, `README.md` and `NOTES.md`.

