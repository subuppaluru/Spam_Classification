# SMS Spam Detection Project

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Steps](#project-steps)
- [Results](#results)
- [Potential Improvements](#potential-improvements)
- [Usage](#usage)
- [Requirements](#requirements)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview
This project aims to build a machine learning model to classify SMS messages as either "ham" (legitimate) or "spam". The notebook covers the entire process from data loading and exploratory data analysis, through text preprocessing and feature extraction, to model training, evaluation, and analysis.

## Dataset
The dataset used in this project is the "SMS Spam Collection v1" dataset, loaded from the `spam.csv` file. It contains SMS messages labeled as either 'ham' or 'spam'.

## Project Steps
1. **Data Loading and Description**: Load dataset, inspect structure, data types, and basic statistics.
2. **Exploratory Data Analysis (EDA)**:
   - Analyze class distribution.
   - Investigate message lengths and distribution.
   - Identify most common words in 'ham' and 'spam' messages.
3. **Text Preprocessing**: Clean and transform messages (lowercasing, noise removal, tokenization, stop word removal, lemmatization).
4. **Feature Extraction**: Convert text data into numerical features using Bag-of-Words (TF-IDF, CountVectorizer).
5. **Model Training**:
   - Split data into training and testing sets.
   - Train Multinomial Naive Bayes and Logistic Regression classifiers.
6. **Model Evaluation**: Evaluate models using classification reports (precision, recall, F1-score).
7. **Model Comparison**: Compare performance metrics to determine best model.
8. **Error Analysis**: Examine misclassified messages to understand errors.
9. **Key Feature Analysis**: Identify most important words for spam/ham classification.
10. **Conclusion and Improvements**: Summarize findings and suggest further improvements.

## Results
Both the Multinomial Naive Bayes and Logistic Regression models achieved high accuracy in classifying SMS messages. Logistic Regression showed slightly better overall performance.

## Potential Improvements
- Address class imbalance using oversampling/undersampling.
- Explore additional feature engineering (N-grams, message length).
- Evaluate advanced classification algorithms.
- Hyperparameter tuning.
- Train on larger/more diverse datasets.
- Implement user feedback mechanisms.

## Usage
To reproduce the analysis:
1. Clone the repository to your local machine.
2. Ensure you have Python and necessary libraries installed:
   ```bash
   pip install pandas numpy scikit-learn nltk seaborn matplotlib
   ```
3. Run the Jupyter notebook or script as described below.

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- nltk
- seaborn
- matplotlib

## Installation
```bash
git clone https://github.com/subuppaluru/Spam_Classification.git
cd Spam_Classification
pip install -r requirements.txt
```
*(If `requirements.txt` is missing, install the libraries as listed above.)*

## How to Run
Open the notebook/script in Jupyter Notebook or run:
```bash
python <script_name>.py
```
Follow the steps and cells in order.

## Evaluation
The evaluation metrics and confusion matrix are displayed in the notebook output.

## Contributing
Contributions are welcome! Please open issues or submit pull requests for improvements.

## License
Include license information here.

## Acknowledgements
- [SMS Spam Collection Dataset](https://www.dt.fee.unicamp.br/~tiago/smsspamcollection/)
- Open source libraries: pandas, numpy, scikit-learn, nltk, seaborn, matplotlib
