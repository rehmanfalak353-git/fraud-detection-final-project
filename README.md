# Credit Card Fraud Detection using Machine Learning and Neural Networks

**Author:** Falak Rehman | **Intern ID:** ZYNVEX-CERT-2000 | AI/ML Internship Final Project

## Problem

Detect fraudulent credit card transactions. Fraud is extremely rare (about 0.17% of transactions), so accuracy is misleading. The project is judged on precision, recall, F1 and PR-AUC.

## Dataset

[Credit Card Fraud Detection (Kaggle, ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud): about 284,000 European card transactions. Features V1-V28 are PCA-anonymised, plus Time, Amount and the target Class (1 = fraud). The CSV is not included in this repo because of its size; download it from the link above.

## Approach

1. Exploration and cleaning (duplicates removed, class imbalance analysed)
2. RobustScaler on Time and Amount (fitted on training data only), stratified 70/15/15 split
3. Three imbalance-aware models: Logistic Regression, Random Forest, Keras Neural Network (BatchNorm, Dropout, class weights, Early Stopping)
4. Decision threshold tuned on the validation set; final results on an untouched test set
5. Business-impact analysis (fraud value caught vs. false alarms)
6. Saved model, prediction function and Gradio demo

## Files

|File|Purpose|
|-|-|
|`Fraud_Detection_Project_FalakRehman.ipynb`|Full project notebook (run in Google Colab)|
|`fraud_model.keras`|Saved neural network|
|`fraud_scaler.pkl`|Fitted scaler|
|`fraud_config.json`|Feature order and chosen threshold|

## How to run

1. Open the notebook in Google Colab.
2. Run all cells. When asked, upload `creditcard.csv` from Kaggle.
3. The notebook trains the models, saves the files above and launches the demo.

## Results

The Neural Network achieved precision 0.82, recall 0.79 and F1 0.81, catching 56 of 71 frauds in the test set with only 12 false alarms. It catches about 65% of the fraud value while flagging only 0.028% of legitimate transactions.

| Model | Precision | Recall | F1 | PR-AUC | Frauds caught (of 71) | False alarms |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.809 | 0.775 | 0.791 | 0.681 | 55 | 13 |
| Random Forest | 0.770 | 0.803 | 0.786 | 0.789 | 57 | 17 |
| Neural Network | 0.824 | 0.789 | 0.806 | 0.671 | 56 | 12 |

## Tech

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow/Keras, Gradio.

