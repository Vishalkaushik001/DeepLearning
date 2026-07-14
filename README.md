# DeepLearning

A simple Artificial Neural Network (ANN) built with TensorFlow/Keras to predict customer churn for a bank, using the classic `Churn_Modelling.csv` dataset.

## Overview

The script (`DeepLearning.py`) trains a binary classifier to predict whether a customer will leave the bank (`Exited` = 1) or stay (`Exited` = 0), based on their demographic and account information.

## Dataset

**`Churn_Modelling.csv`** — 10,000 bank customer records with the following columns:

| Column | Description |
|---|---|
| RowNumber, CustomerId, Surname | Identifiers (not used as model features) |
| CreditScore | Customer's credit score |
| Geography | Customer's country (France, Germany, Spain) |
| Gender | Customer's gender |
| Age | Customer's age |
| Tenure | Years the customer has been with the bank |
| Balance | Account balance |
| NumOfProducts | Number of bank products used |
| HasCrCard | Whether the customer has a credit card (1/0) |
| IsActiveMember | Whether the customer is an active member (1/0) |
| EstimatedSalary | Customer's estimated salary |
| **Exited** | **Target** — whether the customer churned (1) or not (0) |

## Model Architecture

A fully connected (Dense) feed-forward network:

- **Input layer + 1st hidden layer**: 6 units, ReLU activation, He-uniform initializer
- **2nd hidden layer**: 6 units, ReLU activation
- **3rd hidden layer**: 6 units, ReLU activation
- **Output layer**: 1 unit, Sigmoid activation (binary classification)

Compiled with the Adam optimizer (learning rate 0.001) and binary cross-entropy loss.

## Preprocessing

1. Drops non-predictive columns (RowNumber, CustomerId, Surname)
2. One-hot encodes `Geography` and `Gender` (with `drop_first=True` to avoid the dummy variable trap)
3. Splits data into 80% training / 20% test sets
4. Scales features using `StandardScaler`

## Training

- Batch size: 32
- Epochs: 100 (with **EarlyStopping** on validation loss, patience = 10, restoring best weights)
- Validation split: 20% of training data

## Evaluation

After training, the model is evaluated on the held-out test set using a confusion matrix and accuracy score. A plot of training vs. validation accuracy over epochs is also displayed.

## Requirements

```
tensorflow
numpy
pandas
matplotlib
scikit-learn
```

Install with:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/Vishalkaushik001/DeepLearning.git
   cd DeepLearning
   ```
2. Make sure `Churn_Modelling.csv` is in the same folder as `DeepLearning.py` (it is, by default).
3. Run the script:
   ```bash
   python DeepLearning.py
   ```

## Output

The script prints:
- TensorFlow version
- Training history keys
- Confusion matrix and test accuracy

...and displays a plot comparing training and validation accuracy across epochs.

## License

No license specified.
