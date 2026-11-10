# CreditWise Loan Approval Prediction

A machine learning project that explores applicant and loan-related data
to predict whether a loan application will be approved. The project is
implemented in a Jupyter Notebook and compares three classification
algorithms: Logistic Regression, K-Nearest Neighbors (KNN), and Gaussian
Naive Bayes.

## Project Overview

The notebook follows an end-to-end classification workflow:

1.  Load the loan approval dataset.
2.  Inspect the data and handle missing values.
3.  Perform exploratory data analysis (EDA).
4.  Encode categorical features.
5.  Examine feature correlations.
6.  Split the data into training and testing sets and standardize the
    features.
7.  Train and evaluate multiple classification models.
8.  Engineer squared features for `DTI_Ratio` and `Credit_Score`, then
    repeat model training and evaluation.

The notebook includes a note identifying **Gaussian Naive Bayes as the
best model based on precision** in the initial model comparison. The
notebook does not contain saved evaluation output values, so exact
performance metrics are not reported here.

## Dataset

The notebook reads a CSV file named:

``` text
loan_approval_data.csv
```

Place this file in the same directory as the notebook before running it.

The code uses the following fields:

-   `Applicant_ID` --- removed before model training.
-   `Loan_Approved` --- target variable.
-   `Gender`
-   `Education_Level`
-   `Employment_Status`
-   `Marital_Status`
-   `Loan_Purpose`
-   `Property_Area`
-   `Employer_Category`
-   `Applicant_Income`
-   `Coapplicant_Income`
-   `Credit_Score`
-   `DTI_Ratio`
-   `Savings`

The dataset should contain these columns with compatible data types. The
README does not include dataset statistics because they are not
specified in the notebook.

## Methods

### 1. Data cleaning

-   Numerical missing values are imputed using the column mean.
-   Categorical missing values are imputed using the most frequent
    value.

### 2. Exploratory data analysis

The notebook visualizes:

-   The distribution of approved and rejected applications.
-   Gender and education-level counts.
-   Applicant and co-applicant income distributions.
-   Applicant income, credit score, debt-to-income ratio, and savings
    against loan approval.
-   Credit score distributions by approval outcome.

### 3. Feature encoding

-   `Education_Level` and `Loan_Approved` are label encoded.
-   The columns `Employment_Status`, `Marital_Status`, `Loan_Purpose`,
    `Property_Area`, `Gender`, and `Employer_Category` are one-hot
    encoded.
-   The first one-hot encoded category is dropped to reduce redundant
    columns.
-   `Applicant_ID` is removed.

### 4. Correlation analysis

A correlation heatmap is created for numeric features, and features are
sorted by their correlation with `Loan_Approved`.

### 5. Model preparation

-   The target is `Loan_Approved`; all remaining columns are used as
    input features.
-   The data is split into training and test sets with an 80/20 ratio
    and `random_state=42`.
-   `StandardScaler` is fitted on the training features and applied to
    both training and test features.

### 6. Classification models

The notebook trains and evaluates:

-   **Logistic Regression**
-   **K-Nearest Neighbors (KNN)** with `n_neighbors=5`
-   **Gaussian Naive Bayes**

Each model is evaluated using precision, recall, F1-score, accuracy, and
a confusion matrix.

### 7. Feature engineering

The notebook creates two squared features:

-   `DTI_Ratio_sq = DTI_Ratio ** 2`
-   `Credit_Score_sq = Credit_Score ** 2`

It then removes the original `DTI_Ratio` and `Credit_Score` columns,
recreates the train/test split and scaling steps, and trains/evaluates
the same three models again.

## Tech Stack

-   Python
-   pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   scikit-learn
-   Jupyter Notebook

## Setup and Usage

### 1. Clone the repository

``` bash
git clone https://github.com/bavneetchhabra/Creditwise_Loan.git
cd Creditwise_Loan
```

### 2. Create and activate a virtual environment (recommended)

``` bash
python -m venv .venv
```

On macOS/Linux:

``` bash
source .venv/bin/activate
```

On Windows:

``` bash
.venv\Scripts\activate
```

### 3. Install dependencies

``` bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 4. Add the dataset

Put `loan_approval_data.csv` in the project directory, alongside
`CreditWise_Loan.ipynb`.

### 5. Run the notebook

``` bash
jupyter notebook CreditWise_Loan.ipynb
```

Run the notebook cells in order.

## Repository Structure

A suggested GitHub repository layout:

``` text
.
├── CreditWise_Loan.ipynb
├── loan_approval_data.csv
├── README.md
└── requirements.txt
```

Only include the dataset in the repository if its license and sharing
terms allow it. If it cannot be shared, document how to obtain it and
where to place it.

## Notes and Limitations

-   The notebook expects `loan_approval_data.csv` to be available
    locally; the CSV was not included with the uploaded notebook.
-   The notebook contains no stored model evaluation outputs, so model
    scores cannot be verified from the notebook file alone.
-   The notebook's Naive Bayes selection is recorded as a
    precision-based conclusion in a Markdown cell; it should be
    confirmed by rerunning the evaluation cells.
-   The notebook fits imputers and encoders on the full dataset before
    the train/test split. For a production-grade evaluation, fit
    preprocessing steps on the training data only (ideally in a
    scikit-learn `Pipeline`) to avoid data leakage.
-   The notebook does not save a trained model or provide a prediction
    interface/API.

## Future Improvements

-   Use a stratified train/test split if the target classes are
    imbalanced.
-   Put preprocessing and model training into a scikit-learn `Pipeline`.
-   Compare cross-validation results and tune model hyperparameters.
-   Review precision and recall together, since false approvals and
    false rejections may have different costs.
-   Save the selected model and preprocessing pipeline for reuse.
-   Add a reproducible `requirements.txt` and document the dataset
    source.

## License

No license is specified in the notebook. Add a license file to the
repository if you want to define how others may use, modify, and
distribute this project.
