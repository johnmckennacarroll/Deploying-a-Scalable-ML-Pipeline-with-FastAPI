# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This project implements a machine learning pipeline to predict whether an individual's annual income exceeds
$50K using the U.S. Census Income dataset. The model was trained using a Random Forest Classifier from scikit-learn.

The pipeline includes data preprocessing, feature encoding using a OneHotEncoder, model training, evaluation, and 
deployment via a FastAPI REST API. The trained model and encoder and serialized using pickle and used for inference
in the deplotyed API.

This project also includes:
- automated testing with pytest
- code quality checks using flake8
- continuous integration using GitHub Actions
- model slice performance evaluation across categorical features

## Intended Use
The model is intended for educational purposes and demonstrates how to build, test, and deploy a machine learning 
model in a production-style pipeline.

Potential use cases can include:
- demonstrating ML pipeline development
- learning MLOps practices such as CI/CD, testing, and API deployment
- practicing model evaluations across demographic slices

This model should not be used for real-world decision making, particularly in financial or employement contexts

## Training Data
The model was trained using the US Census Income Dataset.

The dataset contains demographic and employment related features such as:
- age
- workclass
- education
- occupation
- marital status
- race
- sex
- house worked per week
- capital gain/loss
- native country

The target variable predicts whether income is:
- <=50K
- >50K

The data set contains approximately 32,561 rows, and 15 features

Categorical features were encoded using one-hot encoding before training.
## Evaluation Data
The dataset was split into training and testing subsets using an 80/20 train-test split.

The model was evaluated on the held-out test set to measure generalization performance.

In addition to overall evaluation, performance metrics were computed across cetegorical feature slices 
to identify potential performance disparities between groups.
## Metrics
The model was evaluated using the following metrics:

- precision
- recall
- F1 score

Model performance on the test datasetL

METRIC                  SCORE
precision               0.7353
recall                  0.6378
F1 score                0.6831

Slice-based evaluation was also performed across categorical features such as workclass, education,
marital status, occupation, relationship, race, sex, and native country.

Results for these slices were saved to slice_output.txt. 

## Ethical Considerations
The Census dataset contains sensitive demographic attributes including race, sex, and nationality.
Models trained on such data may inadvertantly learn patterns reflecting societal bias.

Potential eithical risks include:
-reinforcing historical biases in income prediction
-producing unequal performance across demographic groups
-misuse of predictions in hiring or financial decisions

Slice-based model evaluation was implemented in this project to help identify potential disparities
across demograpgic groups. 

## Caveats and Recommendations
Several limitations exist:

- the dataset is historical and may not reflect current economic conditions
- the model has not been tuned extensively for optimal performance
- bias and fairness mitigation techniques were not implemented
- the dataset contains missing or ambiguous categorical values

Future improvements could include:
- hyperparameter tuning
- fairness evaluation and bias mitigation
- additional model explainability techniques
- more robust validation strategies such as cross-validation