# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Ensemble method, GradienBoostingClassifier by sklearn. Easy and fast to train.

## Intended Use

Not the best model by far, but one that will be deployed.T Used to predict the salary category of a person.

## Training Data

Data is coming from https://archive.ics.uci.edu/ml/datasets/census+income ; training is done using 80% of this data.

## Evaluation Data

Data is coming from https://archive.ics.uci.edu/ml/datasets/census+income ; evaluation is done using 20% of this data.

## Metrics

Accuracy: 0.829 (0.018)
Recall: 0.354 (0.116)
Precision: 0.889 (0.099)
Fbeta: 0.486 (0.103)

## Ethical Considerations

Bias could be inherit in the dataset, as gender and race are considered. interpret results with caution.

## Caveats and Recommendations

Deploy on Heroku was done using Heroku CLI instead of Github (which would be the preferred way) due to a problemn on Githubs side.