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

The model was evaluated using with an accuracy score of 0.829 (0.012).

## Ethical Considerations

Bias could be inherit in the dataset, as gender and race are considered. interpret results with caution.

## Caveats and Recommendations

None