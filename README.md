# Hate Speech Detection

## Usage
This code provides a framework for hate speech classification using different models and feature extraction techniques. To run the code, follow these steps:

1. Install the required libraries by running `pip install -r requirements.txt`.
2. Make sure you have the necessary data files in the specified path (`./data/` by default). The data files should be in CSV format with columns named 'text' and 'label'.
3. Run the code using the command: `python main.py --model <model> --feature <feature> --path <data_path> --lr <learning_rate> --epochs <num_epochs> --lam <regularization_param>`.
   - `<model>`: Choose the model to use for classification. Options are 'AlwaysPredictZero', 'NaiveBayes', 'LogisticRegression', 'L2Reg', and 'BonusClassifier'.
   - `<feature>`: Choose the feature extraction technique. Options are 'unigram', 'bigram', or 'customized'.
   - `<data_path>`: Specify the path to the data files. Default is `./data/`.
   - `<learning_rate>`: Learning rate for the logistic regression models. Default is 0.1.
   - `<num_epochs>`: Number of epochs for the logistic regression models. Default is 5000.
   - `<regularization_param>`: Regularization parameter for L2-regularized logistic regression. Default is 0.01.

## Models
The code provides several models for hate speech classification:

- `AlwaysPredictZero`: This model always predicts the label 0 (non-hate speech) for any input.
- `NaiveBayes`: Implements a Naive Bayes classifier for hate speech classification.
- `LogisticRegression`: Implements a logistic regression classifier for hate speech classification.
- `L2Reg`: Implements logistic regression with L2 regularization for hate speech classification.

## Feature Extraction
The code supports three feature extraction techniques:

- `unigram`: Extracts unigram features from the text.
- `bigram`: Extracts bigram features from the text.
- `customized`: Allows customization of the feature extraction process. You can modify the code in the `CustomFeature` class to define your own feature extraction logic.

## References
The code is based on the following resources:

- [Natural Language Toolkit (NLTK)](https://www.nltk.org/): A library for natural language processing in Python.
- [NumPy](https://numpy.org/): A library for efficient numerical operations in Python.

Please refer to these resources for more information on the underlying concepts and functions used in the code.