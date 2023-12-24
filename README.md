# Multilayer-perceptron
Implement a multilayer perceptron from scratch for breast cancer recognition.

## Data analysis

Create graph to have a better understanding of the data and filter them into "filter_data.csv".

The first column is the waited output, the features selected are :
- M Radius, M Texture, M Compactness, M Concavity, M Concave Points
- SE Radius, SE Perimeter, SE Area, SE Compactness, SE Concavity, SE Concave Points
- W Radius, W Texture, W Smoothness, W Compactness, W Concavity, W Concave Points

For more information, look DataAnalysis.md.

For the Breast Cancer Detection the command is :
```bash
python3 analysis.py data/Diagnostic_Breast_Cancer/data.csv
```

## Dataset preparation

Preprocess and separate the dataset into a training dataset (used to train the MLP)
and a validation dataset (used to validate the training).


More information about the program with
```bash
python3 preparation.py -h
```

For the Breast Cancer Detection the command is :
```bash
python3 preparation.py data/Diagnostic_Breast_Cancer/filter_data.csv --cancer_diagnostic
```

## Training of the MLP
