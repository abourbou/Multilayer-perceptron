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

Create a Multi Layer Perceptron network and train it using the training dataset and validation dataset.
The neural network parameters are loaded from a JSON.

```bash
python3 training.py data/Diagnostic_Breast_Cancer/training_data.csv data/Diagnostic_Breast_Cancer/validation_data.csv params/breast_cancer_params.json
```

The weights of the network can be saved afterward.

## Validation of the MLP

Load weights and a dataset to validate model by computing loss and accuracy.
```bash
python3 prediction.py weights_file.npz data/Diagnostic_Breast_Cancer/training_data.csv
```
