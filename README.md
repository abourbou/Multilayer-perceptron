# Multilayer-perceptron
Implement a multilayer perceptron from scratch for breast cancer recognition.

## Data analysis

Create graph to have a better understanding of the data and filter them into "filter_data.csv".
```bash
python3 analysis.py data.csv [filter_only]
```

The first column is the waited output, the features selected are :
- M Radius, M Texture, M Compactness, M Concavity, M Concave Points
- SE Radius, SE Perimeter, SE Area, SE Compactness, SE Concavity, SE Concave Points
- W Radius, W Texture, W Smoothness, W Compactness, W Concavity, W Concave Points

For more information, look DataAnalysis.md.
