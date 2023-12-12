import argparse
import json
from multi_layer_perceptron import Hyperparameters, MultiLayerPerceptron

def main():
  # Parse arguments
  parser = argparse.ArgumentParser(prog='separate.py', description='Separate the dataset into training dataset and validation dataset')
  parser.add_argument('training_dataset', type=str,
                      help='path to training dataset')
  parser.add_argument('validation_dataset', type=str,
                      help='path to validation dataset')
  parser.add_argument('parameter_file', type=str,
                      help='path to parameter file of the net')

  args = parser.parse_args()

  with open(args.parameter_file, 'r') as file:
    json_data = json.load(file)
    input = Hyperparameters(**json_data)

  mlp = MultiLayerPerceptron(input)

  print(input)


if __name__== "__main__":
  main()