import argparse
import pandas as pd

# Custom function for float range
def float_range(value):
    value = float(value)
    if 0.0 <= value <= 0.5:
        return value
    else:
        raise argparse.ArgumentTypeError(f"{value} is not in the range [0.0, 0.5]")

def main():
  # Parse arguments
  parser = argparse.ArgumentParser(prog='separate.py', description='Separate the dataset into training dataset and validation dataset')
  parser.add_argument('file', type=str,
                      help='file to separate')
  parser.add_argument('--seed', type=int, default=42,
                      help='seed number to randomly separate the dataset (default 42)')
  parser.add_argument('--perc_validation', type=float_range, default=0.2,
                      help='percentage of data used for validation (default 20%)')
  args = parser.parse_args()

  dataset = args.file
  seed = args.seed
  perc_validation = args.perc_validation
  print(f"Using seed {seed} to randomly separate \'{dataset}\' using {(1 - perc_validation) * 100}% for training and {perc_validation * 100}% for validation")

  # Separate dataset
  data = pd.read_csv(dataset)

  validation_data = data.sample(frac=perc_validation, random_state=seed)
  training_data = data.drop(validation_data.index)

  validation_data.to_csv("data/validation_data.csv", index=True, header=False)
  training_data.to_csv("data/training_data.csv", index=True, header=False)

if __name__== "__main__":
  main()