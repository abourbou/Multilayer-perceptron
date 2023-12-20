import argparse
import pandas as pd
import os


# Custom function for float range
def float_range(value):
    value = float(value)
    if 0.0 <= value <= 0.5:
        return value
    else:
        raise argparse.ArgumentTypeError(f"{value} is not in the range [0.0, 0.5]")


# Parse arguments
def Parse():
    parser = argparse.ArgumentParser(
        prog="separate.py",
        description="Preprocess and eparate the dataset into training dataset and validation dataset",
    )
    parser.add_argument("file", type=str, help="file to separate")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed number to randomly separate the dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--perc_validation",
        type=float_range,
        default=0.2,
        help="percentage of data used for validation (default: %(default)s)",
    )

    parser.add_argument(
        "--preprocessing",
        choices=["none", "std", "norm"],
        default="std",
        help="Preprocessing of the data, choice between no preprocessing, standardization or normalization (default: %(default)s)",
    )
    return parser.parse_args()


# Preprocess the data : none, standardization or normalization
def Preprocessing(
    dataset: pd.DataFrame, path_dataset: str, preprocess_mode: str
) -> pd.DataFrame:
    data = pd.read_csv(dataset)
    preprocess_file = open(f"{path_dataset}/preprocess.csv", "w")
    preprocess_file.write(f"#Preprocessing mode : {preprocess_mode}\n")
    if preprocess_file == "none":
        preprocess_file.write("none")
    elif preprocess_mode == "std":
        preprocess_file.write("column,mean,std\n")
        for column in data.columns[1:]:
            preprocess_file.write(
                f"{column},{data[column].mean()},{data[column].std()}\n"
            )
            data[column] = (data[column] - data[column].mean()) / data[column].std()
    elif preprocess_mode == "norm":
        preprocess_file.write("column,min,max\n")
        for column in data.columns[1:]:
            preprocess_file.write(
                f"{column},{data[column].min()},{data[column].max()}\n"
            )
            data[column] = (data[column] - data[column].min()) / (
                data[column].max() - data[column].min()
            )
    print(f"Put preprocessing information in '{preprocess_file.name}'\n")
    return data


# Separate dataset
def Separate(data, path_dataset, perc_validation, seed):
    validation_data = data.sample(frac=perc_validation, random_state=seed)
    training_data = data.drop(validation_data.index)

    validation_path = f"{path_dataset}/validation_data.csv"
    training_path = f"{path_dataset}/training_data.csv"

    validation_data.to_csv(validation_path, index=False, header=False)
    training_data.to_csv(training_path, index=False, header=False)

    print(f"Creation of '{training_path}' and '{validation_path}'\n")


def main():
    # Parse arguments
    args = Parse()

    dataset = args.file
    path_dataset = os.path.dirname(dataset)
    seed = args.seed
    perc_validation = args.perc_validation
    print(
        f"Using seed {seed} to randomly separate '{dataset}' using {(1 - perc_validation) * 100}% for training and {perc_validation * 100}% for validation\n"
    )

    # Preprocessing dataset
    data = Preprocessing(dataset, path_dataset, args.preprocessing)

    # Separate dataset
    Separate(data, path_dataset, perc_validation, seed)


if __name__ == "__main__":
    main()
