import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__), description="Analyse and filter the dataset"
    )
    parser.add_argument("file", type=str, help="file to analyse and filter")
    parser.add_argument(
        "--filter_only",
        action="store_true",
        help="skip the analysis and only filter dataset with choosen features",
    )
    args = parser.parse_args()

    columns = [
        "ID",
        "Diagnosis",
        "M Radius",
        "M Texture",
        "M Perimeter",
        "M Area",
        "M Smoothness",
        "M Compactness",
        "M Concavity",
        "M Concave Points",
        "M Symmetry",
        "M Fractal Dim",
        "SE Radius",
        "SE Texture",
        "SE Perimeter",
        "SE Area",
        "SE Smoothness",
        "SE Compactness",
        "SE Concavity",
        "SE Concave Points",
        "SE Symmetry",
        "SE Fractal Dim",
        "W Radius",
        "W Texture",
        "W Perimeter",
        "W Area",
        "W Smoothness",
        "W Compactness",
        "W Concavity",
        "W Concave Points",
        "W Symmetry",
        "W Fractal Dim",
    ]
    data = pd.read_csv(args.file, names=columns)

    # Visualize the dataset using Seaborn
    if not args.filter_only:
        sns.set(style="whitegrid")

        # Pairplot of the 4 first mean features
        selected_features = [
            "M Radius",
            "M Texture",
            "M Perimeter",
            "M Area",
            "Diagnosis",
        ]
        sns.pairplot(data[selected_features], hue="Diagnosis", palette="husl")
        plt.suptitle("Pairplot of the 4 first mean features")
        plt.show()

        # Pairplot of the subset of mean features
        selected_features = [
            "M Radius",
            "M Texture",
            "M Smoothness",
            "M Compactness",
            "M Concavity",
            "M Concave Points",
            "M Symmetry",
            "M Fractal Dim",
            "Diagnosis",
        ]
        sns.pairplot(data[selected_features], hue="Diagnosis", palette="husl")
        plt.suptitle("Pairplot of the mean features")
        plt.show()

        # Pairplot of the subset of Standard Error features
        selected_features = [
            "SE Radius",
            "SE Texture",
            "SE Perimeter",
            "SE Area",
            "SE Smoothness",
            "SE Compactness",
            "SE Concavity",
            "SE Concave Points",
            "SE Symmetry",
            "SE Fractal Dim",
            "Diagnosis",
        ]
        sns.pairplot(data[selected_features], hue="Diagnosis", palette="husl")
        plt.suptitle("Pairplot of the SE features")
        plt.show()

        # Pairplot of the subset of Worst features
        selected_features = [
            "W Radius",
            "W Texture",
            "W Smoothness",
            "W Compactness",
            "W Concavity",
            "W Concave Points",
            "Diagnosis",
        ]
        sns.pairplot(data[selected_features], hue="Diagnosis", palette="husl")
        plt.suptitle("Pairplot of the worst features")
        plt.show()

    # Filter the data
    selected_features = data[
        [
            "Diagnosis",
            "M Radius",
            "M Texture",
            "M Compactness",
            "M Concavity",
            "M Concave Points",
            "SE Radius",
            "SE Perimeter",
            "SE Area",
            "SE Compactness",
            "SE Concavity",
            "SE Concave Points",
            "W Radius",
            "W Texture",
            "W Smoothness",
            "W Compactness",
            "W Concavity",
            "W Concave Points",
        ]
    ]

    filter_name = os.path.dirname(args.file) + "/filter_data.csv"
    selected_features.to_csv(filter_name, index=False, header=True)
    print(f"Save the filtered data in {filter_name}")


if __name__ == "__main__":
    main()
