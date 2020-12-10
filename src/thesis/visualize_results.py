import pandas as pd

from . import classify, data


def main():
    DATA_DIR = "./data/csv-files/prepared/for_debugging/longer"
    RESULTS_DIR = "/tmp"
    MEAN_RESULTS = f"{RESULTS_DIR}/classifiers_balanced_accuracy_means.csv"
    STD_RESULTS = f"{RESULTS_DIR}/classifiers_balanced_accuracy_stds.csv"
    OUTPUT_DIRECTORY = "./output/tmp"

    measurements, _ = data.read_recursive(DATA_DIR)
    classificationHandler = classify.ClassificationHandler(
        measurements, OUTPUT_DIRECTORY, False
    )
    classificationHandler.mean_accuracies = pd.read_csv(MEAN_RESULTS, index_col=0)
    classificationHandler.std_accuracies = pd.read_csv(STD_RESULTS, index_col=0)
    classificationHandler.plot_results()
