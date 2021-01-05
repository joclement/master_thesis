from pathlib import Path
import pickle
import sys
import tkinter as tk
from tkinter.filedialog import askopenfilename
import traceback
from typing import Any, Dict, Final

from . import data, prepared_data


CLASSIFIER_PATH: Final = Path("./output/refactored2/dt-finger_both/model.p")
PREPARE_DATA_FUNCTION: Final = prepared_data.finger_both
MODEL_CONFIG: Final[Dict[str, Any]] = {}


class ClassifierGui(tk.Toplevel):
    def __init__(self, parent, *args, **kwargs):
        tk.Tk.report_callback_exception = self._handle_error

        self.parent = parent
        self.label_file_explorer = tk.Label(
            parent, text="", width=100, height=4, fg="blue"
        )
        self.label_file_explorer.grid(column=1, row=1)

        self.button_explore = tk.Button(
            parent,
            text="Select partial discharge CSV data file",
            command=self._browseFiles,
        )
        self.button_explore.grid(column=1, row=2)

        self.polarity_options = ["", data.POS_VOLTAGE, data.NEG_VOLTAGE]
        polarity_variable = tk.StringVar(parent)
        polarity_variable.set(self.polarity_options[0])
        self.polarity = self.polarity_options[0]

        polarity_options_menu = tk.OptionMenu(
            parent,
            polarity_variable,
            *self.polarity_options,
            command=self._polarity_chosen,
        )
        polarity_options_menu.grid(column=1, row=3)

        self.button_classify = tk.Button(
            parent, text="Classify", state=tk.DISABLED, command=self._predict
        )
        self.button_classify.grid(column=1, row=4)

        self.text_field = tk.Text(parent, height=10, width=90)
        self.text_field.grid(column=1, row=5)

        self.csv_data = None
        self.classifier = pickle.load(open(CLASSIFIER_PATH, "rb"))
        self.prepare_data = PREPARE_DATA_FUNCTION
        self.model_config = MODEL_CONFIG

    def _polarity_chosen(self, value):
        self.polarity = value
        self._check_classification_readiness()

    def _check_classification_readiness(self):
        if self.csv_data is not None and self.polarity != self.polarity_options[0]:
            self.button_classify["state"] = "normal"

    def _browseFiles(self):
        filepath = askopenfilename(
            initialdir="./",
            title="Select the measurement file",
            filetypes=(("csv files", "*.csv*"),),
        )
        self.csv_data = data.read(filepath, labeled_file=False)
        self.label_file_explorer.configure(text="File valid: " + filepath)
        self._check_classification_readiness()

    def _predict(self):
        X, _ = self.prepare_data([self.csv_data], **self.model_config)
        prediction = data.Defect(self.classifier.predict(X)[0])
        probabilities_list = self.classifier.predict_proba(X)[0]
        probabilities_dict = {
            data.Defect(idx): prob for idx, prob in enumerate(probabilities_list)
        }
        probabilities_output = "".join(
            [
                f"{data.DEFECT_NAMES[defect]}: {prob}\n"
                for defect, prob in probabilities_dict.items()
            ]
        )
        self.text_field.insert(
            tk.END,
            "Classification results:\n"
            f"class {data.DEFECT_NAMES[prediction]}\n\n"
            "Probabilities:\n"
            f"{probabilities_output}",
        )
        self.button_classify["state"] = "disabled"
        self.label_file_explorer.configure(text="")
        self.csv_data = None

    def _handle_error(self, *args):
        error = traceback.format_exception(*args)
        for line in error:
            print(line)
        tk.messagebox.showerror("Error", error)
        sys.exit()


def main():
    root_window = tk.Tk()
    ClassifierGui(root_window)

    root_window.config(background="white")
    root_window.geometry("800x500")
    root_window.title("UHF HVDC GIL defects classifier")

    root_window.mainloop()
