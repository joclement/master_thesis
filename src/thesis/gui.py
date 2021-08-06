from pathlib import Path
import sys
import tkinter as tk
from tkinter.filedialog import askopenfilename
import traceback
from typing import Final

from . import data
from .predict import PredictionHandler


MODEL_FILES_DIR: Final = Path(Path(__file__).parent, "./model_files")
FINGER_PREPROCESSOR_PATH: Final = Path(MODEL_FILES_DIR, "./finger_preprocessor.p")
PREPROCESSOR_PATH: Final = Path(MODEL_FILES_DIR, "./preprocessor.p")
MODEL_PATH: Final = Path(MODEL_FILES_DIR, "./model.p")

UNSET_POLARITY_VAR: Final = ""


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

        self.polarity_variable = tk.StringVar(parent)

        polarity_options_menu = tk.OptionMenu(
            parent,
            self.polarity_variable,
            *[UNSET_POLARITY_VAR, data.POS_VOLTAGE, data.NEG_VOLTAGE],
            command=self._polarity_chosen,
        )
        polarity_options_menu.grid(column=1, row=3)

        self.button_classify = tk.Button(
            parent, text="Classify", state=tk.DISABLED, command=self._predict
        )
        self.button_classify.grid(column=1, row=4)

        self.text_field = tk.Text(parent, height=10, width=90)
        self.text_field.grid(column=1, row=5)

        if FINGER_PREPROCESSOR_PATH.exists():
            finger_preprocessor_path = FINGER_PREPROCESSOR_PATH
        else:
            finger_preprocessor_path = None
        self.predictionHandler = PredictionHandler(
            PREPROCESSOR_PATH,
            MODEL_PATH,
            finger_preprocessor_path=finger_preprocessor_path,
        )

        self._reset()

    def _reset(self):
        self.polarity_variable.set(UNSET_POLARITY_VAR)
        self.button_classify["state"] = "disabled"
        self.label_file_explorer.configure(text="")
        self.df = None
        self.filepath = None

    def _polarity_chosen(self, _):
        self._check_classification_readiness()

    def _check_classification_readiness(self):
        if self.df is not None and self.polarity_variable.get() != UNSET_POLARITY_VAR:
            self.text_field.delete(1.0, tk.END)
            self.button_classify["state"] = "normal"

    def _browseFiles(self):
        filepath = askopenfilename(
            initialdir="./",
            title="Select the measurement file",
            filetypes=(("csv files", "*.csv*"),),
        )
        if isinstance(filepath, str):
            self.df = data.read(filepath, labeled_file=False)
            self.label_file_explorer.configure(text="File valid: " + filepath)
            self._check_classification_readiness()
            self.filepath = filepath

    def _predict(self):
        self.df.attrs[data.VOLTAGE_SIGN] = data.VoltageSign.from_str(
            self.polarity_variable.get()
        )
        prediction, probabilities_list, _, _ = self.predictionHandler.predict_one(
            self.df
        )
        probabilities_dict = {
            data.Defect(idx): prob for idx, prob in enumerate(probabilities_list)
        }
        probabilities_output = "".join(
            [
                f"  {str(defect)}: {prob}\n"
                for defect, prob in probabilities_dict.items()
            ]
        )
        self.text_field.insert(
            tk.END,
            f"Filepath: {self.filepath}\n\n"
            f"Classification result: {str(prediction)}\n"
            "Probabilities:\n"
            f"{probabilities_output}",
        )

        self._reset()

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
