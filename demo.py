# BankCard-Recognizer with GUI
#
# Author: Shawn Hu    © Copyright 2019
# License: MIT
#
# Usage: Run this demo with Python.
#

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow

from batch_predict import batch_recognize
from gui.main import UIMainWindow
from gui.app import APP

if __name__ == "__main__":
    config = {
        "input_dir": r"F:\JetBrains\tt",
        "output_file": "./results.csv",
        "east_model": "east/model/east_model.h5",
        "crnn_model": "crnn/model/crnn_model.h5"
    }
    batch_recognize(**config)
