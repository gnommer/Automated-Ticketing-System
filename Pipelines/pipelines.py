"""
Pipelines Class used for the project

"""

import pandas as pd


class NLP_Pipeline():
    def __init__(self, file_path):
        self.file_path = file_path

    def read_dataset(self):
        self.df = pd.read_excel(self.file_path)
        return self.df
    
    def preprocess_data(self):
        pass

    def train_model(self):
        pass

    def inference(self, X):
        pass