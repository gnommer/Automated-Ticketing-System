import warnings

import pandas as pd

from .preprocessing import Preprocessing

warnings.filterwarnings('ignore')


class NLP_Pipeline():
    def __init__(self, file_path):
        self.file_path = file_path

    def read_dataset(self):
        self.df = pd.read_excel(self.file_path, encoding='utf-32')
        self.df["Assignment group"] = self.df["Assignment group"].apply(lambda x: int(x.split("_")[1]))
        return self.df
    
    def preprocess_data(self):
        self.df = self.df.dropna()
        pp1 = Preprocessing(self.df["Description"].values)
        pp2 = Preprocessing(self.df["Short description"].values)
        self.df["clean_des"] = pp1.run()
        self.df["clean_sdes"] = pp2.run()
        self.df["des_lang"] = self.df["clean_des"].apply(lambda x: x._.language[0])
        self.df["sdes_lang"] = self.df["clean_sdes"].apply(lambda x: x._.language[0])
        self.df["des_has_email"] = self.df["clean_des"].apply(lambda x: x._.has_email)
        self.df["sdes_has_email"] = self.df["clean_sdes"].apply(lambda x: x._.has_email)
        self.df["des_has_domain"] = self.df["clean_des"].apply(lambda x: x._.has_domain)
        self.df["sdes_has_domain"] = self.df["clean_sdes"].apply(lambda x: x._.has_domain)
        self.df["des_has_url"] = self.df["clean_des"].apply(lambda x: x._.has_url)
        self.df["sdes_has_url"] = self.df["clean_sdes"].apply(lambda x: x._.has_url)
        return self.df

    def train_model(self):
        pass

    def inference(self, X):
        pass
