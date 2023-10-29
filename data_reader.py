import pandas as pd


class DataReader(object):

    def __init__(self,filepath="data/",filename="nytimes front page.csv",usecols=["title","content"]):
        self.filepath=filepath
        self.filename=filename
        self.usecols=usecols
    

    def _read_data(self):
        full_filepath = self.filepath + self.filename
        self.data= pd.read_csv(full_filepath,usecols=self.usecols)
    
    def _select_text_features(self):
        text = self.data[self.usecols[0]]
        for col in self.usecols[1:]:
            text+=self.data[col]
        self.text = text
    

    def obtain_text_data(self,**kwargs):
        self._read_data()
        self._select_text_features()
        return self.text
        
    
