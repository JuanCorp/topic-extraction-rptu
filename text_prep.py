from nltk.corpus import stopwords
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer



class TextPreparation(object):
    def __init__(self,text_series,cv_params={"min_df":0.001,"max_df":0.7}):
        self.text_series = text_series
        self.cv_params = cv_params

    def _get_stopwords(self):
        nltk.download("stopwords")
        english_stopwords = stopwords.words("english")
        self.stopwords = english_stopwords

    
    def _clean_text(self,text):
        self._get_stopwords()
        lower_text = text.str.lower()
        alpha_text =  lower_text.str.replace(r"(@\[a-z0-9]+)|([^0-9a-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "")
        nostops = alpha_text.apply(lambda x: " ".join([word for word in x.split() if word not in self.stopwords]))
        return nostops
    
    def _filter_words(self,text):
        cv = CountVectorizer(min_df=self.cv_params["min_df"],max_df=self.cv_params["max_df"])
        cv.fit(text)
        filtered_text = text.apply(lambda x: " ".join([word for word in x.split() if word in cv.vocabulary_]))
        return filtered_text
    
    def prepare_text(self,pipeline=["clean","filter"]):
        functions = {"clean":self._clean_text,"filter":self._filter_words}
        text = self.text_series
        for step in pipeline:
            text = functions[step](text)
        
        return text


        
        

        