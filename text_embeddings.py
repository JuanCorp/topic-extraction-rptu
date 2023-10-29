from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import numpy as np

class TextEmbeddingGenerator(object):

    def __init__(self,texts,model='all-MiniLM-L6-v2'):
        self.model = model
        self.texts = texts

    def _load_transformer(self):
        self.transformer = SentenceTransformer(self.model)


    def _generate_text_embeddings(self):
        text_embeddings = list()
        for text in self.texts.values:
            tokenized_sentences = sent_tokenize(text)
            sentence_embeddings = self.transformer.encode(tokenized_sentences)
            averaged_sentences = sentence_embeddings.mean(axis=0)
            text_embeddings.append(averaged_sentences)
        
        return np.array(text_embeddings)
    

    def calculate_embeddings(self):
        self._load_transformer()
        embeddings = self._generate_text_embeddings()
        return embeddings
    

        
        