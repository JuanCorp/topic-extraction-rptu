from sklearn.mixture import GaussianMixture
import numpy as np

class TopicModel(object):

    def __init__(self,n_topics=20):
        self.model = GaussianMixture(n_components=n_topics,random_state=777,init_params="k-means++",covariance_type="full")


    def _fit_model(self,embeddings):
        self.model.fit(embeddings)

    
    def _select_topics(self,embeddings):
        probs = self.model.predict_proba(embeddings)
        topics = [np.where(p > 0) for p in probs]
        self.topics = topics
        self.probs = probs.tolist()
    

    def get_topics(self,embeddings):
        self._fit_model(embeddings)
        self._select_topics(embeddings)
        return self.topics


        