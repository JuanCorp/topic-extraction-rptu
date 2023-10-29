import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize


class TopicUtilities(object):

    def __init__(self,coherence_mode="u_mass",n_topics=20):
        self.coherence_mode = coherence_mode
        self.n_topics = n_topics


    def create_utility_objects(self,data):
        self.tokenized_word_sentences = [word_tokenize(s) for s in data.values]
        self.id2word = corpora.Dictionary(self.tokenized_word_sentences)
        self.corpus  = [self.id2word.doc2bow(text) for text in self.tokenized_word_sentences]

    

    def get_top_topic_tokens(self,topics):
        topic_top_n = list()
        for topic in range(self.n_topics):
            topic_indexes = [i for i,t in enumerate(topics) for p in t if topic in p]
            topic_sentences = [t for i,t in enumerate(self.tokenized_word_sentences) if i in topic_indexes]
            all_topic_words = [word for sentence in topic_sentences for word in sentence]
            frequency = FreqDist(all_topic_words)
            top_n = frequency.most_common(5)
            top_n_words = [t[0] for t in top_n]
            topic_top_n.append(top_n_words)
        
        self.topic_top_n = topic_top_n
        return topic_top_n
    

    def get_coherence(self,top_tokens):
        cm = CoherenceModel(topics=top_tokens,texts = self.tokenized_word_sentences,corpus=self.corpus, dictionary=self.id2word, coherence=self.coherence_mode)
        coherence = cm.get_coherence()
        return coherence

    

    

