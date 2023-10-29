# Topic Extraction Test Project

This project includes a topic extraction from the provided New York Times articles dataset from Kaggle. The project is divided into several files, each representing their function in the overall architecture. Gluing them together, is a file called main.py. The purpose of this separation is to make it easier to experiment with different models, embeddings, text preparation strategies, and more. The classes are as follows:

- DataReader: Reads the .csv in the data folder. Project assumes the csv from Kaggle exists in this folder. 
- TextPreparation: Applies text cleaning and stopword removal to the text data. I used a combination of the title of the article and its content as the main text data.
- TextEmbeddings: Uses the sentence_transformers module to use a pretrained sentence embedding models on the text data. 
- TopicModel: Fits a GaussianMixture model to model topics based on the embeddings.
- TopicUtilities: Calculates measures based on the selected topics, such as coherence and top N tokens.
- DataSaver: Save the results of the Topic Extraction.


It also includes a Jupyter Notebook with a short analysis of the results and further possible improvements. It also includes a requirements.txt that is necessary to run the project. To run:


1. run `pip install -r requirements.txt`
2. run `python main.py`



