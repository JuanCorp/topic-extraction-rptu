from data_reader import DataReader
from data_saver import DataSaver
from text_prep import TextPreparation
from text_embeddings import TextEmbeddingGenerator
from topic_model import TopicModel
from topic_utilities import TopicUtilities




def run_experiment():
    print("Reading Data")
    dr = DataReader()
    text_data = dr.obtain_text_data()
    
    print("Preparing Text")
    tp = TextPreparation(text_data)
    prepped_text = tp.prepare_text()

    print("Calculating Embeddings")
    teg = TextEmbeddingGenerator(prepped_text)
    embeddings = teg.calculate_embeddings()

    print("Generating Topics")
    model = TopicModel()
    topics = model.get_topics(embeddings)

    print("Calculating Utilities")
    utils = TopicUtilities()
    utils.create_utility_objects(prepped_text)
    top_tokens = utils.get_top_topic_tokens(topics)
    coherence = utils.get_coherence(top_tokens)
    


    final_object = {"coherence":coherence,"top_tokens":top_tokens,"probabilities":model.probs}

    print("Saving...")
    data_saver = DataSaver()
    data_saver.save_object(final_object,"results.json")
    print("Done")


if __name__ == "__main__":
    run_experiment()




