import numpy as np
from karateclub import GL2Vec, Graph2Vec


def get_embeddings(graphs):
    indexes = [len(gesture) for gesture in graphs]
    flatten_graph_list = [item for sublist in graphs for item in sublist]
    model = Graph2Vec()
    model.fit(flatten_graph_list)
    embeds = model.get_embedding()
    print(embeds)

    np.save('datasets/embeddings', embeds)
    np.save('datasets/index_embeddings', np.array(indexes))