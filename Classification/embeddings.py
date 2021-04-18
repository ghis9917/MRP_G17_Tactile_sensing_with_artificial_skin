import numpy as np
from karateclub import GL2Vec, Graph2Vec


def get_embeddings(graphs):
    model = GL2Vec()
    model.fit(graphs)
    embeds = model.get_embedding()
    print(embeds)
    np.save('embeds', embeds)