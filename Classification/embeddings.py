import random
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from karateclub import GL2Vec, Graph2Vec, FeatherGraph, IGE, FGSD
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib.colors import ListedColormap
import os


def get_embeddings(graphs):
    # create a new directory if it doesn't exist yet
    ROOT_DIR = os.path.abspath(os.curdir)
    path = ROOT_DIR + '/embeddings'
    try:
        os.mkdir(path)
    except OSError:
        pass

    indexes = [len(gesture) for gesture in graphs]
    flatten_graph_list = [item for sublist in graphs for item in sublist]

    embeddings = {
        'graph2vec': Graph2Vec(),
        'gl2vec': GL2Vec(),
        # 'feather_graph': FeatherGraph(),
        # 'ige': IGE(),
        # 'fgsd': FGSD()
    }

    for name, func in embeddings.items():
        model = func
        model.fit(flatten_graph_list)
        np.save(f'embeddings/embeddings_{name}', model.get_embedding())

    np.save('embeddings/index_embeddings', np.array(indexes))
    print(f'embeddings successfully created for {len(flatten_graph_list)} graphs')


def embeddings_PCA(embeddings, indexes=None, components=3):
    # print(embeddings.shape, indexes.shape if indexes else '')

    pca = PCA(n_components=components).fit_transform(embeddings)

    if indexes:
        visualize_3d_plot(pca, indexes)
    return pca


def visualize_embeddings_TSNE(embeddings, indexes, n_samples=4):
    print(embeddings.shape, indexes.shape)
    print('ids ', [value for value in indexes])

    # TODO get 10 frames in the center

    # extract number of samples to visualize
    gestures = []
    start = 0
    for entry in indexes:
        gestures.append(embeddings[start:start + entry])
        start += entry
    gestures = np.asarray(gestures)
    ids = np.arange(len(gestures))
    samples = random.sample(list(ids), k=n_samples)
    print('samples ', samples)

    embedding_samples = []
    for sample in samples:
        embedding_samples.append(gestures[sample])
    embedding_samples = [j for sub in embedding_samples for j in sub]

    # reduce dimensionality using PCA
    pca = embeddings_PCA(np.asarray(embedding_samples), components=50)

    # calculate TSNE
    tsne = TSNE(n_components=3, perplexity=50).fit_transform(pca)
    visualize_3d_plot(tsne, [indexes[id] for id in samples])
    return tsne


def visualize_3d_plot(components, indexes):
    fig = plt.figure(figsize=(6, 6))
    ax = Axes3D(fig)

    # get colormap from seaborn
    cmap = ListedColormap(sns.color_palette("husl", 256).as_hex())

    start = 0
    for count, entry in enumerate(indexes):
        ax.scatter(components[start:start + entry, 0], components[start:start + entry, 1],
                   components[start:start + entry, 2], label=count, s=20, marker='o', cmap=cmap, alpha=1)
        ax.legend()
        start += entry
    plt.show()


def visualize_pearson_correlation(embeddings):
    print(embeddings.shape)
    pearson = np.corrcoef(embeddings.T)
    print(pearson.shape)
    #plt.imshow(pearson, cmap='hot', interpolation='nearest')
    sns.heatmap(pearson, linewidth=0.0001)
    plt.show()


def visualization():
    try:
        labels = np.load('datasets/labels.npy')
    except FileNotFoundError:
        print('Not possible to load data. Directory "datasets" does not exist. '
              '\nPlease create directory and add "labels.npy"')
        exit()
    embeddings = np.load('datasets/embeddings_tests_gl2vec.npy')
    indexes = np.load('datasets/index_embeddings.npy')
    #visualize_embeddings_TSNE(embeddings, indexes, n_samples=3)
    #visualize_pearson_correlation(embeddings)