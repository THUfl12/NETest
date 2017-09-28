from gem.embedding.gf import GraphFactorization as gf
from gem.embedding.lle import LocallyLinearEmbedding as lle
from gem.embedding.hope import HOPE
from gem.embedding.lap import LaplacianEigenmaps as lap

from gem.node2vec.main import parse_args as input_args
from gem.node2vec.main import main as nv

from gem.evaluation import evaluate_graph_reconstruction as gr
from gem.utils import graph_util
from gem.evaluation import visualize_embedding as viz
from time import time
import matplotlib.pyplot as plt
from subprocess import call


edge_f = 'data/karate.edgelist'
G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=False)
res_pre = 'results/testKarate'
print 'Num nodes: %d, num edges: %d' % (G.number_of_nodes(), G.number_of_edges())

# graph factorization
embedding = gf(2, 100, 1*10**-4, 1.0)   # d, max_iter, eta, regu
Y, t = embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
print 'Graph Factorization:\n\tTraining time: %f' % t

# Locally Linear Embedding
embedding = lle(2)  # d
Y, t = embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
print 'Locally Linear Embedding:\n\tTraining time: %f' % t

# Hope
embedding = HOPE(4, 0.01)   # d, beta
Y, t = embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
print 'HOPE:\n\tTraining time: %f' % t

# Laplacian Eigen maps
embedding = lap(4)   # d
Y, t = embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
print 'Laplacian Eigenmaps:\n\tTraining time: %f' % t

# node2vec
args = input_args(file_path='data/karate.edgelist', output_path='data/karate.emb')
nv(args)


