from gem.embedding.gf import GraphFactorization as gf
from gem.embedding import node2vec as nv
from gem.evaluation import evaluate_graph_reconstruction as gr
from gem.utils import graph_util
from gem.evaluation import visualize_embedding as viz
from time import time
import matplotlib.pyplot as plt
from subprocess import call


# em = gf(2, 1000, 1*10**-4, 1.0)
# graph = graph_util.loadGraphFromEdgeListTxt('data/karate.edgelist', directed=False)
#
# Y, t = em.learn_embedding(graph, edge_f=None, is_weighted=True, no_python=True)
# MAP, prec_curv = gr.evaluateStaticGraphReconstruction(graph, em, Y, None)

edge_f = 'data/karate.edgelist'
G = graph_util.loadGraphFromEdgeListTxt(edge_f, directed=False)
G = G.to_directed()
res_pre = 'results/testKarate'
print 'Num nodes: %d, num edges: %d' % (G.number_of_nodes(), G.number_of_edges())
t1 = time()
embedding = nv.node2vec(2, 1, 80, 10, 10, 1, 1)
embedding.learn_embedding(graph=G, edge_f=None, is_weighted=True, no_python=True)
print 'node2vec:\n\tTraining time: %f' % (time() - t1)

viz.plot_embedding2D(embedding.get_embedding(), di_graph=G, node_colors=None)
plt.show()