'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec
from six import iteritems


def parse_args(file_path, output_path, dimension=2, wlength=80, nwalks=10, wsize=10, epochs=2,
			   p=1, q=1, weighted=False, directed=False):
	'''
	Parses the node2vec arguments.
	'''
	parser = argparse.ArgumentParser(description="Run node2vec.")

	parser.add_argument('--input', nargs='?', default=file_path,
	                    help='Input graph path')

	parser.add_argument('--output', nargs='?', default=output_path,
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=dimension,
	                    help='Number of dimensions. Default is 2.')

	parser.add_argument('--walk-length', type=int, default=wlength,
	                    help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=nwalks,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=wsize,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=epochs, type=int,
                      help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=8,
	                    help='Number of parallel workers. Default is 8.')

	parser.add_argument('--p', type=float, default=p,
	                    help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=q,
	                    help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=weighted)

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=directed)

	return parser.parse_args()

def read_graph(args):
	'''
	Reads the input network in networkx.
	'''
	if args.weighted:
		G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	return G

def learn_embeddings(args, walks):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	walks = [map(str, walk) for walk in walks]
	model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
	model.wv.save_word2vec_format(args.output)

	for word, vocab in sorted(iteritems(model.wv.vocab), key=lambda item: -item[1].count):
		print(word, vocab.index)
		print(word, model.wv.syn0[vocab.index])

	return

def main(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	nx_G = read_graph(args)
	G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
	G.preprocess_transition_probs()
	walks = G.simulate_walks(args.num_walks, args.walk_length)
	learn_embeddings(args, walks)

if __name__ == "__main__":
	args = parse_args()
	main(args)
