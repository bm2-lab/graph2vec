import os
import networkx as nx
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

class Graph2Vec:
    def __init__(self, fnames, max_degree, node_label_attr_name, edge_label_attr_name):
        self.fnames = fnames
        self.max_degree = max_degree
        self.node_label_attr_name = node_label_attr_name
        self.edge_label_attr_name = edge_label_attr_name
        self.label_to_compressed_label_map = {}
        self.get_base_name = lambda fname: os.path.basename(fname).split('.')[0]
        self.get_int_node_label = lambda l: int(l.split('+')[-1])
        self.graph_tag_to_vec_dict = {}

    def initial_relabel(self, g):
        g = nx.convert_node_labels_to_integers(g, first_label=0)
        for node in g.nodes():
            g.node[node]['relabel'] = {}

        for node in g.nodes:
            try:
                label = g.node[node][self.node_label_attr_name]
            except:
                g.node[node]['relabel'][0] = '0+0'
                continue
            if label not in self.label_to_compressed_label_map:
                compressed_label = len(self.label_to_compressed_label_map) + 1
                self.label_to_compressed_label_map[label] = compressed_label
                g.node[node]['relabel'][0] = f'0+{compressed_label}'
            else:
                g.node[node]['relabel'][0] = f'0+{self.label_to_compressed_label_map[label]}'
        return g

    def wl_relabel(self, g, num_iter):
        prev_iter = num_iter - 1
        for node in g.node:
            prev_iter_node_label = self.get_int_node_label(g.node[node]['relabel'][prev_iter])
            node_label = [prev_iter_node_label]
            neighbors = list(nx.all_neighbors(g, node))
            neighborhood_label = sorted([(self.get_int_node_label(g.node[nei]['relabel'][prev_iter]), int(g.edges[node, nei][self.edge_label_attr_name])) for nei in neighbors])
            node_neighborhood_label = tuple(node_label + neighborhood_label)
            if node_neighborhood_label not in self.label_to_compressed_label_map:
                compressed_label = len(self.label_to_compressed_label_map) + 1
                self.label_to_compressed_label_map[node_neighborhood_label] = compressed_label
                g.node[node]['relabel'][num_iter] = f'{num_iter}+{compressed_label}'
            else:
                g.node[node]['relabel'][num_iter] = f'{num_iter}+{self.label_to_compressed_label_map[node_neighborhood_label]}'
        return g


    def extract_subgraph(self, g):
        sentences = []
        for n in g.node:
            d = g.node[n]
            subsentences = []
            for i in range(self.max_degree+1):
                try:
                    center = d['relabel'][i]
                except:
                    continue
                neis_labels_prev_deg = []
                neis_labels_next_deg = []
                if i != 0:
                    neis_labels_prev_deg = list(set([g.node[nei]['relabel'][i-1] for nei in nx.all_neighbors(g, n)]))
                    neis_labels_prev_deg.sort()
                neis_labels_same_deg = list(set([g.node[nei]['relabel'][i] for nei in nx.all_neighbors(g,n)]))
                if i != self.max_degree:
                    neis_labels_next_deg = list(set([g.node[nei]['relabel'][i+1] for nei in nx.all_neighbors(g,n)]))
                    neis_labels_next_deg.sort()
                nei_list = neis_labels_same_deg + neis_labels_prev_deg + neis_labels_next_deg
                nei_list = '_'.join(nei_list)
                subsentence = f'{center}_{nei_list}'
                subsentences.append(subsentence)
            sentence = '_'.join(subsentences)
            sentences.append(sentence)
        return sentences

    def weisfeiler_lehman_relabel(self):
        graphs = [nx.read_gexf(fname) for fname in self.fnames]
        graphs = [self.initial_relabel(g) for g in graphs]
        for it in range(1, self.max_degree+1):
            self.label_to_compressed_label_map = {}
            graphs = [self.wl_relabel(g, it) for g in graphs]
            print(f'num of WL rooted subgraphs in iter {it} is {len(self.label_to_compressed_label_map)}')
        doc_words = [self.extract_subgraph(g) for g in graphs]
        self.tags = [self.get_base_name(fname) for fname in self.fnames]
        docs = [TaggedDocument(w, tags=[f'{self.tags[i]}']) for (i, w) in enumerate(doc_words)]
        return docs

    def train(self, docs, vector_size=128, epochs=5, learning_rate=0.025):
        self.model = Doc2Vec(docs, vector_size=vector_size, window=0, dm=0, epochs=epochs, alpha=learning_rate, min_count=1)
        self.graph_tag_to_vec_dict = dict.fromkeys(self.tags)
        for t in self.tags:
            self.graph_tag_to_vec_dict[t] = self.model.docvecs[t]
        return self.graph_tag_to_vec_dict
