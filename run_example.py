from graph2vec import Graph2Vec


graph_files = [
    'examples/test.gexf',
    'examples/test2.gexf',
    'examples/test3.gexf',
]

g2v = Graph2Vec(graph_files, max_degree=2, node_label_attr_name='Value', edge_label_attr_name='Value')
docs = g2v.weisfeiler_lehman_relabel()
g2v_dt = g2v.train(docs, vector_size=100, epochs=50)

for g_name in g2v_dt:
    print(f'{g_name}: {g2v_dt[g_name]}')

