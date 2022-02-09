import numpy as np
import networkx as nx
import time
import pickle


# DATASET = ['facebook', 'ciao_clean']
# TOPIC = 5
# DATASET_INDEX = 1


def generate_network(file_path, TOPIC):
    G = nx.read_edgelist(file_path, create_using=nx.DiGraph(), nodetype=int)

    # generate a random network for test
    # G = nx.gnm_random_graph(10, 30, directed=True)

    print(nx.info(G))

    node_dic = {}
    edge_dic = {}

    for u in G.nodes():
        node_dic[u] = {
            'preference': np.ones(TOPIC)/TOPIC,
            'sendList': {},  # action
            'sendCount': {},
            'receiveList': {},
            'receiveCount': {},
        }
    nx.set_node_attributes(G, node_dic)
    nx.set_edge_attributes(G, edge_dic)

    # save generated network
    pickle.dump(G, open(file_path[:-4] + '_' + str(TOPIC) + '.G', 'wb'))
    pickle.dump(node_dic, open(file_path[:-4] + '_' + str(TOPIC) + '.node', 'wb'))
    pickle.dump(edge_dic, open(file_path[:-4] + '_' + str(TOPIC) + '.edge', 'wb'))


def load_network(dataset, TOPIC):
    path = './dataset/' + dataset + '_' + str(TOPIC) + '.G'
    G = pickle.load(open(path, 'rb'))
    return G


def init_dataset(dataset, TOPIC):
    path = './dataset/' + dataset + '.txt'
    generate_network(path, TOPIC)

