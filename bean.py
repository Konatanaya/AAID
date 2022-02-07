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
            'preference': np.random.random(TOPIC),
            'sendList': {},  # action
            'sendCount': {},
            'receiveList': {},
            'receiveCount': {},
        }

    # physical influence probability PIP = 1/(dT)(r^x_i * s^x_i - r^x_j * s^x_j)(r^x_i/r^x_j)
    # assign PIP
    for u in G.nodes():
        for v in G[u]:
            # todo use len s by topics
            # prob = node_dic[u]['influence'] * (node_dic[u]['preference'] * node_dic[u]['receiveList'] + node_dic[v]['preference'] * node_dic[v]['receiveList']) * (node_dic[u]['preference'] / node_dic[v]['preference'])
            edge_dic[(u, v)] = {'PIP': 0.1}
    nx.set_node_attributes(G, node_dic)
    nx.set_edge_attributes(G, edge_dic)

    # save generated network
    pickle.dump(G, open(file_path[:-4] + '_' + str(TOPIC) + '.G', 'wb'))
    pickle.dump(node_dic, open(file_path[:-4] + '_' + str(TOPIC) + '.node', 'wb'))
    pickle.dump(edge_dic, open(file_path[:-4] + '_' + str(TOPIC) + '.edge', 'wb'))


# physical influence probability PIP = 1/(dT)(r^x_i * s^x_i - r^x_j * s^x_j)(r^x_i/r^x_j)
#  return 1/dT
# def assign_edge_values(G, u, TOPIC):
#     d = len(G[u])
#     print(len(list(G[u])),len(list(G.successors(u))))
#     if d == 0:
#         edge_value = 0
#     else:
#         edge_value = 1 / (d * TOPIC)
#     return edge_value

def load_network(dataset, TOPIC):
    path = './dataset/' + dataset + '_' + str(TOPIC) + '.G'
    G = pickle.load(open(path, 'rb'))
    return G


def init_dataset(dataset, TOPIC):
    path = './dataset/' + dataset + '.txt'
    generate_network(path, TOPIC)


if __name__ == '__main__':
    # path = './dataset/'+ DATASET[DATASET_INDEX] + '.txt'
    path = './dataset/' + 'facebook' + '.txt'
    generate_network(path, 5)

    # load_network(path)
