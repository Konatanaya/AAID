import time
import os
import pickle
import datetime
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


#
# DATASET = ['facebook.txt', 'ciao.txt']
# TOPIC = 2
# DATASET_INDEX = 0
# ITERATION = 200

def quantify_filter_bubble(G, user, topic_num):
    qf_Ri = 0
    total_count_list = [v for k, v in G.nodes()[user]['receiveCount'].items()]
    if len(total_count_list) > 0:
        matrix = np.concatenate(total_count_list).reshape((-1, topic_num))
        s = matrix.sum()
        if s != 0:
            r_x = np.sum(matrix, axis=0) / s
            qf_Ri = np.sum([-1 * v * np.log(v) if v > 0 else 0 for v in r_x])
    return qf_Ri


def quantify_echo_chamber(G, user, topic, prob):
    qe_Nu_topic = []
    ru_x = G.nodes[user]['preference'][topic]
    for v in G[user]:
        rv_x = G.nodes[v]['preference'][topic]
        sim_uv = 1 - abs((0.5 - ru_x) - (0.5 - rv_x))
        prob_uv = prob
        qe_uv_x = sim_uv * prob_uv
        qe_Nu_topic.append(qe_uv_x)
    return qe_Nu_topic
