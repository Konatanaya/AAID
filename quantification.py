import time
import os
import pickle
import datetime
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def quantify_filter_bubble(G, user, topic_num):
    qf_Ri = 0
    total_count_list = [v for k, v in G.nodes()[user]['sendCount'].items()]
    if len(total_count_list) > 0:
        matrix = np.concatenate(total_count_list).reshape((-1, topic_num))
        s = matrix.sum()
        if s != 0:
            r_x = np.sum(matrix, axis=0) / s
            qf_Ri = np.sum([-1 * v * np.log(v) if v > 0 else 0 for v in r_x])
    return qf_Ri


def quantify_echo_chamber(G, user, time_step, topic_num):
    Q = 0.
    if time_step > 0:
        user_pre = G.nodes[user]['preference']
        neighbors = set([msg.sender for msg in G.nodes[user]['receiveList'][time_step - 1]])
        if len(neighbors) > 0:
            neighbors_pre = np.array([G.nodes[u]['preference'] for u in list(neighbors)]).reshape(-1, len(user_pre))
            Q = user_pre.dot(neighbors_pre.T) / (np.linalg.norm(user_pre) * np.linalg.norm(neighbors_pre, axis=1))
            Q = np.average(Q)

    return Q
