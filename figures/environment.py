import networkx as nx
import numpy as np
import datetime
from matplotlib import pyplot as plt
import time
import pickle
import csv
import bean
import random
import quantification
import copy

DATASET = ['facebook', 'ciao_clean', 'random']
TOPIC = 5
DATASET_INDEX = 2
PROB_SEND_MSG = 0.2
ITERATION = 100
LAMBDA = 0.5
FEED = 1


class environment:
    def __init__(self, G):
        self.G = G
        self.statTime = datetime.datetime.now()
        self.nodes = {}
        self.edges = {}

        # network attributes
        self.repository = []
        # self.preferences = []
        self.topics = []

    # def init(sitory = np.array([G.nodes[user]['sendList'] for user in self.G.nodes()])

    # AI feed message to users based on AI recommendation strategies
    def feed_messages(self, user, iter):
        feed = []
        #     randomly feed messages.
        indexes = np.random.randint(len(self.repository), size=FEED)
        for i in indexes:
            G.nodes[user]['receiveList'].append(self.repository[i])
            feed.append(self.repository[i])
        return feed

    """
    At each time step t, each user agent has a propability p to send a message which contains its most interested topic to AI platform.
    """

    def send_message(self, user, iter):
        topic = np.argmax(G.nodes[user]['preference'])
        if random.random() < PROB_SEND_MSG:
            curtime = iter
            msg = bean.message(topic,user, curtime)
            self.repository.append(msg)
            G.nodes[user]['sendList'].append(msg)
            return True
        else:
            return False

    def influence_diffusion(self):
        for iter in range(ITERATION):
            new_active = []
            filter_bubble_row = []
            echo_chamber_row = []
            for u in env.G.nodes:
                env.send_message(u, iter)
                # AI influence diffusion
                if iter > 0:
                    feed = env.feed_messages(u, iter)
                    qf_Ri = quantification.quantify_filter_bubble(G,  u)
                    filter_bubble_row.append(qf_Ri)
                    influenced = env.calculate_influence(u, feed, iter)
                    if influenced[0]:
                        # take action
                        # todo
                        new_active.append(u)
            print('======== ITERATION ' + str(iter) + '=======')
            print('======== FILTER BUBBLE ' + str(filter_bubble_row) + '=======')

    def influence_diffusion_without_AI(self, mc):
        state = np.zeros((len(G.nodes())), dtype=np.int64)

        activated = []
        seed = []
        coverage = 0
        old_state = state.copy()
        state = old_state.copy()
        new_state = state.copy()
        for iter in range(ITERATION):
            new_active = []
            filter_bubble_row = []
            for u in env.G.nodes:
                if env.send_message(u, iter):
                    seed.append(u)
                    activated.append(seed)
            if iter > 0:
                activated_num = 0

                for j in range(mc - 1):
                    new_activated = []
                    for m in range(len(activated)):
                        for n in G.nodes():
                            for m in list(self.G.predecessors(n)):
                                if (new_state[n] == 0):
                                    rand = random.random() * 10
                                    if (rand < self.G.edges[(m, n)]['PIP']):
                                        print('influenced ' + 'random: ' + str(rand) + 'PIP: ' + str(self.G.edges[(m, n)]['PIP']))
                                        new_state[n] = 1
                                        new_activated.append(n)
                                else:
                                    continue
                    activated = new_activated.copy()
                    activated_num += len(new_activated)
                    coverage += activated_num
            # qf_Ri = quantification.quantify_filter_bubble(G,u)
            # filter_bubble_row.append(qf_Ri)
            print('======== ITERATION ' + str(iter) + ' =======')
            print('======== INFLUENCE COVERAGE ' + str(coverage/mc) + ' =======')
            # print('======== FILTER BUBBLE ' + str(filter_bubble_row) + '=======')


    # prob = TP * [\lambda * PIP - (1-\lambda) * VIP]
    def calculate_influence(self, user, feed, iter):
        # message list at time t-1
        if iter > 0:
            Mi = get_arr_by_time(G.nodes()[user]['receiveList'], iter-1)
            M = get_arr_by_time(self.repository, iter-1)
            # print(len(M))
            si = np.array(self.calculate_topic_distribution(Mi),dtype=float)
            s = np.array(self.calculate_topic_distribution(M),dtype=float)

        # TP = R^{xy}(t) * s^{xy}_i(t-1)/s^{xy}(t-1)
        #     TP = si/s
            TP = [0] * len(s)
            for index in range(len(s)):
                if s[index] == 0:
                    TP[index] = 0
                else:
                    TP[index] = si[index] / s[index]

            for v in G.nodes() - G.nodes()[user]:
                #     VIP = 1/nT p^(x,x') * \beta_i * s[Ms_i \cap Mr_j]/s[Ms_i \cup Mr_j]
                # s[Ms_i \cap Mr_j]/s[Ms_i \cup Mr_j]
                Mrj = np.array(get_arr_by_time(G.nodes()[v]['receiveList'], iter - 1), dtype=bean.message)
                Msi = np.array(get_arr_by_time(G.nodes()[user]['sendList'], iter - 1), dtype=bean.message)
                # print(len(M))
                intersection = set(Mrj).intersection(set(Msi))
                intersection_topic = self.calculate_topic_distribution(intersection)
                # print(intersection_topic)
                union = set(Mrj).union(set(Msi))
                union_topic = self.calculate_topic_distribution(union)
                if len(union_topic) > 0:
                    VIP = [0] * len(union_topic)
                    for index in range(len(union_topic)):
                        if union_topic[index] == 0:
                            VIP[index] = 0
                        else:
                            VIP[index] = 1 /TOPIC * (intersection_topic[index] / union_topic[index])
                else:
                    VIP = np.zeros(TOPIC)
        else:
            TP = np.zeros(TOPIC)
            VIP = np.zeros(TOPIC)
#             PIP
        PIP = np.full(TOPIC, 0)
        for v in list(self.G.predecessors(user)):
            PIP = self.G.edges[(v, user)]['PIP']

        for f in feed:
            topic = f.topic
            influence_prob = TP[topic] * ((1-LAMBDA) * VIP[topic] + LAMBDA * np.full(TOPIC, PIP)[topic])

            if random.random() < influence_prob:
                print(influence_prob)
                return True, f
            else:
                return False, None

    """
        If user v is influenced by a message msg send from i at time step t, it will send a message at t+1 to AI platfrom which has the same topic with msg.
    """

    def take_action(self, user, message, iteration, influence='True'):
        if influence:
            topic = message.topic
            sendMsg = bean.message(topic, user, iteration)
            G.nodes[user]['sendList'].append(sendMsg)
            self.repository.append(sendMsg)
            print('influenced ' + str(G.nodes[user]))
        else:
            if random.random() < PROB_SEND_MSG:
                topic = np.argmax(G.nodes[user]['preference'])

    #  return the list of s^x
    def calculate_topic_distribution(self, arr):
        dict = {}
        for t in range(TOPIC):
            dict[t] = 0
        for msg in arr:
            topic = msg.topic
            value = dict.get(topic)
            dict[topic] = value + 1
        return list(dict.values())


    # change to txt
    def save_result(self, rows, exp):
        filename = time.strftime('%Y%m%d%H%M%S', time.localtime()) + '_' + exp + '.csv'
        with open('./result/' + filename, 'w', newline='') as csvfile:
            fieldnames = ['timestep', 'result']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for row in rows:
                writer.writerow({fieldnames[0]: row[0], fieldnames[1]: row[1]})

    # todo
    def plot_result(self):
        pass

# return a list in dict {timestep t: message list at timestep t}
def get_arr_by_time(messages,iter):
    msgDict = {}
    for m in messages:
        if msgDict.get(m.timestep) is None:
            msgDict[m.timestep] = []
            msgDict[m.timestep].append(m)
        else:
            msgDict[m.timestep] = msgDict.get(m.timestep)
            msgDict[m.timestep].append(m)
    if msgDict.get(iter) is None:
        return []
    else:
        return msgDict.get(iter)


if __name__ == '__main__':
    start = time.time()
    path = './dataset/' + DATASET[DATASET_INDEX] + '.txt'

    print(path[:-4] + '.G')

    G = pickle.load(open(path[:-4] + '_' + str(TOPIC) + '.G', 'rb'))

    print(nx.info(G))

    env = environment(G)
    env.influence_diffusion()
    # env.influence_diffusion_without_AI(mc=1000)
