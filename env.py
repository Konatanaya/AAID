import math
import numpy as np
import datetime
from matplotlib import pyplot as plt
import os
import random
import quantification
from message import Message
import Workers
from Workers import CB_Thread
import threading

DATASET = ['facebook', 'ciao_clean', 'random']
TOPIC = 5
DATASET_INDEX = 0
PROB_SEND_MSG = 0.2
ITERATION = 100
LAMBDA = 0.5
FEED = 1


class environment:
    def __init__(self, G, args):
        self.G = G
        self.topic_num = args.topic_num
        self.LAMBDA = args.LAMBDA
        self.k = args.k
        self.prob_send_msg = args.prob_send_msg
        self.AI = args.AI
        self.dataset = args.dataset
        self.time_steps = args.time_steps
        self.worker_num = args.worker_num

        self.topic_correlation = self.init_topic_correlation()

        self.recommendation = None  # time_step:[msg]
        self.sendList_total = None
        self.index_id = None
        self.sendCount = None
        self.receiveCount = None
        self.recommendation_score_index = None

        # result
        self.filter_bubble = None
        self.init()

    def init(self):
        self.recommendation = {t: [] for t in range(self.time_steps)}  # time_step:[msg]
        self.sendList_total = {t: [] for t in range(self.time_steps)}  # time_step:[msg]

        self.index_id = {index: user for index, user in enumerate(self.G.nodes())}

        # self.filter_bubble = {user: [] for user in self.G.nodes()}
        self.filter_bubble = []

    def init_topic_correlation(self):
        dim = self.topic_num
        Z = np.random.random(size=(dim, dim))
        Z = np.triu(Z)
        Z += Z.T
        Z[range(dim), range(dim)] = 1.
        return Z

    def simulation(self):
        for time_step in range(self.time_steps):
            start = datetime.datetime.now()
            if time_step > 0:
                self.init_recommendation(time_step, alg=self.AI)
            fb = []
            for user in self.G.nodes():
                filter_bubble = quantification.quantify_filter_bubble(self.G, user, self.topic_num)
                fb.append(filter_bubble)
                self.user_takes_action(user, time_step)
            execution_time = (datetime.datetime.now() - start).seconds
            self.filter_bubble.append([np.min(fb), np.average(fb), np.max(fb)])
            print("Time step: %d | Execution time: %ds | Avg. FB: %f" % (time_step, execution_time, self.filter_bubble[-1][1]))
        self.save_result()
        # self.plot_result(time_steps)

    def data_check(self, time_step):
        for user in self.G.nodes():
            # if self.G.predecessors(user):
            print(np.sum(self.G.nodes[user]['receiveCount'][time_step], dtype=int) == len(list(self.G.predecessors(user))))
            # sendList = self.G.nodes[user]['sendList']
            # receiveList = self.G.nodes[user]['receiveList']
            # if time_step in sendList:
            #     print("send:" + str(len(self.G.nodes[user]['sendList'][time_step])), end=" ")
            # if time_step in receiveList:
            #     print("receive:" + str(len(self.G.nodes[user]['receiveList'][time_step])), end=" ")
            # print(len(list(self.G.predecessors(user))))

    def user_takes_action(self, user, time_step):
        if time_step not in self.G.nodes[user]['receiveList']:  # initialize the receiving repository
            self.G.nodes[user]['receiveList'][time_step] = []
        if time_step not in self.G.nodes[user]['sendList']:  # initialize the sending repository
            self.G.nodes[user]['sendList'][time_step] = []
        if 'total' not in self.G.nodes[user]['sendCount']:
            self.G.nodes[user]['sendCount']['total'] = np.zeros(self.topic_num)
        if time_step not in self.G.nodes[user]['sendCount']:
            self.G.nodes[user]['sendCount'][time_step] = np.zeros(self.topic_num)
        if time_step not in self.G.nodes[user]['receiveCount']:
            self.G.nodes[user]['receiveCount'][time_step] = np.zeros(self.topic_num)

        # The user agent would choose its favorite topic at time step 0
        topics = []
        if time_step == 0:
            topics = [np.argmax(self.G.nodes[user]['preference'])]
        else:
            # Check if the user agent is influenced by its in-neighbors
            topics = self.calculate_influence(user, time_step)
            # if random.random() < 1:
            topics += [np.argmax(self.G.nodes[user]['preference'])]

        # create the msg
        msgs = [Message(topic, user, time_step) for topic in topics]
        self.G.nodes[user]['sendList'][time_step] = msgs
        self.sendList_total[time_step] += msgs
        for msg in msgs:
            self.G.nodes[user]['sendCount'][time_step][msg.topic] += 1
            self.G.nodes[user]['sendCount']['total'][msg.topic] += 1

        for neighbor in self.G.successors(user):
            receive_list = self.G.nodes[neighbor]['receiveList']
            if time_step not in receive_list:  # initialize the receiving repository
                receive_list[time_step] = []
            receive_list[time_step] += msgs
            if time_step not in self.G.nodes[neighbor]['receiveCount']:
                self.G.nodes[neighbor]['receiveCount'][time_step] = np.zeros(self.topic_num)
            for msg in msgs:
                self.G.nodes[neighbor]['receiveCount'][time_step][msg.topic] += 1
        if time_step > 0:
            self.update_preference(user, time_step)

    def update_preference(self, user, time_step):
        for topic in range(self.topic_num):
            r_t_ = self.G.nodes[user]['preference'][topic]
            total_count_t = self.G.nodes[user]['sendCount'][time_step][topic]
            total_count_t_1 = self.G.nodes[user]['sendCount'][time_step - 1][topic]
            count_diff = np.abs(total_count_t - total_count_t_1)
            count_sum = total_count_t + total_count_t_1
            if count_sum == 0:
                count_sum = 1
            if r_t_ < 0.5:
                if count_diff > 0:
                    r_t = r_t_ * (1 - count_diff / count_sum)
                else:
                    r_t = r_t_ * (1 + count_diff / count_sum)
            else:
                if count_diff > 0:
                    r_t = r_t_ * (1 + count_diff / count_sum)
                    if r_t > 1: r_t = 1
                else:
                    r_t = r_t_ * (1 - count_diff / count_sum)
            self.G.nodes[user]['preference'][topic] = r_t

    def calculate_sum_of_dict(self, dict):
        return np.sum([np.sum(v) for k, v in dict.items()])

    def init_recommendation(self, time_step, alg='user-cf'):
        # multi process
        if alg == "None":
            return
        index_range = [math.floor(len(self.G.nodes()) / self.worker_num) * i for i in range(self.worker_num)] + [len(self.G.nodes())]
        msg_list = self.sendList_total[time_step - 1]
        random.shuffle(msg_list)
        if alg == 'CB':
            # multi-threads
            threadLock = threading.Lock()
            threads = []
            for worker_id in range(self.worker_num):
                worker_range = index_range[worker_id: worker_id + 2] if worker_id < self.worker_num - 1 else index_range[worker_id:]
                t = CB_Thread(threadLock, self.G, self.topic_correlation, msg_list, self.index_id, worker_range, time_step, self.recommendation)
                t.start()
                threads.append(t)
            Workers.finish_threads(threads)
        if alg == 'UC':
            threadLock = threading.Lock()
            threads = []
            for worker_id in range(self.worker_num):
                worker_range = index_range[worker_id: worker_id + 2] if worker_id < self.worker_num - 1 else index_range[worker_id:]
                t = CB_Thread(threadLock, self.G, self.topic_correlation, msg_list, self.index_id, worker_range, time_step, self.recommendation)
                t.start()
                threads.append(t)
            Workers.finish_threads(threads)

    # def generate_recommendation(self, user, msg_list, time_step):
    # print(user)

    # for index, user in enumerate(self.G.nodes()):
    #     msg_index = np.argpartition(score[index], -self.topic_num)[-self.topic_num:]
    #
    #     # pass

    # correlation_std = (sum_count.dot((correlation_matrix - correlation_avg) ** 2) / np.sum(sum_count)) ** 1 / 2
    # sim_dict = {msg:sim for msg,sim in zip(msg_list, correlation_std)}

    # sorted_sim_pairs = sorted(sim_dict.items(), key=lambda v: v[1], reverse=True)
    # recommendation_list = [sorted_sim_pairs[index][0] for index in range(self.k)]

    # for msg in self.sendList_total[time_step - 1]:
    #     if msg.sender == user:
    #         continue
    #     else:
    #         # user_sendlist = [send[0] for t, send in self.G.nodes[msg.sender]['sendList'].items()]
    #         if alg == "user-cf":
    #             # focal_user_sendlist_set = set(focal_user_sendlist)
    #             # user_sendlist_set = set(user_sendlist)
    #             user_count = np.sum(self.G.nodes[msg.sender]['sendCount'][time_step - 1])
    #             topic_count = np.sum(self.G.nodes[msg.sender]['sendCount'][time_step - 1])
    #             focal_user_topic_count = np.sum(self.G.nodes[user]['sendCount'][time_step - 1])
    #             # jaccard similarity
    #             similarity = self.preference_similarity(self.G.nodes[user]['preference'], self.G.nodes[msg.sender]['preference']) * \
    #                          (focal_user_topic_count + topic_count) / (user_count + focal_user_count)
    #             # similarity = len((focal_user_sendlist_set & user_sendlist_set)) / len((focal_user_sendlist_set | user_sendlist_set))
    #         elif alg == "content-based":
    #             correlation_vec = np.std(np.array([self.topic_correlation[msg.topic, m.topic] for m in focal_user_sendlist]))
    #             similarity = correlation_vec
    #         sim_dict[msg] = similarity
    # sorted_sim_pairs = sorted(sim_dict.items(), key=lambda v: v[1], reverse=True)
    # recommendation_list = [sorted_sim_pairs[index][0] for index in range(self.k)]
    # self.G.nodes[user]['receiveList'][time_step - 1] += recommendation_list
    # for msg in recommendation_list:
    #     self.G.nodes[user]['receiveCount'][time_step - 1][msg.topic] += 1
    # self.recommendation[time_step] += recommendation_list

    # IP = lambda * PIP + (1-lambda) * VIP
    def calculate_IP(self, PIP, VIP):
        if PIP == 0.:
            LAMBDA = 0
        elif VIP == 0.:
            LAMBDA = 1
        else:
            LAMBDA = self.LAMBDA
        IP = LAMBDA * PIP + (1 - LAMBDA) * VIP
        return IP

    def preference_similarity(self, vec1, vec2):
        sim = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return 0.5 + 0.5 * sim

    def calculate_influence(self, user, time_step):
        PIP_vec = np.zeros(self.topic_num, )
        VIP_vec = np.ones((self.topic_num,))

        user_send_count = self.G.nodes[user]['sendCount'][time_step - 1]
        user_preference = np.array(self.G.nodes[user]['preference'])
        in_neighbors = list(self.G.predecessors(user))
        rho_sum = np.sum(self.topic_correlation, axis=0)

        # Calculate PIP
        if len(in_neighbors) > 0:
            count_sum_arrays = [user_send_count + self.G.nodes[neighbor]['sendCount'][time_step - 1] for neighbor in in_neighbors]
            send_count_matrix = np.concatenate(count_sum_arrays).reshape((-1, self.topic_num))
            matrix_sum = 1 / np.sum(send_count_matrix, axis=1).reshape((-1, 1))
            C = np.dot(matrix_sum.T, send_count_matrix).flatten()
            # \rho_sum * r_j / (|T| * |N|)
            coefficient = rho_sum * user_preference / (self.topic_num * len(in_neighbors))
            PIP_vec = (coefficient * C).flatten()
        # print(PIP_vec)

        # Calculate VIP
        user_send_set = set(self.G.nodes[user]['sendList'][time_step - 1])
        user_receive_set = set(self.G.nodes[user]['receiveList'][time_step - 1])
        user_all_msg_set = user_send_set | user_receive_set
        recommendation_list = list(set(self.recommendation[time_step]) & user_receive_set)
        recommendation_users = list(set([msg.sender for msg in recommendation_list]))
        users_list = recommendation_users

        if len(users_list) > 0:
            VIP_matrix_row = []
            for u in users_list:
                u_send_set = set(self.G.nodes[u]['sendList'][time_step - 1])
                u_receive_set = set(self.G.nodes[u]['receiveList'][time_step - 1])
                u_all_msg_set = u_send_set | u_receive_set
                all_msg_list = [msg.topic for msg in list(user_all_msg_set | u_all_msg_set)]
                topic_msg_union_num = [all_msg_list.count(topic) if all_msg_list.count(topic) != 0 else 1 for topic in range(self.topic_num)]
                topic_msg_union_num = 1 / np.array(topic_msg_union_num)
                msg_intersection = list(u_send_set & user_receive_set)
                temp = np.zeros(self.topic_num)
                for msg in msg_intersection:
                    temp[msg.topic] += 1
                temp = temp * topic_msg_union_num
                VIP_matrix_row.append(temp)
            VIP_matrix = np.concatenate(VIP_matrix_row).reshape((-1, self.topic_num))
            sum_of_topic = np.sum(VIP_matrix, axis=0)
            VIP_vec = sum_of_topic * rho_sum / (self.topic_num * len(users_list))
        # print(VIP_vec)
        IP = [self.calculate_IP(v[0], v[1]) for index, v in enumerate(zip(PIP_vec, VIP_vec))]
        topic_list = []
        # rand = random.random()
        for index, ip in enumerate(IP):
            if random.random() < ip:
                topic_list.append(index)
        return topic_list

    # change to txt
    def save_result(self):
        if not os.path.exists("./results"):
            os.makedirs('./results/')
        filename = self.dataset + '_' + str(self.topic_num) + '_' + str(self.time_steps) + '_' + self.AI + '.txt'
        with open('./results/' + filename, 'w', newline='') as f:
            for v in self.filter_bubble:
                f.write(str(v) + ' ')

    # todo


def plot_result():
    filename = './result/twitter_5_1000_CB.txt'
    data1 = np.loadtxt('./result/twitter_5_1000_CB.txt', delimiter=',')
    data1 = np.average(data1, axis=0)
    data2 = np.loadtxt('./result/twitter_5_1000_None.txt', delimiter=',')
    data2 = np.average(data2, axis=0)

    plt.figure()
    plt.plot(data1)
    plt.plot(data2)
    plt.show()
    # with open(filename, 'r') as f:
    #     for line in f.readlines():
    #         line = line.strip().split(',')
    #         print(line)


if __name__ == '__main__':
    plot_result()
