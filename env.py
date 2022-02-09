import math
import numpy as np
import datetime
from matplotlib import pyplot as plt
import os
import random
import quantification
from message import Message
import Workers
from Workers import CB_Thread, UC_Thread
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

        self.multi_thread = args.multi_thread
        self.topic_correlation = self.init_topic_correlation()

        self.recommendation = None  # time_step:[msg]
        self.sendList_total = None
        self.index_id = None
        self.sendCount = None
        self.receiveCount = None
        self.recommendation_score_index = None

        random.seed(args.seed)
        np.random.seed(args.seed)

        # result
        self.results = []
        self.init()

    def init(self):
        self.recommendation = {t: [] for t in range(self.time_steps)}  # time_step:[msg]
        self.sendList_total = {t: [] for t in range(self.time_steps)}  # time_step:[msg]

        self.index_id = {index: user for index, user in enumerate(self.G.nodes())}

        # self.filter_bubble = {user: [] for user in self.G.nodes()}

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
            ec = []
            for user in self.G.nodes():
                filter_bubble = quantification.quantify_filter_bubble(self.G, user, self.topic_num)
                fb.append(filter_bubble)
                echo_chamber = quantification.quantify_echo_chamber(self.G, user, time_step, self.topic_num)
                ec.append(echo_chamber)
                self.user_takes_action(user, time_step)
            execution_time = (datetime.datetime.now() - start).seconds
            result = [np.average(fb), np.average(ec)]
            # result = [np.average(fb), quantification.quantify_echo_chamber(self.G)]
            self.results.append(result)
            print("Time step: %d | Execution time: %ds | Avg. FB: %f | Avg. EC: %f" % (time_step, execution_time, result[0], result[1]))
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
            self.G.nodes[user]['sendCount']['total'] = np.ones(self.topic_num)
        if time_step not in self.G.nodes[user]['sendCount']:
            self.G.nodes[user]['sendCount'][time_step] = np.zeros(self.topic_num)
        if time_step not in self.G.nodes[user]['receiveCount']:
            self.G.nodes[user]['receiveCount'][time_step] = np.zeros(self.topic_num)

        # The user agent would choose its favorite topic at time step 0
        topics = []
        if time_step == 0:
            topics = np.random.choice(np.arange(self.topic_num), p=self.G.nodes[user]['preference'], size=1)
            # topics = [np.argmax(self.G.nodes[user]['preference'])]
        else:
            # Check if the user agent is influenced by its in-neighbors
            topics = self.calculate_influence(user, time_step)
            # if random.random() < 1:
            # topics += [np.argmax(self.G.nodes[user]['preference'])]
            topics += list(np.random.choice(np.arange(self.topic_num), p=self.G.nodes[user]['preference'], size=1))


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
            self.G.nodes[user]['preference'] = self.G.nodes[user]['sendCount']['total'] / np.sum(self.G.nodes[user]['sendCount']['total'])


    def calculate_sum_of_dict(self, dict):
        return np.sum([np.sum(v) for k, v in dict.items()])

    def init_recommendation(self, time_step, alg='user-cf'):
        # multi process
        if alg == "None":
            return
        index_range = [math.floor(len(self.G.nodes()) / self.worker_num) * i for i in range(self.worker_num)] + [len(self.G.nodes())]
        msg_list = self.sendList_total[time_step - 1]
        random.shuffle(msg_list)
        # if self.multi_thread:
        #     if alg == 'CB':
        #         # multi-threads
        #         threadLock = threading.Lock()
        #         threads = []
        #         for worker_id in range(self.worker_num):
        #             worker_range = index_range[worker_id: worker_id + 2] if worker_id < self.worker_num - 1 else index_range[worker_id:]
        #             t = CB_Thread(threadLock, self.G, self.topic_correlation, msg_list, self.index_id, worker_range, time_step, self.recommendation, self.k)
        #             t.start()
        #             threads.append(t)
        #         Workers.finish_threads(threads)
        #     if alg == 'UC':
        #         # all_users_pre = [self.G.nodes[user]['preference'] for user in self.G.nodes()]
        #         # users_pre_matrix = np.concatenate(all_users_pre).reshape((-1, self.topic_num))
        #         # similarity_matrix = users_pre_matrix.dot(users_pre_matrix.T) / (np.linalg.norm(users_pre_matrix, axis=1).reshape(-1, 1) * np.linalg.norm(users_pre_matrix, axis=1))
        #         # similarity_matrix[np.isneginf(similarity_matrix)] = 0
        #         # dim = len(self.G.nodes())
        #         # # similarity_matrix = similarity_matrix
        #         # similarity_matrix[range(dim), range(dim)] = 0.
        #
        #         threadLock = threading.Lock()
        #         threads = []
        #         for worker_id in range(self.worker_num):
        #             worker_range = index_range[worker_id: worker_id + 2] if worker_id < self.worker_num - 1 else index_range[worker_id:]
        #             t = UC_Thread(threadLock, self.G, self.topic_correlation, msg_list, self.index_id, worker_range, time_step, self.recommendation, self.k)
        #             t.start()
        #             threads.append(t)
        #         Workers.finish_threads(threads)
        # else:
            # multi-processes
        if alg == 'CB':
            workers = Workers.create_CB_workers(self.worker_num, self.G, self.topic_correlation, msg_list, self.k, self.index_id)
            # allocate workers
            for worker_id in range(self.worker_num):
                worker_range = index_range[worker_id: worker_id + 2] if worker_id < self.worker_num - 1 else index_range[worker_id:]
                workers[worker_id].inQ.put((worker_range, time_step))
            for w in workers:
                recommendation_list = w.outQ.get()
                self.recommendation[time_step] += recommendation_list
            Workers.finish_worker(workers)
        elif alg == 'UC':
            workers = Workers.create_UC_workers(self.worker_num, self.G, self.topic_correlation, msg_list, self.k, self.index_id)
            # allocate workers
            for worker_id in range(self.worker_num):
                worker_range = index_range[worker_id: worker_id + 2] if worker_id < self.worker_num - 1 else index_range[worker_id:]
                workers[worker_id].inQ.put((worker_range, time_step))
            for w in workers:
                recommendation_list = w.outQ.get()
                self.recommendation[time_step] += recommendation_list
            Workers.finish_worker(workers)
        else:
            return

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
        # topic_list = [msg.topic for msg in self.G.nodes[user]['receiveList'][time_step - 1] if rand < IP[msg.topic]]
        for index, ip in enumerate(IP):
            if random.random() < ip:
                topic_list.append(index)
        return topic_list

    # change to txt
    def save_result(self):
        if not os.path.exists("./results"):
            os.makedirs('./results/')
        filename = self.dataset + '_' + str(self.topic_num) + '_' + str(self.time_steps) + '_' + self.AI + '_' + str(self.k) + '.txt'
        with open('./results/' + filename, 'w') as f:
            for l in self.results:
                for v in l:
                    f.write(str(v) + ' ')
                f.write('\n')
