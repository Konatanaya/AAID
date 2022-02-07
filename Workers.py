import numpy as np
import threading

worker = []


class CB_Thread(threading.Thread):
    def __init__(self, thread_lock, G, topic_correlation, msg_list, index_id, index_range, time_step, recommendation):
        super(CB_Thread, self).__init__()
        self.G = G
        self.topic_correlation = topic_correlation
        self.msg_list = msg_list
        self.index_id = index_id
        self.index_range = index_range
        self.time_step = time_step
        self.recommendation = recommendation
        self.lock = thread_lock

    def run(self):
        index_range = self.index_range
        time_step = self.time_step
        # print(index_range)
        topic_num = len(self.topic_correlation[0])
        user_ids = [self.index_id[i] for i in range(index_range[0], index_range[1])]
        all_users_count = [self.G.nodes[user]['sendCount']['total'] for user in user_ids]
        users_count_matrix = np.concatenate(all_users_count).reshape((-1, topic_num))
        correlation_matrix = np.array([self.topic_correlation[msg.topic] for msg in self.msg_list]).reshape((-1, topic_num)).T
        score = users_count_matrix.dot(correlation_matrix)

        recommendation_list = []
        for index, user in enumerate(user_ids):
            msg_index = np.argpartition(score[index], -topic_num)[-topic_num:]
            # self.recommendation_score_index = {user:  for index, user in enumerate(self.G.nodes())}
            recommendation_list = [self.msg_list[ind] for ind in msg_index]
            self.G.nodes[user]['receiveList'][time_step - 1] += recommendation_list
            for msg in recommendation_list:
                self.G.nodes[user]['receiveCount'][time_step - 1][msg.topic] += 1
            # self.recommendation[time_step] += recommendation_list
        self.lock.acquire()
        self.recommendation[time_step] = recommendation_list
        self.lock.release()


class UC_Thread(threading.Thread):
    def __init__(self, thread_lock, G, topic_correlation, msg_list, index_id, index_range, time_step, recommendation):
        super(UC_Thread, self).__init__()
        self.G = G
        self.topic_correlation = topic_correlation
        self.msg_list = msg_list
        self.index_id = index_id
        self.index_range = index_range
        self.time_step = time_step
        self.recommendation = recommendation
        self.lock = thread_lock

    def run(self):
        index_range = self.index_range
        time_step = self.time_step
        # print(index_range)
        topic_num = len(self.topic_correlation[0])
        user_ids = [self.index_id[i] for i in range(index_range[0], index_range[1])]
        all_users_count = [self.G.nodes[user]['sendCount']['total'] for user in user_ids]
        users_count_matrix = np.concatenate(all_users_count).reshape((-1, topic_num))
        correlation_matrix = np.array([self.topic_correlation[msg.topic] for msg in self.msg_list]).reshape((-1, topic_num)).T
        score = users_count_matrix.dot(correlation_matrix)

        recommendation_list = []
        for index, user in enumerate(user_ids):
            msg_index = np.argpartition(score[index], -topic_num)[-topic_num:]
            # self.recommendation_score_index = {user:  for index, user in enumerate(self.G.nodes())}
            recommendation_list = [self.msg_list[ind] for ind in msg_index]
            self.G.nodes[user]['receiveList'][time_step - 1] += recommendation_list
            for msg in recommendation_list:
                self.G.nodes[user]['receiveCount'][time_step - 1][msg.topic] += 1
            # self.recommendation[time_step] += recommendation_list
        self.lock.acquire()
        self.recommendation[time_step] = recommendation_list
        self.lock.release()

def finish_threads(threads):
    for t in threads:
        t.join()
    threads = []
