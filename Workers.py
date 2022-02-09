import numpy as np
import threading
import multiprocessing as mp

worker = []


class CB_Thread(threading.Thread):
    def __init__(self, thread_lock, G, topic_correlation, msg_list, index_id, index_range, time_step, recommendation, k):
        super(CB_Thread, self).__init__()
        self.G = G
        self.topic_correlation = topic_correlation
        self.msg_list = msg_list
        self.index_id = index_id
        self.index_range = index_range
        self.time_step = time_step
        self.recommendation = recommendation
        self.k = k
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
            # score = np.array([score_[index][msg_index] * self.G.nodes[user]['preference'][msg.topic] for msg_index, msg in enumerate(self.msg_list)])
            user_msg_index = [ind for ind, msg in enumerate(self.msg_list) if msg.sender == user]
            if len(user_msg_index) > 0:
                score[index][user_msg_index] = 0.
            msg_index = np.argpartition(score[index], -self.k)[-self.k:]
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
    def __init__(self, thread_lock, G, topic_correlation, msg_list, index_id, index_range, time_step, recommendation, k, similarity_matrix=None):
        super(UC_Thread, self).__init__()
        self.G = G
        self.topic_correlation = topic_correlation
        self.msg_list = msg_list
        self.index_id = index_id
        self.index_range = index_range
        self.time_step = time_step
        self.recommendation = recommendation
        self.k = k
        self.similarity_matrix = similarity_matrix

        self.lock = thread_lock

    def run(self):
        index_range = self.index_range
        time_step = self.time_step
        topic_num = len(self.topic_correlation[0])
        user_ids = [self.index_id[i] for i in range(index_range[0], index_range[1])]
        id_index = {ID: index for index, ID in self.index_id.items()}

        users_pre_matrix = np.array([self.G.nodes[user]['preference'] for user in user_ids]).reshape((-1, topic_num))
        all_users_pre_matrix = np.array([self.G.nodes[user]['preference'] for user in self.G.nodes()]).reshape((-1, topic_num))
        similarity_matrix = users_pre_matrix.dot(all_users_pre_matrix.T) / (np.linalg.norm(users_pre_matrix, axis=1).reshape(-1, 1) * np.linalg.norm(all_users_pre_matrix, axis=1))
        similarity_matrix[np.isneginf(similarity_matrix)] = 0

        recommendation_list = []
        for index, user in enumerate(user_ids):
            score = np.array([similarity_matrix[index, id_index[msg.sender]] * self.G.nodes[user]['preference'][msg.topic] for msg in self.msg_list])
            user_msg_index = [ind for ind, msg in enumerate(self.msg_list) if msg.sender == user]
            if len(user_msg_index) > 0:
                score[user_msg_index] = 0.

            msg_index = np.argpartition(score, -self.k)[-self.k:]
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


class CB_Worker(mp.Process):
    def __init__(self, inQ, outQ, G, topic_correlation, msg_list, index_id):
        super(CB_Worker, self).__init__(target=self.start)
        self.inQ = inQ
        self.outQ = outQ
        self.G = G
        self.topic_correlation = topic_correlation
        self.msg_list = msg_list
        self.index_id = index_id

    def run(self):
        while True:
            block = self.inQ.get()
            index_range = block[0]
            time_step = block[1]
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
            self.outQ.put(recommendation_list)


def create_CB_workers(num, G, topic_correlation, msg_list, index_id):
    worker = []
    for i in range(num):
        worker.append(CB_Worker(mp.Queue(), mp.Queue(), G, topic_correlation, msg_list, index_id))
        worker[i].start()
    return worker


def finish_worker(worker):
    for w in worker:
        w.terminate()
    worker = []


if __name__ == '__main__':
    a = np.array([[2,3],
                  [3,4],
                  [1,3.5]])
    b = np.array([1,1])
    print(a-b)
    print(np.linalg.norm(a-b, axis=1))
    # b = np.array([[1,1,1.2], [1,2,3], [1,3,2]])

