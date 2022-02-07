import argparse
import bean
import env
import random
import numpy as np
from env import environment
from datetime import datetime
import networkx as nx
import os


# Hyper parameters
def init_parser():
    parser = argparse.ArgumentParser(description="Hyper Parameters")
    parser.add_argument('--dataset', type=str, default='twitter')
    parser.add_argument('--topic_num', type=int, default=5)
    parser.add_argument('--init_dataset', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=321)
    parser.add_argument('--time_steps', type=int, default=10)
    parser.add_argument('--AI', type=str, default="None")
    parser.add_argument('--LAMBDA', type=float, default=0.5)
    parser.add_argument('--prob_send_msg', type=float, default=0.2)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--worker_num', type=int, default=4)
    args = parser.parse_args()

    parser.print_help()
    return args


if __name__ == '__main__':
    args = init_parser()
    print(args)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.init_dataset:
        bean.init_dataset(args.dataset, args.topic_num)

    G = bean.load_network(args.dataset, args.topic_num)
    env = env.environment(G, args)
    env.simulation()
