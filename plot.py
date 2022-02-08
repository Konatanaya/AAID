import matplotlib.pyplot as plt
import numpy as np
import os

AI_settings = {
    'None': {
        'name': 'No recommendation algorithm',
        'color': 'orange'
    },
    'CB': {
        'name': 'Content-based Filtering',
        'color': 'royalblue'
    },
    'UC': {
        'name': 'User-based Collaborative Filtering',
        'color': 'darkgreen'
    }
}

k_settings = {
    '5': {
        'name': 'k=5',
        'color': 'orange'
    },
    '10': {
        'name': 'k=10',
        'color': 'royalblue'
    },
    '20': {
        'name': 'k=20',
        'color': 'darkgreen'
    },
    '40': {
        'name': 'k=40',
        'color': 'crimson'
    }
}

fontsize = 12
plt.style.use('ggplot')


def filter_bubble_line_chart(args):
    results = {}
    for AI in AI_settings.keys():
        path = './results/' + args.dataset + '_' + str(args.topic_num) + '_' + str(args.time_steps) + '_' + AI + '_' + str(args.k) + '.txt'
        print(path)
        if not os.path.exists(path):
            continue
        results[AI] = np.loadtxt(path)

    legend = []
    plt.figure()
    x = np.arange(0, args.time_steps, 1)
    for AI in AI_settings.keys():
        if AI not in results:
            continue
        value = results[AI].T
        plt.plot(x, value[1], ms=6, color=AI_settings[AI]['color'], mec='black')
        # plt.fill_between(x, value[0], value[2], color=AI_settings[AI]['color'], alpha=0.2)
        legend.append(AI_settings[AI]['name'])
    plt.legend(legend, fontsize=fontsize, loc='lower right', ncol=1, facecolor='white', fancybox=True, framealpha=0.4)
    # plt.xticks(np.arange(0, length + 1, 25), fontsize=12)
    # plt.yticks(np.arange(dataset_config['y_min'], dataset_config['y_max'] + 1, dataset_config['y_interval']), fontsize=12)
    #
    plt.xlabel('Time steps', fontsize=fontsize)
    plt.ylabel('Filter bubble', fontsize=fontsize)
    plt.title('Filter bubble in ' + args.dataset + ' when k='+str(args.k))
    # plt.grid()
    plt.savefig('./figures/fb_' + args.dataset + '.pdf', bbox_inches='tight', dpi=300)
    plt.show()


def filter_bubble_line_chart_diff_k(args):
    results = {}
    for k in k_settings.keys():
        path = './results/' + args.dataset + '_' + str(args.topic_num) + '_' + str(args.time_steps) + '_' + args.AI + '_' + str(k) + '.txt'
        print(path)
        if not os.path.exists(path):
            continue
        results[k] = np.loadtxt(path)

    legend = []
    plt.figure()
    x = np.arange(0, args.time_steps, 1)
    for k in k_settings.keys():
        if k not in results:
            continue
        value = results[k].T
        plt.plot(x, value[1], ms=6, color=k_settings[k]['color'], mec='black')
        # plt.fill_between(x, value[0], value[2], color=k_settings[k]['color'], alpha=0.2)
        legend.append(k_settings[k]['name'])
    plt.legend(legend, fontsize=fontsize, loc='lower right', ncol=1, facecolor='white', fancybox=True, framealpha=0.4)
    # plt.xticks(np.arange(0, length + 1, 25), fontsize=12)
    # plt.yticks(np.arange(dataset_config['y_min'], dataset_config['y_max'] + 1, dataset_config['y_interval']), fontsize=12)
    #
    plt.xlabel('Time steps', fontsize=fontsize)
    plt.ylabel('Filter bubble', fontsize=fontsize)
    plt.title('Filter bubble in ' + args.dataset + ' with CB deployed')
    # plt.grid()
    plt.savefig('./figures/fb_' + args.dataset + '_k.pdf', bbox_inches='tight', dpi=300)
    plt.show()
