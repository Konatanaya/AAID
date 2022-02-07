import matplotlib.pyplot as plt
import numpy as np

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

fontsize = 12
plt.style.use('ggplot')


def filter_bubble_line_chart(args):
    results = {}
    for AI in AI_settings.keys():
        path = './results/' + args.dataset + '_' + str(args.k) + '_' + str(args.time_steps) + '_' + AI + '.txt'
        print(path)
        results[AI] = np.loadtxt(path)

    legend = []
    plt.figure()
    x = np.arange(0, args.time_steps, 1)
    for AI in AI_settings.keys():
        value = results[AI].T
        plt.plot(x, value[1], ms=6, color=AI_settings[AI]['color'], mec='black')
        plt.fill_between(x,value[0],value[2],color=AI_settings[AI]['color'], alpha=0.2)
        legend.append(AI_settings[AI]['name'])
    plt.legend(legend, fontsize=fontsize, loc='lower right', ncol=1, facecolor='white', fancybox=True, framealpha=0.4)
    # plt.xticks(np.arange(0, length + 1, 25), fontsize=12)
    # plt.yticks(np.arange(dataset_config['y_min'], dataset_config['y_max'] + 1, dataset_config['y_interval']), fontsize=12)
    #
    plt.xlabel('Time steps', fontsize=fontsize)
    plt.ylabel('Filter bubble', fontsize=fontsize)
    # plt.grid()
    plt.savefig('./figures/fb_' + args.dataset + '.pdf', bbox_inches='tight', dpi=300)
    plt.show()
