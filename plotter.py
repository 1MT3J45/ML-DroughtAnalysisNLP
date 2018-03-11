import matplotlib.pyplot as plt
import numpy as np


def bars(predictions, plt_name="graph"):
    prediction = predictions
    plt.style.use('ggplot')


    fig, axes = plt.subplots(ncols=1, nrows=1)
    ax3= axes

    x = np.arange(3)
    y1, y2, y3 = prediction
    width = 0.20
    plt.title('%s\n Polarity Score for POS, NEG, NEU' % plt_name)
    plt.xlabel('Parameters')
    plt.ylabel('Score')
    ax3.bar(x, y1, width, label="Correct Positives")
    ax3.bar(x + width, y2, width, color=list(plt.rcParams['axes.prop_cycle'])[2]['color'], label="Correct Negatives")
    ax3.bar(x + width + width, y3, width, color=list(plt.rcParams['axes.prop_cycle'])[3]['color'], label="Correct Neutrals")
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(['Positive', 'Negative', 'Neutral'])
    plt.legend()
    plt.show()


def biplt(groundTruth, predictedValues, plt_name='<name>'):
    gt = groundTruth
    pr = predictedValues
    x = np.arange(3)

    y1, y2 = gt.values, pr.values
    fig, axes = plt.subplots(ncols=1, nrows=1)

    width = 0.20
    plt.title('Accuracy with \n %s' % plt_name)
    plt.xlabel('Parameters')
    plt.ylabel('Score')
    axes.bar(x, y1, width, label="Ground Truth")
    axes.bar(x + width, y2, width, color=list(plt.rcParams['axes.prop_cycle'])[2]['color'], label="Predicted")
    axes.set_xticks(x + width/2)
    axes.set_xticklabels(['Positive', 'Negative', 'Neutral'])
    plt.legend()
    plt.show()
    # y, z, k = prediction
    #
    # ax = plt.subplot(111)
    # ax.bar(x - 0.2, y, width=0.2, color='b', align='center')
    # ax.bar(x, z, width=0.2, color='g', align='center')
    # ax.bar(x + 0.2, k, width=0.2, color='r', align='center')
    # ax.xaxis_date()
    #
    # plt.show()


def uslplt(groundTruth, predictedValues, plt_name="<name>"):
    # data to plot
    n_groups = 3
    gt = groundTruth.values
    pr = predictedValues.values

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, gt, bar_width,
                     alpha=opacity,
                     color='b',
                     label='Frank')

    rects2 = plt.bar(index + bar_width, pr, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Guido')

    plt.xlabel('Person')
    plt.ylabel('Scores')
    plt.title('%s\nScores by person'%plt_name)
    plt.xticks(index + bar_width, ('A', 'B', 'C'))
    plt.legend()

    plt.tight_layout()
    plt.show()


def stackplotter(HighlyNEG, ModeratelyNEG, NEG, HighlyPOS, ModeratelyPOS, POS, text):
    H_NG, M_NG, NG, H_PS, M_PS, PS = HighlyNEG, ModeratelyNEG, NEG, HighlyPOS, ModeratelyPOS, POS
    Polarities = [H_NG, M_NG, NG, H_PS, M_PS, PS]
    N = len(Polarities)  # Number of Bars

    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, Polarities, width)

    plt.xlabel('Polarities')
    plt.ylabel('Scores')
    plt.title(text)
    plt.xticks(ind, (
    'Highly Negative', 'Moderately Negative', 'Negative', 'Highly Positive', 'Moderately Positive', 'Positive'))
    # plt.yticks(np.arange(0, 200, 10))
    # plt.legend(p1[0], 'Polarity')

    plt.show()

def stackplotter3d(HighlyNEG, ModeratelyNEG, NEG, HighlyPOS, ModeratelyPOS, POS, text):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    H_NG, M_NG, NG, H_PS, M_PS, PS = HighlyNEG, ModeratelyNEG, NEG, HighlyPOS, ModeratelyPOS, POS
    Polarities = [H_NG, M_NG, NG, H_PS, M_PS, PS]
    # N = len(Polarities)  # Number of Bars

    # ind = np.arange(N)  # the x locations for the groups
    # width = 0.35  # the width of the bars: can also be len(x) sequence

    #p1 = plt.bar(ind, Polarities, width)

    ax.set_xlabel('Polarities')
    ax.set_ylabel('Scores')
    ax.set_zlabel('z')
    #ax.title("TESTING")

    ax.set_xlim3d(0, 14)
    ax.set_ylim3d(0, 4)

    #ax.xticks(ind, ('Highly Negative', 'Moderately Negative', 'Negative', 'Highly Positive', 'Moderately Positive', 'Positive'))

    xpos = [2, 4, 6, 8, 10, 12]
    ypos = [1, 1, 1, 1, 1, 1]
    zpos = Polarities

    dx = np.ones(6)
    dy = np.ones(6)
    dz = [np.random.random(6) for i in range(4)]# the heights of the 4 bar sets

    _zpos = zpos  # the starting zpos for each bar

    for i in range(len(dz)):
        ax.bar3d(xpos, ypos, _zpos, dx, dy, dz[i], color='red')
        _zpos += dz[i]  # add the height of each bar to know where to start the next

    plt.gca().invert_xaxis()

    # plt.yticks(np.arange(0, 200, 10))
    # plt.legend(p1[0], 'Polarity')

    plt.show()

    # # ------------------------------------------------------------
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import numpy as np
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
#
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# ax.set_xlim3d(0, 10)
# ax.set_ylim3d(0, 10)
#
# xpos = [2, 5, 8, 2, 5, 8, 2, 5, 8]
# ypos = [1, 1, 1, 5, 5, 5, 9, 9, 9]
# zpos = np.zeros(9)
#
# dx = np.ones(9)
# dy = np.ones(9)
# dz = [np.random.random(9) for i in range(4)]  # the heights of the 4 bar sets
#
# _zpos = zpos  # the starting zpos for each bar
# colors = ['r', 'b', 'g', 'y', 'r', 'b', 'y']
# for i in range(4):
#     ax.bar3d(xpos, ypos, _zpos, dx, dy, dz[i], color=colors[i])
#     _zpos += dz[i]  # add the height of each bar to know where to start the next
#
# plt.gca().invert_xaxis()
# plt.show()
