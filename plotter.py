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