import matplotlib.pyplot as plt
import numpy as np


def bars(predictions):
    prediction = predictions
    plt.style.use('ggplot')

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    fig, axes = plt.subplots(ncols=1, nrows=1)
    ax3= axes

    x = np.arange(3)
    y1, y2, y3 = prediction#25, 20# np.random.randint(1, 25, size=(2, 3))
    width = 0.20
    plt.title('Polarity Score for POS, NEG, NEU')
    plt.xlabel('Parameters')
    plt.ylabel('Score')
    plt.legend()
    ax3.bar(x, y1, width)
    ax3.bar(x + width, y2, width, color=list(plt.rcParams['axes.prop_cycle'])[2]['color'])
    ax3.bar(x + width + width, y3, width, color=list(plt.rcParams['axes.prop_cycle'])[3]['color'])
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(['True Positive', 'True Negative', 'True Neutral'])
    plt.show()


def bar2plt(predictions):
    prediction = predictions
    x = np.arange(3)

    y, z, k = prediction

    ax = plt.subplot(111)
    ax.bar(x - 0.2, y, width=0.2, color='b', align='center')
    ax.bar(x, z, width=0.2, color='g', align='center')
    ax.bar(x + 0.2, k, width=0.2, color='r', align='center')
    ax.xaxis_date()

    plt.show()