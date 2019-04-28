import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def _plot_confusion_matrix(cm,labels, cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    xlocations = np.array(range(len(labels)))
    plt.xticks([])
    plt.yticks(xlocations, labels)

def plot_confusion_matrix(y_true, y_pred, save_dir, labels):

    tick_marks = np.array(range(len(labels))) + 0.5
    cm = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # print (cm_normalized )
    if len(labels) > 10:
         plt.figure(figsize=(12, 8), dpi=120)
    else:
         plt.figure(figsize=(6, 4), dpi=120)

    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
         c = cm_normalized[y_val][x_val]
         if c > 0.01 and c<0.4 :
              plt.text(x_val, y_val, "%0.2f" % (c,), color='black', fontsize=6, va='center', ha='center')
         if c>=0.4 and c<=1 :
              plt.text(x_val, y_val, "%0.2f" % (c,), color='white', fontsize=6, va='center', ha='center')

    plt.gca().set_xticks(tick_marks, minor=False)
    plt.gca().set_yticks(tick_marks, minor=False)
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('left')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    _plot_confusion_matrix(cm_normalized, labels)
    plt.savefig(save_dir, format='png')
#     plt.show()
