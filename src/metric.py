from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np

# Compute EER
def compute_eer(y, y_pred, plot_curves=True):
    fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    _filter = threshold <= 1
    if plot_curves:
        plt.plot(threshold[_filter], fnr[_filter], label='FRR')
        plt.plot(threshold[_filter], fpr[_filter], label='FAR')
        plt.legend()
        plt.show()

    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer_threshold, eer