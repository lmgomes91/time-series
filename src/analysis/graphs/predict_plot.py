import numpy as np
import matplotlib.pyplot as plt


def predict_plot(y_test: np.ndarray, y_pred: np.ndarray):
    # Plot the actual vs. predicted values
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.legend()
    plt.show()
