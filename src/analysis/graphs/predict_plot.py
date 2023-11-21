import numpy as np
import matplotlib.pyplot as plt


def predict_plot(y_test, y_pred):
    # Plot the actual vs. predicted values
    plt.plot(y_test, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.legend()
    plt.show()
