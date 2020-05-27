#from predict_dec import predict_dec
import matplotlib.pyplot as plt
from plot_decision_boundary import plot_decision_boundary
from predict import predict

def twoD_plot(parameters, train_X, train_Y):
    plt.figure(3)
    plt.title("Model with regularization")
    axes = plt.gca()
    axes.set_xlim([-0.75,0.40])
    axes.set_ylim([-0.75,0.65])
    plot_decision_boundary(lambda x: predict(x.T, parameters), train_X, train_Y)