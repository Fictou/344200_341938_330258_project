import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

from main import main

def test_mlp_combinations():
    learning_rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    max_iters = [0, 25, 50, 75, 100, 125, 150]
    accuracies = np.zeros((len(learning_rates), len(max_iters)))

    args = argparse.Namespace(
        data="dataset",
        nn_type="mlp",
        nn_batch_size=64,
        device="cpu",
        use_pca=False,
        pca_d=100,
        lr=0.1,
        max_iters=100,
        test=False
    )

    test = 0
    for i, lr in enumerate(learning_rates):
        for j, iters in enumerate(max_iters):
            args.lr = lr
            args.max_iters = iters
            print("=================================================================")
            print("Test N°", test)
            print("lr=", lr)
            print("max_iter=", iters)
            test = test + 1
            accuracy = main(args)
            print("=================================================================")
            accuracies[i, j] = accuracy

    X, Y = np.meshgrid(max_iters, learning_rates)
    Z = accuracies

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_xlabel('Max Iterations')
    ax.set_ylabel('Learning Rate')
    ax.set_zlabel('Accuracy')

    plt.title('MLP Model Accuracy by Learning Rate and Max Iterations')
    plt.show()

if __name__ == '__main__':
    test_mlp_combinations()
