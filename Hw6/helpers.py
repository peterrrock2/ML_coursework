import matplotlib.pyplot as plt
import matplotlib.animation
from PIL import Image
import numpy as np
import io
from base64 import b64encode
from IPython.display import HTML
import logging

logging.getLogger().setLevel(logging.ERROR)
plt.rcParams["animation.html"] = "jshtml"
plt.rcParams['figure.dpi'] = 100


def show_decision_surface(model, X, y, ax=None):
    """
    Helper function to visualize the decision surface of the trained model
    :param model with predict method
    :return: None
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x_grid, y_grid = np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1)
    xx, yy = np.meshgrid(x_grid, y_grid)
    r1, r2 = xx.reshape(-1, 1), yy.reshape(-1, 1)
    grid = np.hstack((r1, r2))
    y_hat = model.predict(grid).reshape(-1, )
    zz = y_hat.reshape(xx.shape)

    if ax is None:
        plt.contourf(xx, yy, zz, cmap='PiYG')
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.show()
    else:
        ax.contourf(xx, yy, zz, cmap='PiYG')
        ax.scatter(X[:, 0], X[:, 1], c=y)


def display_matrix(c_matrix):
    """
    Displays the confusion matrix using matrix show
    Args:
        c_matrix: square confusion matrix, shape (num_classes, num_classes)
    """
    _, ax = plt.subplots()
    ax.matshow(c_matrix, cmap=plt.cm.Blues)
    for i in range(c_matrix.shape[0]):
        for j in range(c_matrix.shape[0]):
            ax.text(i, j, "{:.2f}".format(c_matrix[j, i]), va='center', ha='center')
    plt.show()


def create_buffer(X, assignments):
    plt.ioff()
    figure = plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=assignments)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    return buf


def show_progress(snapshots):
    num_snapshots = len(snapshots)
    figure = plt.figure()
    figure.set_figheight(5), figure.set_figwidth(8)

    def animate(t):
        t = int(t)
        plt.cla()
        snapshots[t].seek(0)
        im = Image.open(snapshots[t])
        plt.axis('off')
        plt.imshow(im)

    anim = matplotlib.animation.FuncAnimation(figure, animate, frames=num_snapshots)
    return anim


def save_frames(frames, target_mp4):
    import imageio
    imageio.mimwrite('{}.mp4'.format(target_mp4), np.array(frames, dtype="uint8"), fps=10)


def display_video(target_mp4):
    mp4 = open('{}.mp4'.format(target_mp4), 'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML("""
        <video width=300 controls>
            <source src="%s" type="video/mp4">
        </video>
        """ % data_url)
