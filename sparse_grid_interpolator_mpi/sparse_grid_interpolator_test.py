from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from sparse_grid_interpolator import SparseGridInterpolator
from chebyshev_interpolation import *

def test_function(points):
    """Тестовая 2D функция для интерполяции."""
    result = ((
        0.5 / np.pi * points[:, 0] -
        0.51 / (0.4 * np.pi ** 2) * points[:, 0] ** 2 +
        points[:, 1] - 0.6) ** 2 +
        (1 - 1 / (0.8 * np.pi)) * np.cos(points[:, 0]) + 0.10)
    return result

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    wrapper = MPIChebyshevInterpolationWrapper()

    if rank != 0:
        wrapper.worker_loop()
        return
    dim_count = 2
    max_level = 6
    grid_shape = (grid_x, grid_y) = (21, 21)
    total_points = grid_x * grid_y

    x_vals = np.linspace(0, 1, grid_x)
    y_vals = np.linspace(0, 1, grid_y)
    X, Y = np.meshgrid(x_vals, y_vals)
    eval_grid = np.array([X.ravel(), Y.ravel()]).T

    interp_bounds = np.array([[0.0, 1.0], [0.0, 1.0]]).T
    target_func = test_function

    interpolator = SparseGridInterpolator(max_level, dim_count, interp_bounds)
    interp_vals = interpolator.train(target_func, eval_grid)
    predicted_vals = interpolator.predict(eval_grid)
    MPIChebyshevInterpolationWrapper().stop_workers()
    print(np.ptp(interp_vals - predicted_vals))

    true_vals = target_func(eval_grid).reshape(grid_shape)
    interp_vals = interp_vals.reshape(grid_shape)

    fig = plt.figure(figsize=(16, 6))

    ax = fig.add_subplot(131, projection='3d', title="Исходные значения")
    ax.plot_surface(X, Y, true_vals, rstride=1, cstride=1, cmap=plt.cm.magma)

    ax = fig.add_subplot(132, projection='3d', title="Интерполированные значения")
    ax.plot_surface(X, Y, interp_vals, rstride=1, cstride=1, cmap=plt.cm.magma)

    ax = fig.add_subplot(133, projection='3d', title="Ошибка интерполяции")
    ax.plot_surface(X, Y, interp_vals - true_vals, rstride=1, cstride=1, cmap=plt.cm.magma)
    ax.set_zlim(0.0, ax.get_zlim()[1] * 2)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
    