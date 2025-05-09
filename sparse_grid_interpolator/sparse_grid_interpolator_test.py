import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    max_error, r2_score
)
from sparse_grid_interpolator import SparseGridInterpolator

def test_function(points):
    """Тестовая 2D функция для интерполяции."""
    result = np.cos(points[:, 0] ** 2) + np.sin(points[:, 1] ** 2)
    return result

def mape(y_true, y_pred):
    """Средняя абсолютная процентная ошибка."""
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

if __name__ == '__main__':
    dim_count = 2
    max_level = 6

    train_grid = np.random.rand(50, dim_count)
    test_grid = np.random.rand(1000, dim_count)

    interp_bounds = np.array([[-2.0, 2.1], [-2.0, 2.1]]).T
    target_func = test_function

    interpolator = SparseGridInterpolator(max_level, dim_count, interp_bounds)
    interpolator.train(target_func, train_grid)

    true_vals = target_func(test_grid)
    predicted_vals = interpolator.predict(test_grid)
    errors = predicted_vals - true_vals

    # Метрики
    mse = mean_squared_error(true_vals, predicted_vals)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_vals, predicted_vals)
    me = max_error(true_vals, predicted_vals)
    mape_val = mape(true_vals, predicted_vals)
    r2 = r2_score(true_vals, predicted_vals)

    print(f"Обучающих точек: {len(train_grid)}")
    print(f"Тестовых точек:  {len(test_grid)}\n")
    print(f"MSE   (Mean Squared Error):           {mse:.6e}")
    print(f"RMSE  (Root Mean Squared Error):      {rmse:.6e}")
    print(f"MAE   (Mean Absolute Error):          {mae:.6e}")
    print(f"ME    (Max Absolute Error):           {me:.6e}")
    print(f"MAPE  (Mean Absolute Percentage Err): {mape_val:.2f} %")
    print(f"R²    (Coefficient of Determination): {r2:.6f}")

    # Визуализация в столбец
    fig = plt.figure(figsize=(8, 12))

    ax1 = fig.add_subplot(311, projection='3d')
    ax1.set_title("Истинные значения")
    ax1.plot_trisurf(test_grid[:, 0], test_grid[:, 1], true_vals, cmap='viridis')

    ax2 = fig.add_subplot(312, projection='3d')
    ax2.set_title("Интерполированные значения")
    ax2.plot_trisurf(test_grid[:, 0], test_grid[:, 1], predicted_vals, cmap='viridis')

    ax3 = fig.add_subplot(313, projection='3d')
    ax3.set_title("Ошибка интерполяции")
    ax3.plot_trisurf(test_grid[:, 0], test_grid[:, 1], errors, cmap='inferno')
    ax3.set_zlim(0.0, np.max(np.abs(errors)) * 2)

    plt.tight_layout()
    plt.show()