import numpy as np

def compute_chebyshev_poly(dim_count, result, residuals, output_size, node_counts, indices, output_grid, input_grid, interp_bounds):
    """Вычисляет интерполяцию с использованием барицентрических полиномов Чебышёва.

    Формула: wght2_j = SUM_x_m[(x - x_m)/(x_j - x_m)], для всех x_m != x_j
    """
    level_count = indices.shape[0]
    poly_weights = np.zeros((level_count, dim_count), dtype=float)
    interp_factors = np.ones(level_count, dtype=float)

    for i in range(output_size):
        for level in range(level_count):
            interp_factors[level] = 1.0
            for dim in range(dim_count):
                poly_weights[level, dim] = 1.0
                if node_counts[indices[level, dim]] != 1:
                    for node in range(node_counts[indices[level, dim]]):
                        x_temp = 0.5 * (1. + (-np.cos((np.pi * node) / (node_counts[indices[level, dim]] - 1))))
                        # Масштабирование x_temp к заданному интервалу
                        delta = abs(interp_bounds[0, dim] - interp_bounds[1, dim])
                        x_temp = x_temp * delta + interp_bounds[0, dim]
                        # Проверка, чтобы избежать деления на ноль
                        if abs(input_grid[level, dim] - x_temp) > 1e-03:
                            factor = (output_grid[i, dim] - x_temp) / (input_grid[level, dim] - x_temp)
                            poly_weights[level, dim] *= factor
                interp_factors[level] *= poly_weights[level, dim]
            result[i] += interp_factors[level] * residuals[level]