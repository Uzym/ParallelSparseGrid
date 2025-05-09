import numpy as np
from numpy.matlib import repmat
import itertools as it

def setup_nodes(level):
    """Инициализирует количество узлов и их координаты для каждого уровня.

    Для уровня i=1: nnodes(1) = 1, x_coord(1,1) = 0.5.

    Аргументы:
        level: int
            Уровень для вычисления узлов.

    Возвращает:
        node_counts: list
            Количество узлов для каждого уровня.
        node_coords: list
            Координаты узлов.
    """
    node_counts = [1]
    node_coords = [0.5]
    for i in range(2, level + 2):
        node_counts.append(2 ** (i - 1) + 1)
        coords = [(1 + (-np.cos((np.pi * (j - 1)) / (node_counts[i - 1] - 1)))) / 2.0
                  for j in range(1, node_counts[i - 1] + 1)]
        node_coords.append(coords)
    return node_counts, node_coords

def build_sparse_grid(dim_count, node_counts, multi_indices, node_coords):
    """Формирует разреженную сетку на основе мультииндексов и координат узлов."""
    points = np.zeros((100000, dim_count), dtype=float)
    start_idx = 0
    index_array = []

    for i in range(len(multi_indices)):
        grid_size = np.prod([node_counts[multi_indices[i, d]] for d in range(dim_count)])
        temp_grid = np.zeros((grid_size, dim_count), dtype=float)
        temp_indices = repmat(multi_indices[i, :], grid_size, 1)
        index_array.extend(temp_indices)

        for dim in range(dim_count):
            coord_vals = np.array(node_coords[multi_indices[i, dim]])
            temp_vals = np.zeros((grid_size, 1), dtype=float)
            temp_vals2 = np.zeros((grid_size, 1), dtype=float)

            for k in range(0, grid_size, node_counts[multi_indices[i, dim]]):
                temp_vals[k:k + node_counts[multi_indices[i, dim]], 0] = coord_vals

            increment = 1
            if i > 0:
                for prev_dim in range(dim):
                    increment *= node_counts[multi_indices[i, prev_dim]]

            for k in range(0, grid_size, increment):
                k_start = k // increment
                k_end = k_start + increment * node_counts[multi_indices[i, dim]]
                if increment == 1:
                    temp_vals2[:, 0] = temp_vals[:, 0]
                else:
                    temp_vals2[k:k + increment, 0] = temp_vals[k_start:k_end:node_counts[multi_indices[i, dim]], 0]

            temp_grid[:, dim] = temp_vals2[:, 0]

        points[start_idx:start_idx + grid_size, :] = temp_grid
        start_idx += grid_size

    points = np.round(points[:start_idx, :], decimals=5)
    index_array = np.array(index_array)
    return index_array, points

def generate_multi_indices(level, dim_count):
    """Генерирует последовательность мультииндексов для разреженных сеток."""
    level_combinations = it.combinations(range(level + dim_count - 1), dim_count - 1)
    total_levels = sum(1 for _ in level_combinations)
    index_seq = np.zeros((total_levels, dim_count), dtype=int)

    index_seq[0, 0] = level
    max_index = level

    level_combinations = it.combinations(range(level + dim_count - 1), dim_count - 1)
    level_iter = 1

    for _ in range(1, total_levels):
        if index_seq[level_iter - 1, 0] > 0:
            index_seq[level_iter, 0] = index_seq[level_iter - 1, 0] - 1
            for dim in range(1, dim_count):
                if index_seq[level_iter - 1, dim] < max_index:
                    index_seq[level_iter, dim] = index_seq[level_iter - 1, dim] + 1
                    for next_dim in range(dim + 1, dim_count):
                        index_seq[level_iter, next_dim] = index_seq[level_iter - 1, next_dim]
                    break
        else:
            sum_indices = 0
            for dim in range(1, dim_count):
                if index_seq[level_iter - 1, dim] < max_index:
                    index_seq[level_iter, dim] = index_seq[level_iter - 1, dim] + 1
                    sum_indices += index_seq[level_iter, dim]
                    for next_dim in range(dim + 1, dim_count):
                        index_seq[level_iter, next_dim] = index_seq[level_iter - 1, next_dim]
                        sum_indices += index_seq[level_iter, next_dim]
                    break
                else:
                    temp_sum = sum(index_seq[level_iter - 1, dim + 2:dim_count])
                    max_index = level - temp_sum
                    index_seq[level_iter, dim] = 0
            index_seq[level_iter, 0] = level - sum_indices
            max_index = level - sum_indices
        level_iter += 1

    return index_seq