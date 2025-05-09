import numpy as np
from copy import deepcopy
from chebyshev_interpolation import *
from sparse_grid_interpolator_utils import *

class SparseGridInterpolator:
    """Выполняет иерархическую полиномиальную интерполяцию на разреженных сетках с использованием полиномов Чебышёва.

    Реализует раннюю остановку, когда абсолютная ошибка становится меньше заданного порога.

    Атрибуты:
        max_level: int
            Максимальный уровень для иерархической интерполяции на разреженных сетках.
        dimensions: int
            Размерность пространства выборки.
        interp_range: ndarray
            Интервал интерполяции для каждого измерения.
        tolerance: float
            Порог ранней остановки по максимальной абсолютной ошибке.
        grids: dict
            Хранит данные сетки для каждого уровня: позиции узлов, значения функции, мультииндексы, остатки и ошибки.
    """
    def __init__(self, max_level, dimensions, interp_range=None, tolerance=1e-3):
        self.max_level = max_level
        self.dimensions = dimensions
        self.tolerance = tolerance
        self.interp_range = interp_range
        self.grids = {}

    def _scale_grid(self, nodes):
        """Масштабирует сетку к заданному интервалу интерполяции для каждого измерения."""
        for dim in range(self.dimensions):
            bounds = self.interp_range[:, dim]
            span = abs(bounds[1] - bounds[0])
            nodes[:, dim] = nodes[:, dim] * span + bounds[0]
        return nodes

    def _should_stop(self, max_err):
        """Проверяет, достигнута ли максимальная ошибка критерия ранней остановки."""
        return max_err < self.tolerance

    def train(self, target_func, eval_points):
        """Выполняет n-мерную интерполяцию на разреженной сетке.

        Аргументы:
            target_func: callable
                Функция для аппроксимации.
            eval_points: ndarray (N, d)
                Координаты точек для оценки интерполяции.

        Возвращает:
            ndarray (N,): Интерполированные значения.
        """
        output_size = eval_points.shape[0]
        self.grids['output'] = eval_points
        result = np.zeros(output_size, dtype=float)

        for level in range(self.max_level + 1):
            nodes, node_counts, indices = self._generate_sparse_nodes(level, self.dimensions)

            if level == 0:
                indices = indices[np.newaxis, :]

            nodes = self._scale_grid(nodes)
            func_vals = target_func(nodes)
            residuals = deepcopy(func_vals)

            if level > 0:
                if self.grids[level - 1]['max_error'] < self.tolerance:
                    return result
                for prev_level in range(level):
                    
                    prev_grid = self.grids[prev_level]
                    interp_vals = self._interpolate_sparse(
                        self.dimensions,
                        prev_grid['residuals'],
                        prev_grid['nodes'],
                        nodes,
                        prev_grid['indices'],
                        prev_grid['node_counts'],
                        self.interp_range
                    )
                    residuals -= interp_vals

            result = self._interpolate_sparse(
                self.dimensions,
                residuals,
                nodes,
                eval_points,
                indices,
                node_counts,
                self.interp_range,
                result
            )

            self.grids[level] = {
                'nodes': nodes,
                'func_vals': func_vals,
                'indices': indices,
                'node_counts': node_counts,
                'residuals': residuals,
                'max_error': np.max(np.abs(residuals)),
                'mean_error': np.mean(np.abs(residuals))
            }
            self.grids['max_depth'] = level

        return result

    def predict(self, eval_points):
        """Вычисляет значения функции интерполяции в заданных точках.

        Аргументы:
            eval_points: ndarray (N, d)
                Координаты точек для оценки интерполяции.

        Возвращает:
            ndarray (N,): Интерполированные значения.
        """
        output_size = eval_points.shape[0]
        result = np.zeros(output_size, dtype=float)
        max_depth = self.grids['max_depth']

        for level in range(max_depth + 1):
            grid_data = self.grids[level]
            result = self._interpolate_sparse(
                self.dimensions,
                grid_data['residuals'],
                grid_data['nodes'],
                eval_points,
                grid_data['indices'],
                grid_data['node_counts'],
                self.interp_range,
                result
            )
        return result

    def _generate_sparse_nodes(self, level, dimensions):
        """Генерирует узлы полинома для построения разреженной сетки.

        Аргументы:
            level: int
                Уровень полинома для интерполяции Чебышёва.
            dimensions: int
                Размерность пространства.

        Возвращает:
            tuple: (nodes, node_counts, indices)
        """
        if level == 0:
            nodes = 0.5 * np.ones((1, dimensions), dtype=float)
            indices = np.array([0] * dimensions)
            node_counts = [1]
            return nodes, node_counts, indices

        multi_indices = generate_multi_indices(level, dimensions)
        node_counts, coords = setup_nodes(level)
        sparse_indices, points = build_sparse_grid(dimensions, node_counts, multi_indices, coords)

        unique_points = []
        index_mapping = []
        point_hashes = [str(pt) for pt in points]

        for idx, point_hash in enumerate(point_hashes):
            if point_hash not in unique_points:
                unique_points.append(point_hash)
                index_mapping.append(idx)

        nodes = np.zeros((len(unique_points), dimensions), dtype=float)
        indices = np.zeros((len(unique_points), dimensions), dtype=int)

        for i, point_hash in enumerate(unique_points):
            indices[i, :] = sparse_indices[index_mapping[i], :]
            coords = np.fromstring(point_hash.strip('[]'), sep=' ')
            nodes[i, :] = coords

        return nodes, node_counts, indices

    @staticmethod
    def _interpolate_sparse(dimensions, residuals, input_nodes, output_nodes, indices, node_counts, interp_range, result=None):
        """Выполняет n-мерную интерполяцию на разреженной сетке.

        Аргументы:
            dimensions: int
                Размерность пространства выборки.
            residuals: ndarray
                Иерархические остатки.
            input_nodes: ndarray
                Входные узлы для интерполяции.
            output_nodes: ndarray
                Точки для вывода интерполяции.
            indices: ndarray
                Мультииндексы для тензорных комбинаций.
            node_counts: list
                Количество узлов для каждого базиса.
            interp_range: ndarray
                Интервал интерполяции.
            result: ndarray, опционально
                Инициализированные интерполированные значения.

        Возвращает:
            ndarray: Интерполированные значения в output_nodes.
        """
        output_size = output_nodes.shape[0]
        if result is None:
            result = np.zeros(output_size, dtype=float)

        compute_chebyshev_poly(
            dimensions, result, residuals, output_size,
            node_counts, indices, output_nodes, input_nodes, interp_range
        )
        return result