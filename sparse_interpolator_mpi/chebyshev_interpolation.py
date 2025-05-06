import numpy as np
from mpi4py import MPI

class MPIChebyshevInterpolationWrapper:
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.is_controller = self.rank == 0

    def run(self, dim_count, residuals, output_size, node_counts, 
           indices, output_grid, input_grid, interp_bounds):
        """
        Запускает вычисления и возвращает результат только на контроллере (rank 0).
        Воркеры автоматически завершаются после выполнения.
        """
        if self.size == 1:
            return self._sequential_version(dim_count, residuals, output_size, node_counts,
                                           indices, output_grid, input_grid, interp_bounds)

        # Воркеры выполняют свою часть и завершаются
        if not self.is_controller:
            self._worker_logic(dim_count, residuals, node_counts, indices, 
                              output_grid, input_grid, interp_bounds)
            return None

        # Контроллер выполняет основную логику
        return self._controller_logic(output_size)

    def _worker_logic(self, dim_count, residuals, node_counts, indices, output_grid, input_grid, interp_bounds):
        """Логика выполнения для воркеров (rank > 0)"""
        while True:
            task = self.comm.bcast(None, root=0)
            if task is None:  # Контроллер сигнализирует о завершении
                break
            
            i = task
            result = self._calculate_single_point(dim_count, residuals, node_counts, indices,
                                                 output_grid[i], input_grid, interp_bounds)
            self.comm.Send(result, dest=0, tag=i)

    def _controller_logic(self, output_size):
        """Логика выполнения для контроллера (rank 0)"""
        interpol = np.zeros(output_size, dtype=np.float64)
        processed = 0

        # Отправляем первоначальные задания
        for i in range(min(output_size, self.size-1)):
            self.comm.send(i, dest=i+1, tag=0)

        # Основной цикл обработки
        while processed < output_size:
            status = MPI.Status()
            result = np.empty(1, dtype=np.float64)
            self.comm.Recv(result, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            i = status.tag
            interpol[i] = result[0]
            processed += 1

            # Отправляем следующее задание
            if processed + self.size - 1 < output_size:
                next_i = processed + self.size - 1
                self.comm.send(next_i, dest=status.source, tag=0)

        # Сигнал завершения
        for i in range(1, self.size):
            self.comm.send(None, dest=i, tag=0)

        return interpol

    def _calculate_single_point(self, dim_count, residuals, node_counts, indices,
                               output_grid_i, input_grid, interp_bounds):
        """Вычисление для одной точки"""
        level_size = indices.shape[0]
        total = 0.0

        for level in range(level_size):
            weight = 1.0
            for dim in range(dim_count):
                current_node = node_counts[indices[level, dim]]
                if current_node == 1:
                    continue

                for node in range(current_node):
                    xtmp = 0.5 * (1.0 + (-np.cos((np.pi * node) / (current_node - 1))))
                    delta = interp_bounds[1, dim] - interp_bounds[0, dim]
                    xtmp = xtmp * delta + interp_bounds[0, dim]
                    if abs(input_grid[level, dim] - xtmp) > 1e-3:
                        val = (output_grid_i[dim] - xtmp) / (input_grid[level, dim] - xtmp)
                        weight *= val

            total += weight * residuals[level]

        return np.array([total], dtype=np.float64)

    def _sequential_version(self, dim_count, residuals, output_size, node_counts,
                           indices, output_grid, input_grid, interp_bounds):
        """Последовательная версия для однопроцессорного режима"""
        interpol = np.zeros(output_size, dtype=np.float64)
        for i in range(output_size):
            result = self._calculate_single_point(dim_count, residuals, node_counts, indices,
                                                 output_grid[i], input_grid, interp_bounds)
            interpol[i] = result[0]
        return interpol
    
def compute_chebyshev_poly(dim_count, result, residuals, output_size, node_counts, indices, output_grid, input_grid, interp_bounds):
    result = MPIChebyshevInterpolationWrapper().run(dim_count, residuals, output_size, node_counts, indices, output_grid, input_grid, interp_bounds)