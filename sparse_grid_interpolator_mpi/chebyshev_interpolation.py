import numpy as np
from mpi4py import MPI

class MPIChebyshevInterpolationWrapper:
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.is_controller = self.rank == 0

    def run(self, interpol, dim_count, residuals, output_size, node_counts,
            indices, output_grid, input_grid, interp_bounds):
        if not self.is_controller:
            return None
        processed = 0
        next_task = 0
        active_workers = self.size - 1

        try:
            for worker in range(1, self.size):
                if next_task < output_size:
                    self.comm.send({
                        'task_id': next_task,
                        'dim_count': dim_count,
                        'residuals': residuals,
                        'node_counts': node_counts,
                        'indices': indices,
                        'output_point': output_grid[next_task],
                        'input_grid': input_grid,
                        'interp_bounds': interp_bounds
                    }, dest=worker, tag=1)
                    next_task += 1

            while processed < output_size:
                status = MPI.Status()
                result, task_id = self.comm.recv(source=MPI.ANY_SOURCE, tag=1, status=status)
                interpol[task_id] = result[0]
                processed += 1
                source = status.Get_source()
                #print(f"[controller] have res from {source} for task {task_id}")

                if next_task < output_size:
                    self.comm.send({
                        'task_id': next_task,
                        'dim_count': dim_count,
                        'residuals': residuals,
                        'node_counts': node_counts,
                        'indices': indices,
                        'output_point': output_grid[next_task],
                        'input_grid': input_grid,
                        'interp_bounds': interp_bounds
                    }, dest=source, tag=1)
                    next_task += 1
        except Exception as e:
            print(f"[controller] error: {e}")
            for worker in range(1, self.size):
                self.comm.send(None, dest=worker, tag=0)
        
        return interpol

    def stop_workers(self):
        for worker in range(1, self.size):
            self.comm.send(None, dest=worker, tag=0)

    def worker_loop(self):
        while True:
            status = MPI.Status()
            data = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            
            if tag == 0:
                print(f"[worker {self.rank}] end")
                return

            #print(f"[worker {self.rank}] solve {data['task_id']}")
            result = self._calculate_single_point(
                data['dim_count'],
                data['residuals'],
                data['node_counts'],
                data['indices'],
                data['output_point'],
                data['input_grid'],
                data['interp_bounds']
            )
            self.comm.send((result, data['task_id']), dest=0, tag=1)

    def _calculate_single_point(self, dim_count, residuals, node_counts, indices,
                                 output_grid_i, input_grid, interp_bounds):
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