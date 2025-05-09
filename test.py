from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print(f"[RANK {rank}] Entered main()")

    if rank != 0:
        print(f"[RANK {rank}] entering dummy loop")
        while True:
            pass  # Просто цикл, чтобы не выйти
    else:
        print(f"[RANK {rank}] I am the controller")
        
if __name__ == '__main__':
    main()