### Exercise 1: Point-to-point communication
from mpi4py import MPI
from datetime import datetime
import logging

comm = MPI.COMM_WORLD
no_of_ranks = comm.Get_size() # Number of processes
my_rank = comm.Get_rank() # Rank executing presently
start_time = datetime.now() # Time code started running

# Logging allows processes to send messages
logging.basicConfig(format='%(name)s: %(message)s')
logger = logging.getLogger("Dot product")

a = [0,1,2,3,4,5]
b = [6,7,8,9,10,11]

def dot_product(a_list, b_list):
    sum=0
    # Zip allows iteration through two lists simultaneously
    for a,b in zip(a_list, b_list):
        sum += a*b # add product to sum
    return sum

if len(a) == len(b):
    # perform operation 100,000 times
    for i in range(100000):
        if my_rank == 0:
        
            chunk_size = len(a)//(no_of_ranks) # Calculate the size of the data each rank will work on
            total_sum = 0
            # Operation for rank 0
            a_chunk = a[0:chunk_size]
            b_chunk = b[0:chunk_size]
            total_sum += dot_product(a_chunk, b_chunk)
            # Chunks for rest of the ranks, iterate through all ranks except 0
            for rank in range(1, no_of_ranks):
                start = rank*chunk_size
                # In case of uneven division, last rank takes remainder
                if rank == no_of_ranks - 1:
                    end = len(a)
                else:
                    end = (rank+1)*chunk_size
                # Send chunks to the remaining ranks
                comm.send((a[start:end], b[start:end]), dest=rank)
                # Receive partial sum from remaining ranks
                sum = comm.recv(source=rank)
                # Add to total sum
                total_sum += sum

        # This handles the receiving of the chunks, performing the dot product and sending partial sum to rank 0
        else:
            a_chunk, b_chunk = comm.recv(source=0)
            part_sum = dot_product(a_chunk, b_chunk)
            comm.send(part_sum, dest=0) # send partial sum to rank 0

    if my_rank == 0:
        logger.warning(f"rank {my_rank}: The dot product is: {total_sum}")
        logger.warning(f"rank {my_rank}: Took {(datetime.now() - start_time)} seconds") # caculates the amount of time

else:
    print(f'The lists are not the same size')