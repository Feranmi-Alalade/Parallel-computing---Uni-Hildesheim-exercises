### Exercise 2: Collective communication
from mpi4py import MPI
from datetime import datetime
import logging

comm = MPI.COMM_WORLD
start_time = datetime.now() # Time the code started running

no_of_ranks = comm.Get_size() # Rank executing presently
my_rank = comm.Get_rank() # Number of processes

# Logging allows the processes to send messages
logging.basicConfig(format='%(name)s: %(message)s')
logger = logging.getLogger("Dot Product")

a = [0,1,2,3,4,5]
b = [6,7,8,9,10,11]

def dot_product(list_a, list_b):
    product = 0
    for a,b in zip(list_a, list_b):
        product += a*b
    return product

if len(a) != len(b):
    exit()

else:
    # Perform the operation 100,000 times
    for i in range(100000):
        if my_rank == 0:
            chunk_size = len(a)//(no_of_ranks) # Calculate the size of the data each rank will work on

            a_chunks = [] # splitted into chunks to be distributed among processes
            b_chunks = []

            # Iterate through processes
            for rank in range(0, no_of_ranks):
                start = rank*chunk_size
                # In case of uneven division, last rank takes remainder
                if rank == no_of_ranks-1:
                    end = len(a) 
                else:
                    end = (rank+1)*chunk_size

                # Add chunks for  each rank to list
                a_chunks.append(a[start:end]) 
                b_chunks.append(b[start:end])

        else:
            a_chunks = None
            b_chunks = None
            chunk_size = None

        
        # Distribute the chunks to processes
        a_chunk = comm.scatter(a_chunks, root=0) 
        b_chunk = comm.scatter(b_chunks, root=0)

        # Perform dot product using function defined above
        part_result = dot_product(a_chunk, b_chunk)
        #Calculate the sum of the results from all processes and send to the root rank
        total_result = comm.reduce(part_result, op=MPI.SUM, root=0)


    if my_rank == 0:
        logger.warning(f"rank {my_rank}: The dot product is: {total_result}")
        logger.warning(f"rank {my_rank}: It took {datetime.now() - start_time} seconds") # Returns how many seconds it took