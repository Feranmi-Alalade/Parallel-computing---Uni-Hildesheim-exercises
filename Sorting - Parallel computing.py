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

### Exercise 3
from mpi4py import MPI
import numpy as np
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank() # Rank executing presently
no_of_ranks = comm.Get_size() # Number of processes


my_list = list(range(10000))
np.random.shuffle(my_list) #Shuffle the list


if my_rank == 0:
    chunk_size = len(my_list)//no_of_ranks # Calculate the size of the data each rank will work on

    list_chunks = [] # Empty list to receive lists for each of the ranks

    # Iterate through ranks
    for rank in range(no_of_ranks):
        start = rank*chunk_size # Calculate starting index
        if rank == no_of_ranks-1: # Last rank
            end = len(my_list) # In case of uneven division, last rank takes remainder
        else:
            end = (rank+1)*chunk_size

        list_chunks.append(my_list[start:end]) # Add to list

else:
    chunk_size = None
    list_chunks = None

list_chunk = comm.scatter(list_chunks, root=0) # This distributes the list_chunks to all processes from the root rank

sublist = sorted(list_chunk) # sort the sublists

final_list = comm.gather(sublist, root=0) # Gathers the sublists from all processes and sends to rank 0

if my_rank == 0:
    ordered_list = []

    # Iterate through all elements of my_list
    for i in range(len(my_list)):
        first_elements = []
        for index, sublist in enumerate(final_list):
            if sublist:
                # Appends the first and smallest value in each sublist along with the sublist index to first_elements
                first_elements.append((sublist[0], index)) 

        # Finds the minimum of the first elements across all sublists
        min_first, min_index = min(first_elements)

        ordered_list.append(final_list[min_index].pop(0)) # Adds the minimum value to ordered_list and removes it from the sublist

    # To check if our sorting is correct
    if sorted(my_list) == ordered_list:
        print("Yes")
        # print(ordered_list)
    else:
        print("No")