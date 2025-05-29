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