# ### Exercise 1: Point-to-point communication
# from mpi4py import MPI
# from datetime import datetime
# import logging

# comm = MPI.COMM_WORLD
# no_of_ranks = comm.Get_size() # Number of processes
# my_rank = comm.Get_rank() # Rank executing presently
# start_time = datetime.now() # Time code started running

# # Logging allows processes to send messages
# logging.basicConfig(format='%(name)s: %(message)s')
# logger = logging.getLogger("Dot product")

# a = [0,1,2,3,4,5]
# b = [6,7,8,9,10,11]

# def dot_product(a_list, b_list):
#     sum=0
#     # Zip allows iteration through two lists simultaneously
#     for a,b in zip(a_list, b_list):
#         sum += a*b # add product to sum
#     return sum

# if len(a) == len(b):
#     # perform operation 100,000 times
#     for i in range(100000):
#         if my_rank == 0:
        
#             chunk_size = len(a)//(no_of_ranks) # Calculate the size of the data each rank will work on
#             total_sum = 0
#             # Operation for rank 0
#             a_chunk = a[0:chunk_size]
#             b_chunk = b[0:chunk_size]
#             total_sum += dot_product(a_chunk, b_chunk)
#             # Chunks for rest of the ranks, iterate through all ranks except 0
#             for rank in range(1, no_of_ranks):
#                 start = rank*chunk_size
#                 # In case of uneven division, last rank takes remainder
#                 if rank == no_of_ranks - 1:
#                     end = len(a)
#                 else:
#                     end = (rank+1)*chunk_size
#                 # Send chunks to the remaining ranks
#                 comm.send((a[start:end], b[start:end]), dest=rank)
#                 # Receive partial sum from remaining ranks
#                 sum = comm.recv(source=rank)
#                 # Add to total sum
#                 total_sum += sum

#         # This handles the receiving of the chunks, performing the dot product and sending partial sum to rank 0
#         else:
#             a_chunk, b_chunk = comm.recv(source=0)
#             part_sum = dot_product(a_chunk, b_chunk)
#             comm.send(part_sum, dest=0) # send partial sum to rank 0

#     if my_rank == 0:
#         logger.warning(f"rank {my_rank}: The dot product is: {total_sum}")
#         logger.warning(f"rank {my_rank}: Took {(datetime.now() - start_time)} seconds") # caculates the amount of time

# else:
#     print(f'The lists are not the same size')

import matplotlib.pyplot as plt

# Points
X1 = [[1.94,2],[5.06,2],[3.58,6.26],[3.42,6.74]]

# Subtle colors
color_X0 = '#6CA6CD'   # muted steel blue
color_X1 = '#CD5C5C'   # muted indian red

plt.figure(figsize=(8,6))

# Plot X⁰ points

# Plot X¹ points
for i, (x, y) in enumerate(X1):
    plt.scatter(x, y, color=color_X1, s=30, label='X¹' if i == 0 else "")
    plt.text(x + 0.1, y + 0.1, f'$X_{{{i+1}}}^{{1}}$', fontsize=9, color=color_X1)

# Highlight axes
plt.axhline(0, color='gray', linestyle='--', linewidth=1)  # y=0
plt.axvline(0, color='gray', linestyle='--', linewidth=1)  # x=0

# Mark the origin
plt.scatter(0, 0, color='black', s=40)
plt.text(0.1, 0.1, 'Origin (0,0)', fontsize=8, color='black')

# Title and formatting
plt.title("X⁰ and X¹ using shifting factor of 0.4 - Kruskal's approach", fontsize=9)
plt.xlabel("X-axis", fontsize=10)
plt.ylabel("Y-axis", fontsize=10)
plt.grid(True)
plt.axis('equal')
plt.tight_layout()

# Show only legend for X⁰ and X¹
plt.legend(fontsize=9)

plt.show()

# import matplotlib.pyplot as plt

# # Original points
# X0 = [[5,6],[5,-2],[5,2],[2,2],[8,2]]
# X1 = [[8,6],[5.6,6.4],[5,0],[6.6,-2.8],[-0.2,0.4]]

# # New points
# X1_0_5 = [[6.5,6],[5.3,2.2],[5,1],[4.3,-0.4],[3.9,1.2]]
# X1_0_25 = [[5.75,6],[5.15,0.1],[5,1.5],[3.15,0.8],[5.95,1.6]]

# # Define subtle colors
# color_X0 = '#6CA6CD'       # soft steel blue
# color_X1 = '#CD5C5C'       # muted indian red
# color_X1_05 = '#66CDAA'    # medium aquamarine (muted green)
# color_X1_025 = '#9370DB'   # medium purple

# plt.figure(figsize=(9,7))

# # Plot X⁰
# for i, (x, y) in enumerate(X0):
#     plt.scatter(x, y, color=color_X0, s=30, label='X⁰' if i == 0 else "")
#     plt.text(x + 0.1, y + 0.1, f'$X_{{{i+1}}}^{{0}}$', fontsize=9, color=color_X0)

# # Plot X¹
# for i, (x, y) in enumerate(X1):
#     plt.scatter(x, y, color=color_X1, s=30, label='X¹' if i == 0 else "")
#     plt.text(x + 0.1, y + 0.1, f'$X_{{{i+1}}}^{{1}}$', fontsize=9, color=color_X1)

# # Plot X₀.₅¹
# for i, (x, y) in enumerate(X1_0_5):
#     plt.scatter(x, y, color=color_X1_05, s=30, label='$X_{0.5}^1$' if i == 0 else "")
#     plt.text(x + 0.1, y + 0.1, f'$X_{{{i+1}}}^{{1}}$', fontsize=9, color=color_X1_05)

# # Plot X₀.₂₅¹
# for i, (x, y) in enumerate(X1_0_25):
#     plt.scatter(x, y, color=color_X1_025, s=30, label='$X_{0.25}^1$' if i == 0 else "")
#     plt.text(x + 0.1, y + 0.1, f'$X_{{{i+1}}}^{{1}}$', fontsize=9, color=color_X1_025)

# # Axis lines
# plt.axhline(0, color='gray', linestyle='--', linewidth=1)
# plt.axvline(0, color='gray', linestyle='--', linewidth=1)

# # Mark origin
# plt.scatter(0, 0, color='black', s=40)
# plt.text(0.1, 0.1, 'Origin (0,0)', fontsize=8, color='black')

# # Title and labels
# plt.title("X⁰, X¹, X₀.₅¹ and X₀.₂₅¹", fontsize=12)
# plt.xlabel("X-axis", fontsize=10)
# plt.ylabel("Y-axis", fontsize=10)
# plt.grid(True)
# plt.axis('equal')
# plt.tight_layout()
# plt.legend(fontsize=9)

# plt.show()



