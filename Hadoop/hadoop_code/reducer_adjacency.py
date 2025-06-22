# import sys

# # Define current node and neighbors
# current_node = None
# current_neighbors = 0

# # For each line in mapper output
# for line in sys.stdin:
#     line = line.strip() # Remove whitespace
#     # If mapper output is empty
#     if not line:
#         continue
#     # split the key-value pair using tab
#     node_id, neighbors = line.split('\t')
#     node_id = node_id.strip()
#     neighbors = int(neighbors.strip()) # Turn to integer

#     try:
#         # if the present node is the same as the current node,
#         # add the neighbor count
#         if node_id == current_node:
#             current_neighbors += neighbors
#         # else, make the present node the current node and print out the 
#         # current node and its number of neighbors
#         else:
#             if current_node is not None:
#                 print(f"{current_node}\t{current_neighbors}")
#             current_node = node_id
#             current_neighbors = neighbors

#     except ValueError:
#         continue
# # Last node_id
# if current_node is not None:
#     print(f"{current_node}\t{current_neighbors}")

import sys

current_node = None
current_count = 0

for line in sys.stdin:
    line = line.strip()

    node_id, num_neighbors = line.split("\t")

    num_neighbors = int(num_neighbors)

    if node_id == current_node:
        current_count += num_neighbors

    else:
        if current_node is not None:
            print(f"{current_node}\t{current_count}")
        
        current_node = node_id
        current_count = num_neighbors

if current_node is not None:
    print(f"{current_node}\t{current_count}")



    

