# import sys

# # For each line in input
# for line in sys.stdin:
#     line = line.strip() # Remove white spaces

#     # If input is empty, continue
#     if not line:
#         continue
#     try:
#         # Split using : which gives the node id and neighbors
#         node_id, neighbors = line.split(":")
#         # Remove white spaces from node_id
#         node_id = node_id.strip()
#         # SPlit neighbors using ,
#         neighbors = neighbors.strip().split(",")
#         # Iterate through neighbors and add 1 if value == "1"
#         neighbor_count = sum(1 for neighbor in neighbors if neighbor.strip() == '1')

#         print(f'{node_id}\t{neighbor_count}')

#     except Exception as e:
#         continue




import sys

for line in sys.stdin:
    line = line.strip()

    node_id, neighbors = line.split(":")
    node_id = node_id.strip()

    count = 0
    for neighbor in neighbors.strip().split(","):
        if neighbor.strip() == "1":
            count += 1

    print(f"{node_id}\t{count}")
