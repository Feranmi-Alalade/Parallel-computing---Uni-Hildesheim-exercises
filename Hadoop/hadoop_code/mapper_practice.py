# import sys

# for line in sys.stdin:
#     line = line.strip()

#     words = line.split()

#     for word in words:
#         word = word.strip()

#         print(f"{word}\t1")

# import sys

# for line in sys.stdin:
#     line = line.strip()

#     words = line.split()

#     for word in words:
#         word = word.strip()
#         print(f"{word[0].upper()}\t1")

# import sys

# for line in sys.stdin:
#     line = line.strip()

#     document, words = line.split("\t")
#     document = document.strip()



#     for word in words.split():
#         word = word.strip()
#         print(f"{word}\t{document}")

# import sys

# for line in sys.stdin:
#     line = line.strip()

#     numbers = line.split()

#     for number in numbers:
#         number = number.strip()
#         print(f"{number}\t1")

import sys

for line in sys.stdin:
    if not line:
        continue
    line = line.strip()
    # line = line. strip()

    line_elements = line.split()

    matrix, value = line_elements[1], float(line_elements[2])

    print(f"{matrix}\t{value}")






