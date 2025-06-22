# import sys

# current_word = None
# current_count = 0

# for line in sys.stdin:
#     line = line.strip()

#     word, count = line.split("\t")
#     word = word.strip()
#     count = int(count)

#     if current_word == word:
#         current_count += count

#     else:
#         if current_word is not None:
#             print(f"{current_word}\t{current_count}")

#         current_word = word
#         current_count = count

# if current_word is not None:
#     print(f"{current_word}\t{current_count}")






# import sys

# current_letter = None
# current_count = 0

# for line in sys.stdin:
#     line = line.strip()

#     letter, count = line.split("\t")
#     letter = letter.strip()
#     count = int(count)

#     if current_letter == letter:
#         current_count += count
#     else:
#         if current_letter is not None:
#             print(f"{current_letter}\t{current_count}")

#         current_letter = letter
#         current_count = count

# if current_letter is not None:
#     print(f"{current_letter}\t{current_count}")

# import sys

# current_word = None
# current_doc = ""

# for line in sys.stdin:
#     line = line.strip()

#     word, document = line.split("\t")
#     word = word.strip()
#     document = document.strip()

#     if current_word == word:
#         current_doc = current_doc + "," + document
#     else:
#         if current_word is not None:
#             print(f"{current_word}\t{current_doc}")
#         current_word = word
#         current_doc = document

# if current_word is not None:
#     print(f"{current_word}\t{current_doc}")

import sys

A = []
B = []

for line in sys.stdin:
    if not line:
        continue
    line = line.strip()

    matrix, value = line.split("\t")

    value = float(value)

    if matrix == "A":
        A.append(value)
    elif matrix == "B":
        B.append(value)

dot_product = 0
for a_i, b_i in zip(A,B):
    dot_product += a_i*b_i

print(f"The dot product is {dot_product}")
    



            





