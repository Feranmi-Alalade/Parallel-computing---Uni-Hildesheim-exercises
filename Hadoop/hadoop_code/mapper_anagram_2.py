import sys
import numpy as np

# Split into lines
for line in sys.stdin:
    # if line is empty
    if not line:
        continue
    #remove white spaces
    line = line.strip()

    anagram, word_list = line.split("\t")
    word_count = word_list.split(",") 

    count_words = len(word_count) 

    count_words = int(count_words)

    # For sorting in descending order, do an inverse of the count
    inv_count = 100 - count_words

    print(f"{inv_count}\t{word_list}\t{anagram}")