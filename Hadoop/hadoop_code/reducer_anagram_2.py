import sys

threshold = 3

for line in sys.stdin:
    if not line:
        continue
    line = line.strip()

    inv_count, word_list, anagram = line.split("\t")
    inv_count = int(inv_count)

    # Turn back to original count
    count = 100 - inv_count

    if count >= threshold:
        print(f"{anagram}\t{word_list}\t{count}")
    else:
        continue