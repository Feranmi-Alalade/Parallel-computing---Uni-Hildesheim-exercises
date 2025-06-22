import sys

for line in sys.stdin:
    if not line:
        continue
    line = line.strip()

    inv_links, word = line.split("\t")
    inv_links = int(inv_links)
    links = 100000 - inv_links

    print(f"{word}\t{links}")