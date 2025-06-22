import sys

for line in sys.stdin:
    if not line:
        continue
    line = line.strip()

    word, links = line.split("\t")
    links = int(links)

    inv_links = 100000 - links

    print(f"{inv_links}\t{word}")