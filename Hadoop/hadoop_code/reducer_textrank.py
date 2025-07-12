import sys

current_word = None
current_links = 0
threshold = 100

for line in sys.stdin:
    if not line:
        continue
    line = line.strip()

    word, links = line.split("\t")
    word = word.strip()
    links = int(links)

    if word == current_word:
        current_links += links

    else:
        if current_word is not None and current_links >= threshold:
            # print only words with links greater than or equal to threshold
            print(f"{current_word}\t{current_links}")

        current_word = word
        current_links = links

if current_word is not None and current_links >= threshold:
    print(f"{current_word}\t{current_links}")