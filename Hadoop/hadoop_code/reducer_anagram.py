import sys

current_anagram = None
word_list = []

for line in sys.stdin:
    if not line:
        continue
    line = line.strip()

    anagram, word = line.split("\t")
    word = word.strip()
    anagram = anagram.strip()

    if anagram == current_anagram:
        if word in word_list:
            continue
        else:
            word_list.append(word)

    else:
        if current_anagram is not None:
            print(f"{current_anagram}\t{word_list}")

        current_anagram = anagram
        word_list = [word]

if current_anagram is not None:
    print(f"{current_anagram}\t{word_list}")