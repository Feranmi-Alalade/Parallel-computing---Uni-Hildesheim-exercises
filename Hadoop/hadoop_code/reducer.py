import sys

current_word = None
current_count = 0

# input comes from STDIN
for line in sys.stdin:
    line = line.strip()
    word, count = line.split()

    try:
        count = int(count)
    except ValueError:
        continue

    if current_word == word:
        current_count += count
    else:
        if current_word:
            # write result to STDOUT
            print(f'{current_word}\t{current_count}')
        current_word = word
        current_count = count

# don't forget the last word
if current_word:
    print(f'{current_word}\t{current_count}')
