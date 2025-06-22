import sys
import re
import string

window_length = 4

# punctuations = set(string.punctuation)

punctuations = r'[.,!?;:\-_—()[\]{}\'"…`‘’“”/\\|@#$%^&*~+=<>-]'

stopwords = set(["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into",
"is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
"they", "this", "to", "was", "will", "with", "you", "your", "we", "he", "she", "his", "her", "i",
"me", "my"
])


for line in sys.stdin:
    if not line:
        continue
    line = line.strip()
    
    cleaned_words = []
    words = line.lower().split() # Convert to lower case

    # Iterate through line to remove stopwords and punctuation
    for word in words:
        word = re.sub(punctuations, '', word)

        if word and word not in stopwords:
            cleaned_words.append(word)

    anagram = []

    for word in cleaned_words:
        
        # letters = word.split()
        letters = [word[i] for i in range(len(word))]

        sorted_letters = sorted(letters) # Sort the letters

        sorted_letters = "".join(sorted_letters)

        if sorted_letters not in anagram:
            anagram.append(sorted_letters)
            print(f"{sorted_letters}\t{word}")

        else:
            print(f"{sorted_letters}\t{word}")


        # anagram.append(sorted(letters))


    


