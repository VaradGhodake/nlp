import nltk
import os

if not os.path.exists('./goodwords.txt'):
    os.system("aspell -d en dump master | aspell -l en expand > ./goodwords.txt")


good_words = set()
with open('./goodwords.txt') as f:
    for line in f:
        good_words.add(line.strip())

with open('/usr/lib/aspell/split.kbd') as f:
    bichars = set()
    for line in f:
        line = line.strip()
        if len(line) == 2:
            bichars.add(line)


def candidates(word, bichars=bichars):
    if bichars is None:
        bichars = set()
    for pos in range(len(word) + 1):
        for alpha in [chr(x) for x in range(ord('a'), ord('z') + 1)]:
            mod = word[:pos] + alpha  + word[pos:]
            if (mod[pos:pos + 2] in bichars or
                    mod[pos: pos + 2][::-1] + alpha in bichars):
                yield mod, 0.6
            else:
                yield mod, 1
    for pos in range(len(word)):
        yield word[:pos] + word[pos + 1:], 2
    for pos in range(len(word)):
        for alpha in [chr(x) for x in range(ord('a'), ord('z') + 1)]:
            yield word[:pos]  + alpha + word[pos + 1:], 1

while True:
    line = input().strip()
    for word in line.split():
        if word not in good_words:
            choices = set()
            possibilities = []
            for new_word, distance in candidates(word):
                if new_word in good_words and new_word not in choices:
                    possibilities.append((new_word, distance))
            possibilities = sorted(possibilities, key=lambda x: x[1])
            print(possibilities)

