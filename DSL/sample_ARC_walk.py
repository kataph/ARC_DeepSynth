import json
from random import randint
def sample_walk():
    fwalks = open("DSL\\ARC_random_walks.txt", "rt")
    # x = json.loads(fwalks.read())
    data = fwalks.read()
    # data = json.load(open("DSL\\ARC_random_walks.txt", "rt"))
    length = len(data)
    r = randint(1,length-1)
    x = ""
    f = False
    for char in data[r:]:
        if char == "[":
            f = True
        if char == "]":
            if f: x+=char; break
        if f: x+=char
    print(x)
    print(length)

sample_walk()    