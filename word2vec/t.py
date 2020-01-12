import random

if __name__ == "__main__":
    target = 2
    targets_to_avoid = [1,2,3,4,1,1]
    while target in targets_to_avoid:
        target = random.randint(0, 5 - 1)
        print(target)