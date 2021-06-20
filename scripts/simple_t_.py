from src import model
from src.__main__ import main, load_data
import math

import sys
import os

sys.path.append(os.path.abspath('./src/'))
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


viz = True

# environment definition
traps = []

# data stuff
true_goal = 'A'

print(os.path.abspath("src/tests/data/"))

data_path, file_name, goals, pix = "tests/data/", "tests/data/testsimple.csv", {
    'A': (
        3, 3), 'B': (
            3, 2)}, 1
df, arena_dims, trajectory = load_data(data_path, file_name)
print(arena_dims)

goals = {g: (int(math.floor(coords[0] / pix)),
             int(math.floor(coords[1] / pix))) for g,
         coords in goals.items()}
goals = [model.Goal(name, pos) for name, pos in goals.items()]


main(arena_dims, goals, traps, df, trajectory, viz={'goal': (3, 3)})
