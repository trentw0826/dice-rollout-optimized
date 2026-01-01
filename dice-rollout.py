import numpy as np

NUM_DICE = 6
NUM_SIDES = 6
TARGET_FACE_VALUE = 6 
DEFAULT_TRIALS = 10000

def rollout():
    non_goal_remaining = NUM_DICE
    rolls = 0

    while non_goal_remaining > 0:
        successes = 0
        for _ in range(non_goal_remaining):
            roll = np.random.randint(1, NUM_SIDES + 1)
            rolls += 1
            if roll == TARGET_FACE_VALUE:
                successes += 1
        non_goal_remaining -= successes

    return rolls


def monte_carlo(trials=DEFAULT_TRIALS):
    total_rolls = 0
    for _ in range(trials):
        total_rolls += rollout()
    average_rolls = total_rolls / trials
    return average_rolls

if __name__ == "__main__":
    average = monte_carlo()
    print(f"Average number of rolls to get all sixes: {average}")
    