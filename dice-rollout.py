import numpy as np
from numba import njit, prange
import time

NUM_DICE = 6
NUM_SIDES = 6
TARGET_FACE_VALUE = 6 
DEFAULT_TRIALS = 1000000
SEED = None

@njit
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

@njit(parallel=True)
def monte_carlo(trials=DEFAULT_TRIALS):
    total_rolls = 0
    for _ in prange(trials):
        total_rolls += rollout()
    average_rolls = total_rolls / trials
    return average_rolls

if __name__ == "__main__":
    if SEED is not None:
        np.random.seed(SEED)
    
    start_time = time.time()
    average = monte_carlo()
    elapsed_time = time.time() - start_time
    
    print(f"Calculated average of {average} in {elapsed_time:.4f} seconds")