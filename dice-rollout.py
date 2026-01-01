import numpy as np
from numba import njit, prange
import time

NUM_DICE = 6
NUM_SIDES = 6
TARGET_FACE_VALUE = 6 
NUM_TRIALS = 1000000

@njit
def rollout():
    non_goal_remaining = NUM_DICE
    rounds = 0

    while non_goal_remaining > 0:
        successes = 0
        for _ in range(non_goal_remaining):
            roll = np.random.randint(1, NUM_SIDES + 1)
            if roll == TARGET_FACE_VALUE:
                successes += 1
        rounds += 1
        non_goal_remaining -= successes

    return rounds

@njit(parallel=True)
def monte_carlo(trials):
    totals = np.zeros(trials, dtype=np.int64)
    for i in prange(trials):
        totals[i] = rollout()
    return totals.sum() / trials

if __name__ == "__main__":
    rollout()  # Warm-up call for JIT compilation
    monte_carlo(1)  # Warm-up call for JIT compilation

    start_time = time.time()
    average = monte_carlo(NUM_TRIALS)
    elapsed_time = time.time() - start_time
    
    print(f"Average of {average} rounds ({NUM_TRIALS} trials, {elapsed_time:.4f} sec)")