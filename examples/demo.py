"""Run this to see the report. No GPU needed - it simulates work."""

import time
import random
from gpu_cost_meter import meter, report


@meter("predict-endpoint")
def predict(x):
    # Simulate inference work
    time.sleep(random.uniform(0.01, 0.05))
    return x * 2


@meter("batch-endpoint")
def batch_predict(items):
    time.sleep(random.uniform(0.05, 0.15))
    return [i * 2 for i in items]


if __name__ == "__main__":
    print("Running demo inferences...")
    for i in range(50):
        predict(i)
    for _ in range(10):
        batch_predict([1, 2, 3, 4, 5])
    report()