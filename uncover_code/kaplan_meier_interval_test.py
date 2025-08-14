from lifelines.fitters.npmle import NPMLEFitter
import numpy as np

# Example data with interval censoring (left and right bounds)
# For exact events, left == right
# For right-censored, right = np.inf
# For left-censored, left = 0
data = [
    (0, 5),    # Interval censored (0, 5]
    (10, 10),  # Exact event at 10
    (15, np.inf), # Right censored at 15
    (7, 12)    # Interval censored (7, 12]
]

left = [d[0] for d in data]
right = [d[1] for d in data]

npmle = NPMLEFitter()
npmle.fit_interval_censoring(left, right)

# Get the estimated survival function
survival_function = npmle.survival_function_
print(survival_function)