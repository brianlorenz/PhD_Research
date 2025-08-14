

from lifelines import KaplanMeierFitter
import pandas as pd
import numpy as np

# Sample data: 'T' is the time to event or censoring, 'E' is the event indicator (1 if event, 0 if censored)
data = {
    'T': [10, 20, 30, 40, 50, 60, 70, 80, 10, 10, 10, 10],
    'E': [1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1]  # 0 for censored, 1 for event
}
df = pd.DataFrame(data)

# Create a KaplanMeierFitter object
kmf = KaplanMeierFitter()

# Fit the model with time and event status
kmf.fit(durations=df['T'], event_observed=df['E'])

# Print the survival probabilities
print("Survival Probabilities:")
print(kmf.survival_function_)

# Plot the Kaplan-Meier curve
kmf.plot_survival_function()
import matplotlib.pyplot as plt
plt.title('Kaplan-Meier Curve with Censored Data')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.show()