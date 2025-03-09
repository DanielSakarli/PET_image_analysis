import numpy as np

plasma_WB_ratio = [1.26, 1.75, 1.62, 1.79, 1.76, 1.86, 1.85]

# Calculate the median using NumPy
median = np.median(plasma_WB_ratio)

print(f"The median is: {median}")