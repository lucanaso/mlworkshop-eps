# Author: Luca Naso
# Creation date: 08 May 2021
# Topic: live coding during Machine Learning Workshop for EPS
#        a simple example of linear regression with synthetic data
#

import numpy as np
import matplotlib.pyplot as plt

obs_number = 200

# features
X = 1 + 2 * np.random.random(obs_number)

# targets
a = 3.5
b = 8
fluctuations = 0.2
y = b + a * X + fluctuations * np.random.randn(obs_number)

# plot
plt.plot(X, y, '+')
plt.show()

pass