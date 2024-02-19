import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np

# Collect Data
data = pd.read_csv("Fish.csv")

data = sns.load_dataset("iris")
# What is the distribution of the target variable(Weight) with respect to fish Species?
# sns.displot(
#   data=data,
#   x="Weight",
#   hue="Species",
#   kind="hist",
#   height=6,
#   aspect=1.4,
#   bins=15
# )

plt.show()
sns.pairplot(data, kind='scatter', hue='Species')