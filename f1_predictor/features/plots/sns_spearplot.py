"""
Visualizes relationships between features in the F1 training dataset using Seaborn pairplot.

This script loads training data, samples a subset of it, and creates a pairplot
colored by finishing position to explore patterns across features.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt 

current_dir = os.path.dirname(os.path.abspath(__file__))
parrent_dir = os.path.dirname(current_dir)
f1_dir = os.path.dirname(parrent_dir)

training_data = pd.read_csv(os.path.join(f1_dir, 'data', 'train_data.csv'))

import seaborn as sns

sampled = training_data.sample(1000, random_state=42)

sns.pairplot(sampled, hue='finishing_position')
plt.show()