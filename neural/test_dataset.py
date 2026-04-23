import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Proveri putanju koju ti je dao 'find' komanda
df = pd.read_csv('dataset.csv')

# Analiziramo samo ono što je bitno za tvoj model
numeric_df = df.select_dtypes(include=['float64', 'int64'])
corr = numeric_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr[['popularity']].sort_values(by='popularity', ascending=False),
            annot=True, cmap='viridis')
plt.title("Factor influence on popularity (Target Analysis)")
plt.show()