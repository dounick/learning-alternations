import pandas as pd
import scipy.stats as stats
import numpy as np
import os

df = pd.read_csv('results.csv')

df = df[df['verb_count'] >= 1]
df['BehavDOpreference'] = df['BehavDOpreference']/100
df = df.groupby('verb_id')[['bert_ratio', 'BehavDOpreference', 'ngram_ratio', 'lstm_ratio', 'gpt2_ratio', 'gpt2-large_ratio', 'lstm-large_ratio', 'balanced_ratio', 'default_ratio', 'loose_balanced_ratio', 'loose_default_ratio']].mean().reset_index()
spearmans = [
    stats.spearmanr(df['BehavDOpreference'], df['balanced_ratio']),
    stats.spearmanr(df['BehavDOpreference'], df['default_ratio']),
    stats.spearmanr(df['BehavDOpreference'], df['loose_balanced_ratio']),
    stats.spearmanr(df['BehavDOpreference'], df['loose_default_ratio'])
]
models = ['balanced_ratio', 'default_ratio', 'loose_balanced_ratio', 'loose_default_ratio']
import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
axes = axes.flatten()

x = df['BehavDOpreference']
for i, model in enumerate(models):
    y = df[model]
    colors = ['blue' if verb_id <= 100 else 'red' for verb_id in df['verb_id']]
    axes[i].scatter(x, y, c=colors, label=['non-alternating' if verb_id <= 100 else 'alternating' for verb_id in df['verb_id']])
    axes[i].set_title(model.replace('_ratio', ''))
    axes[i].set_xlabel('BehavDOpreference')
    axes[i].set_ylabel(model.replace('_ratio', ''))
    axes[i].set_xticks([0.2, 0.4, 0.6])
    axes[i].set_ylim(-2.5, 0)

    # Calculate and plot the best fit line
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    best_fit_line = slope * x + intercept
    axes[i].plot(x, best_fit_line, color='black')
    axes[i].text(0.95, 0.05, r"$\rho$" + f' = {spearmans[i][0]:.2f}', 
                 verticalalignment='bottom', horizontalalignment='right', 
                 transform=axes[i].transAxes, color='black', fontsize=12)

# Create a legend for the colors
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='non-alternating'),
           plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='alternating')]
fig.legend(handles=handles, loc='upper right')

plt.tight_layout()
plt.show()
