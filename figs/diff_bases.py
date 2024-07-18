from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from sympy import primefactors

from spm import ANALYSIS_DIR


def num_unique_prime_factors(n):
    return len(primefactors(n))


# Which bases do we have? All files of the format bBASE_sSEED.txt in logs/heatmap_scores
bases_seeds = []
path = ANALYSIS_DIR / 'diffbases_scores'
for f in path.glob('b*_s*.txt'):
    b_str, s_str = f.stem.split('_')
    base = int(b_str[1:])
    seed = int(s_str[1:])
    bases_seeds.append((base, seed))
# Load the scores from the files.
bs_to_verifiability = {}
bs_to_k = {}
for base, seed in bases_seeds:
    with open(path / f'b{base}_s{seed}.txt') as f:
        # first line is verifiability score, second line is k score
        verifiability_score = float(f.readline().strip())
        correctness_score = float(f.readline().strip())
        bs_to_verifiability[(base, seed)] = verifiability_score
        bs_to_k[(base, seed)] = correctness_score

verifiability = defaultdict(list)
correctnesss = defaultdict(list)
for b, s in bs_to_verifiability:
    verifiability[num_unique_prime_factors(b)].append(bs_to_verifiability[(b, s)])
    correctnesss[num_unique_prime_factors(b)].append(bs_to_k[(b, s)])

score_to_draw = verifiability  # change to correctness if desired

# Data from the table
# Take mean_verifiability from the df
names = [r'$\omega(B)=1$', r'$\omega(B)=2$', r'$\omega(B)=3$', r'$\omega(B)=4$']
keys_sorted = [1, 2, 3, 4]
assert set(keys_sorted) == set(score_to_draw.keys())
means = [np.mean(score_to_draw[key]) for key in keys_sorted]
stderrs = [np.std(score_to_draw[key]) / np.sqrt(len(score_to_draw[key])) for key in keys_sorted]

# print mean, stderr and number of samples for each base
for key, mean, stderr in zip(keys_sorted, means, stderrs):
    print(f'Base {key}: mean={mean:.3f}, stderr={stderr:.3f}, n={len(score_to_draw[key])}')

# Paul Tol's color scheme: colorblind and monochrome-friendly <3
colors = ['#DDAA33', '#BB5566', '#004488', '#000000']

# Create a figure and axis
fig, ax = plt.subplots(figsize=(8, 2))  # Adjust figsize to make the plot more flat

# Plot the means with error bars on the same horizontal line (y=0)
y_position = 0
for i, (exp, mean, stderr, color) in enumerate(zip(names, means, stderrs, colors)):
    ax.errorbar(mean, y_position, xerr=stderr, fmt='o', color=color, capsize=5)
    ax.text(mean, y_position + 0.02, exp, ha='center', color=color)

# Setting labels and title
ax.set_xlabel('Verifiability')
ax.set_yticks([])  # Remove y-axis ticks
ax.set_ylim(-0.05, 0.1)  # Set y-axis limits to make the plot more compact

# Percentages on the x-axis
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

# Save space and save
# plt.tight_layout(pad=0.5)
plt.subplots_adjust(left=0.1, right=0.9, top=0.5, bottom=0.3)
# No box around the plot, only x axis
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
FORMAT = 'eps'
plt.savefig(f'diffbases.{FORMAT}', dpi=1200, bbox_inches='tight', format=FORMAT)
plt.show(bbox_inches='tight')
