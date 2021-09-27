import csv
import pandas
import matplotlib.pyplot as plt
from loadPreProc import *
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from numpy import arange

textDF=pandas.read_csv('TIST_files/classwise_performance.csv')
Categories = ["%d. %s" % (i,textDF['label'][i]) for i in range(23)]

fig, ax = plt.subplots(figsize=(10,5))
textDF['best_baseline_F score'].plot.bar(width=0.3,  ylim=[0.1, 0.9], xlim=[0,22], position=2.0, color="orange", ax=ax, alpha=1)
textDF['best_proposed_F score'].plot.bar(width=0.3, position=1.0, ylim=[0, 0.9], xlim=[0,22],color="red", ax=ax, alpha=1)
ax.set_facecolor("white")
ax.set_xticklabels(range(0,23), rotation=0, fontsize=7)
ax.set_yticklabels([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],rotation=0, fontsize=7)
ax.set_xlabel("Label IDs",fontsize=8)

for i in range(23):
    plt.text(x = i-0.6 , y = max(textDF['best_baseline_F score'][i],textDF['best_proposed_F score'][i])+.0125, s = ("%.1f" % textDF["train cov"][i]), size = 7,rotation=0, bbox=dict(facecolor='lightgray', edgecolor='none',alpha=1, pad=1))

tr_c = mpatches.Patch(color='lightgray', label='Label coverage %')
best_base = mpatches.Patch(color='orange', label='F score for the best baseline')
best_proposed = mpatches.Patch(color='red', label='F score for our best method')

legend = plt.legend(handles=[tr_c,best_base,best_proposed],loc="upper right", prop={'size': 8},bbox_to_anchor=(1.0054,1.010))
extra = Rectangle((0, 0), 0.5, 0.5, fc="w", fill=False, linewidth=0)
legend1 = plt.legend([extra]*(len(textDF['label'])), Categories, loc = (1,1), fontsize='extralarge', framealpha=0, handlelength=0, handletextpad=0, prop={'size': 9.4},bbox_to_anchor=(1.017,-0.070))
plt.gca().add_artist(legend)
plt.gca().add_artist(legend1)


# ax.legend(handles=[tr_c,best_base,best_proposed],loc="upper right", prop={'size': 7},bbox_to_anchor=(1.0035,1.007))
plt.savefig('TIST_files/class_wise_performance.pdf',bbox_extra_artists=(ax,), bbox_inches='tight')