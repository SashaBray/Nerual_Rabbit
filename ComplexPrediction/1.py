import matplotlib.pyplot as plt

x = [2002, 2003, 2004, 2005, 2006]
y = [30, 32, 34, 28, 31]

a = [2002, 2003, 2004, 2005, 2006, 2007]
b = [0.29, 0.3, 0.31, 0.27, 0.3, 0.25]

fig, ax1 = plt.subplots(figsize=(12, 6))        # График урожайности и пика NDVI
ax2 = ax1.twinx()
ax1.plot(x, y, label=' NDVI ', color = 'blue')
ax2.plot(a, b, label=' Урожайность ', color='darkred')

fig.legend(bbox_to_anchor=(0.9, 0.88))


# fig.legend(fontsize=14,  framealpha=1, title='Легенда')

ax1.set_ylabel('NDVI', color = 'blue', size = 11)
ax2.set_ylabel('Урожайность (ц/га)', color='darkred', size = 11)
plt.title('NDVI/Урожайность по годам - ', size = 14)
plt.grid(True)
plt.show()