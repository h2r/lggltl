import numpy as np
from matplotlib import pyplot as plt

'''
For generalization experiments, training samples varied between 300-3000
'''


color_ls = [[240, 163, 255], [113, 113, 198], [197, 193, 170],
            [113, 198, 113], [85, 85, 85], [198, 113, 113],
            [142, 56, 142], [125, 158, 192], [184, 221, 255],
            [153, 63, 0], [142, 142, 56], [56, 142, 142]]

markers = ['o', 's', 'D', '^', '*', '+', 'p', 'x', 'v', '|']
colors = [[shade / 255.0 for shade in rgb] for rgb in color_ls]


x_vals = [4, 8, 12, 16, 20, 24, 28, 32, 36]
gen_data = [[0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0169851380042, 0.0, 0.0379924953096],
            [0.181196581197, 0.0606557377049, 0.12735088662],
            [0.0352941176471, 0.037691401649, 0.0590476190476],
            [0.0270083102493, 0.131381381381, 0.304507819687],
            [0.452852153667, 0.287846481876, 0.426108374384],
            [0.666666666667, 0.6484375, 0.69779286927],
            [0.638069705094, 0.180327868852, 0.67374005305]]
data = np.array(gen_data)

plt.rcParams['legend.loc'] = 'best'
num_samples = data.shape[-1]
means = np.mean(data, axis=-1)
stds = np.std(data, axis=-1)
# plt.plot(x_vals, means, color=colors[0], marker='o')
# top = np.add(means, stds * (1.96 / np.sqrt(num_samples)))
# bot = np.subtract(means, stds * (1.96 / np.sqrt(num_samples)))
# plt.fill_between(x_vals, top, bot, facecolor=colors[0], edgecolor=colors[0], alpha=0.25)
plt.errorbar(x_vals, means, yerr=stds * (1.96 / np.sqrt(num_samples)), fmt='o-', ecolor=colors[0])

plt.legend()
plt.xlim(0, 40)
plt.ylim(0.0, 1.0)
plt.xlabel('# of GLTL Formulas Seen During Training')
plt.ylabel('Unseen GLTL Formula Grounding Accuracy')
plt.grid(True)
plt.show()
