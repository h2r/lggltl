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


# x_vals = [4, 8, 12, 16, 20, 24, 28, 32, 36]
x_vals = np.array(range(1, 10)) * 0.1
gen_data = [[0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0169851380042, 0.0, 0.0379924953096],
            [0.181196581197, 0.0606557377049, 0.12735088662],
            [0.0352941176471, 0.037691401649, 0.0590476190476],
            [0.0270083102493, 0.131381381381, 0.304507819687],
            [0.452852153667, 0.287846481876, 0.426108374384],
            [0.666666666667, 0.6484375, 0.69779286927],
            [0.638069705094, 0.180327868852, 0.67374005305]]
gen_data = np.array(gen_data)

eff_data = [[0.842312746386, 0.924981522542, 0.933699324324, 0.919704433498, 0.950325251331, 0.958610495196, 0.945812807882, 0.932053175775, 0.967551622419],
            [0.841655716163, 0.937915742794, 0.949746621622, 0.905418719212, 0.940863394441, 0.964523281596, 0.948768472906, 0.970457902511, 0.94395280236],
            [0.861366622865, 0.912416851441, 0.914273648649, 0.944827586207, 0.963335304554, 0.966001478197, 0.907389162562, 0.93353028065, 0.955752212389]]
eff_data = np.array(eff_data)
eff_data = np.transpose(eff_data)
data = eff_data
print data.shape

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
# plt.xlim(0, 40)
plt.xlim(0.0, 1.0)
plt.ylim(0.8, 1.0)
#plt.xlabel('# of GLTL Formulas Seen During Training')
plt.xlabel('% of Training Data Utilized')
#plt.ylabel('Unseen GLTL Formula Grounding Accuracy')
plt.ylabel('Held-Out Data Grounding Accuracy')
plt.grid(True)
plt.show()
