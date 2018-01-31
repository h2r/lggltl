import sys
import numpy as np
from matplotlib import pyplot as plt

'''
For generalization experiments, training samples varied between 300-3000
'''

MODE = int(sys.argv[1])

color_ls = [[240, 163, 255], [113, 113, 198], [197, 193, 170],
            [113, 198, 113], [85, 85, 85], [198, 113, 113],
            [142, 56, 142], [125, 158, 192], [184, 221, 255],
            [153, 63, 0], [142, 142, 56], [56, 142, 142]]

markers = ['o', 's', 'D', '^', '*', '+', 'p', 'x', 'v', '|']
colors = [[shade / 255.0 for shade in rgb] for rgb in color_ls]


if MODE == 0:
    x_vals = [4, 8, 12, 16, 20, 24, 28, 32, 36]
else:
    x_vals = np.array(range(1, 10)) * 0.1

# gen_data2 = [[0.0, 0.0, 0.028025477707, 0.136752136752, 0.0598465473146, 0.135041551247, 0.286379511059, 0.63184079602, 0.219839142091],
#              [0.0, 0.00120627261761, 0.00891181988743, 0.182160128963, 0.111111111111, 0.241950321987, 0.051724137931, 0.640067911715, 0.318302387268],
#              [0.0, 0.00241212956582, 0.00506427736658, 0.047131147541, 0.0583038869258, 0.0735735735736, 0.221748400853, 0.521484375, 0.559718969555],
#              [0.0, 0.00655358519661, 0.000446229361892, 0.0391705069124, 0.0168010752688, 0.100085543199, 0.201797385621, 0.143636363636, 0.171102661597],
#              [0.0,0.0,0.0167286245353,0.00111482720178,0.112345679012,0.0808356039964,0.379802414929,0.218867924528,0.131868131868],
#              [0.0,0.008014571949,0.048439683279,0.00813890396093,0.0651744568795,0.127458693942,0.314479638009,0.0282485875706,0.0643564356436]]
# gen_data2 = np.array(gen_data2)
# gen_data2 = np.transpose(gen_data2)
#
# eff_data = [[0.842312746386, 0.924981522542, 0.933699324324, 0.919704433498, 0.950325251331, 0.958610495196, 0.945812807882, 0.932053175775, 0.967551622419],
#             [0.841655716163, 0.937915742794, 0.949746621622, 0.905418719212, 0.940863394441, 0.964523281596, 0.948768472906, 0.970457902511, 0.94395280236],
#             [0.861366622865, 0.912416851441, 0.914273648649, 0.944827586207, 0.963335304554, 0.966001478197, 0.907389162562, 0.93353028065, 0.955752212389]]
# eff_data = np.array(eff_data)
# eff_data = np.transpose(eff_data)


data = gen_data2
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

if MODE == 0:
    plt.xlim(0, 40)
else:
    plt.xlim(0.0, 1.0)

plt.ylim(0.0, 1.0)

if MODE == 0:
    plt.xlabel('# of GLTL Formulas Seen During Training')
    plt.ylabel('Unseen GLTL Formula Grounding Accuracy')
else:

    plt.xlabel('% of Training Data Utilized')
    plt.ylabel('Held-Out Data Grounding Accuracy')
plt.grid(True)
plt.show()
