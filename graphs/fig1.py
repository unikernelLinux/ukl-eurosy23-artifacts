import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

labels = ['getppid', 'read', 'write', 'sendto', 'recvfrom']
Linux = [342.62, 614.54, 660.16, 739.17, 649.55]
UKL = [325.26, 599.84, 647.49, 706.10, 621.5]
UKL_Bypass = [56.55, 311.65, 391.02, 462.83, 408.97]

UKL_p = np.round([((Linux[i] - UKL[i]) / Linux[i]) * 100 for i in range(len(UKL))],2)
UKL_Bypass_p = np.round([((Linux[i] - UKL_Bypass[i]) / Linux[i]) * 100 for i in range(len(UKL_Bypass))],2)

x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(5,3), dpi= 100, facecolor='w', edgecolor='k')
rects1 = ax.bar(x - width, Linux, width, label='Linux')
rects2 = ax.bar(x, UKL, width, label='UKL')
rects3 = ax.bar(x + width, UKL_Bypass, width, label='UKL_Bypass')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Time (ns)', fontsize = 12, color = 'black')
ax.set_xticks(x, labels, fontsize = 12, color = 'black')
ax.set_ylim(0,900)
ax.legend(bbox_to_anchor=(0.01,.78,0.98,0.2), mode="expand", borderaxespad=0, ncol=3)
ax.grid(axis='y')
ax.set_axisbelow(True)

for i in range(len(UKL)):
    height = rects2[i].get_height()
    ax.text(rects2[i].get_x() + 3*rects2[i].get_width()/3., 1.02*height,
            '%d' % UKL_p[i] + "%", ha='center', va='bottom')
    
for i in range(len(UKL)):
    height = rects3[i].get_height()
    ax.text(rects3[i].get_x() + 3*rects3[i].get_width()/3., 1.02*height,
            '%d' % UKL_Bypass_p[i] + "%", ha='center', va='bottom')


fig.tight_layout()

plt.show()

fig.savefig('syscall.pdf')
