import pandas as pd
import numpy as np
import math
import scipy.stats as st
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import MultipleLocator
np.set_printoptions(suppress=True)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

values = pd.read_csv('MemcachedResults.csv')
Linux_qps = values['QLinux']
UKL_qps = values['QUKL']
UKL_BP_qps = values['QUKL_BP']
UKL_BP_SC_qps = values['QUKL_BP_SC']
Ref_qps = [50000, 90000]

Linux_lat = values['LLinux']
UKL_lat = values['LUKL']
UKL_BP_lat = values['LUKL_BP']
UKL_BP_SC_lat = values['LUKL_BP_SC']
Ref_lat = [500, 500]

fig, ax1 = plt.subplots(1,figsize=(5,3), dpi= 100, facecolor='w', edgecolor='k')
plt.plot(Ref_qps, Ref_lat,linestyle='dashed')#, label="500 us SLA")
plt.plot(Linux_qps, Linux_lat, label="Linux")
plt.plot(UKL_qps, UKL_lat, label="UKL_RET")
plt.plot(UKL_BP_qps, UKL_BP_lat, label="UKL_RET_BYP")
plt.plot(UKL_BP_SC_qps, UKL_BP_SC_lat, label="UKL_RET_BYP (shortcut)")
ax1.set_xlim(xmin=60000, xmax=85000)
ax1.set_ylim(ymin=0.8, ymax=1000)
ax1.legend(fontsize=9)#bbox_to_anchor=(0.5, 1.5, 0.1, 0.1))#, ncol=3)
ax1.text(61000, 430, "500us SLA", fontsize=9)
ax1.set_xlabel("Querries per Second")
ax1.set_ylabel("Latency (us)")
ax1.grid(True)
fig.tight_layout()
plt.savefig('memcached.pdf')
