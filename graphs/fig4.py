import pandas as pd
import numpy as np
import math
import scipy.stats as st
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

kdirs = ["Linux", 
        "ukl", 
        "ukl-ist", 
        "ukl-ret"]

lsizes = np.arange(4096, 409601, 4096)

colors = ['blue', 'orange', 'green', 'purple', 'teal', 'red']
xtix = np.arange(4096, 409601, 4096)
xtixlabels = np.arange(1, 101, 1)

np.set_printoptions(suppress=True)

def ksinglelatencyline(testname):
    col_names = ['Size']
    final_matrix = np.expand_dims(lsizes, axis=1)
    for d in kdirs:
        colArr = np.empty((0,3), float)
        filename = d + "/new_lebench_" + testname + ".csv"
        tmp = pd.read_csv(filename)
        for i in np.nditer(lsizes):
            col = tmp.loc[tmp["Size"] == i, "Latency"]
            col_len = len(col)
            col = col.mul(1000000)
            cim = st.norm.interval(alpha=0.95, loc=np.mean(col), scale=st.sem(col))
            rowArr = np.array([[np.mean(col),
                                np.mean(col)-cim[0], 
                                cim[1]-np.mean(col)
                               ]])
            colArr = np.append(colArr, rowArr, axis=0)
        if d == 'Linux':
            lArr = colArr[:,0]
        elif d == 'ukl-ret':
            uArr = colArr[:,0]
        final_matrix = np.append(final_matrix, colArr, axis=1)
        col_names.append(d)
        col_names.append(d+'_low')
        col_names.append(d+'_hi')
    dArr = lArr - uArr
    pArr = (dArr/lArr)*100
    out = pd.DataFrame(final_matrix,columns=col_names)
    out.to_csv(testname+'.csv')
    return out, pArr

def kplotpctgline(mpd, pArr, ax, title, pos, yt):
    tmax = 0
    linux = mpd['Linux']
    ptgs = pd.DataFrame()
    mdirs = ["ukl", "ukl-ist", "ukl-ret"]
    ldirs = ["UKL_PF_DF", "UKL_PF_SS", "UKL_RET_PF_DF"]
    for d in mdirs:
        pdval = mpd[d]
        max = np.amax(pdval)
        if tmax < max:
            tmax = max
        rowptgs = pd.DataFrame((linux-pdval)*100/linux)
        rowptgs.columns=[d]
        ptgs = pd.concat([ptgs, rowptgs], axis=1)
    for d in mdirs:
        temppd = ptgs[d]
        ax.plot(lsizes, temppd, label=d, linewidth=2)
#         if d == "ukl-ist":
#             ax.text(lsizes[-1]-100000, temppd.iloc[-1]-2, ldirs[mdirs.index(d)], ha='center', va='bottom')
#         else:
#             ax.text(lsizes[-1]-100000, temppd.iloc[-1]+0.5, ldirs[mdirs.index(d)], ha='center', va='bottom')
#     print(ptgs)
    tmax = math.ceil(tmax*2)/2
    ax.set_ylim(ymin=0)
    #ax.legend()
    ax.set_ylim(ymax=tmax)
    ytix_steps = 100/yt
    ytix = np.arange(0, 101, ytix_steps)
#     ax.set_title(title, fontsize = 12, color = 'black')
    ax.set_xlabel("No. of Stack Pagefaults", fontsize = 12, color = 'black')
    ax.set_ylabel("Improvement over Linux", fontsize = 12, color = 'black')
    ax.set_ylim(ymax=15)
    ax.set_xlim(xmax=409600)
    ax.set_xlim(xmin=0)
    xt = np.arange(0, 409601, 40960)
    xlabels = np.arange(0, 101, 10)
    ax.set_xticks(xt)
    ax.set_xticklabels(xlabels)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(labels=ldirs,bbox_to_anchor=(0.55,0.4,0.44,0.2),loc="lower left",
            mode="expand", borderaxespad=0)
    ax.grid(axis='y')

fig, ax1 = plt.subplots(1,1,figsize=(5,2.5), dpi= 100, facecolor='w', \
                       edgecolor='k')

pd1, pArr1 = ksinglelatencyline("pagefault")
kplotpctgline(pd1, pArr1, ax1, "Pagefault Latency", 1, 4)

fig.tight_layout()
