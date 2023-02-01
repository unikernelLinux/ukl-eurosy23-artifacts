import pandas as pd
import numpy as np
import math
import scipy.stats as st
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import MultipleLocator

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

kdirs = ["Linux", 
        "ukl",  
        "ukl-ret-bp"]

kdirs2 = ["Linux", 
        "ukl",   
        "ukl-ret-bp"]

lsizes = np.arange(256, 8193, 256)
lsizes = np.insert(lsizes, 0 ,1)

colors = ['blue', 'orange', 'green', 'purple', 'teal', 'red']
xtix = np.arange(0, 8193, 2048)

np.set_printoptions(suppress=True)

def singlelatencyline(testname):
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
        elif d == 'ukl-ret-bp':
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


def plotdataline(pd, pArr, ax, title, pos, yt):
    tmax = 0
    for d in kdirs:
        pdval = pd[d]
        pdlow = pd[d+"_low"].to_numpy()
        pdhi = pd[d+"_hi"].to_numpy()
        max = np.amax(pdval)
        if tmax < max:
            tmax = max
        ax.plot(lsizes, pdval, label=d, linewidth=1.5)#, \
                 #color=colors[kdirs.index(d)])
    tmax = math.ceil(tmax*2)/2
    ax.set_xlim(xmin=0)
    ax.set_ylim(ymin=0)
    #ax.legend()
    ax.set_ylim(ymax=tmax)
    ytix_steps = 100/yt
    ytix = np.arange(0, 101, ytix_steps)
    ax.set_title(title, fontsize = 12, color = 'black')
    ax.set_xlim(xmin=0)
    if pos == -1:
        ax.set_xlabel("Bytes", fontsize = 12, color = 'black')
    ax.set_ylabel("Time (us)", fontsize = 12, color = 'black')
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmax=8200)
    ax.set_xticks(xtix)
    if pos == 1:
        ax.legend(labels=['Linux','UKL','UKL_Bypass'],bbox_to_anchor=(0,1.2,1,0.2),loc="lower left",
            mode="expand", borderaxespad=0, ncol=3)
    ax.grid()
    ax2 = ax.twinx()
    ax2.fill_between(lsizes, pArr, label = 'Improvement', \
         color = 'darkorange', alpha=0.3, facecolor='darkorange')
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.set_ylim(0,100)
    ax2.set_yticks(ytix)
    ax2.set_ylabel('UKL_Bypass Improv.',  fontsize=8)
    ax2.set_zorder(ax.get_zorder()-1)
    
    ax.text(lsizes[0]+370, (pArr[0]/100)*tmax,'%d' % pArr[0] + "%", ha='center', va='bottom')
    ax.text(lsizes[8], (pArr[8]/100)*tmax,'%d' % pArr[8] + "%", ha='center', va='bottom')
    ax.text(lsizes[16], (pArr[16]/100)*tmax,'%d' % pArr[16] + "%", ha='center', va='bottom')
    ax.text(lsizes[24], (pArr[24]/100)*tmax,'%d' % pArr[24] + "%", ha='center', va='bottom')
    ax.text(lsizes[32]-320, (pArr[32]/100)*tmax,'%d' % pArr[32] + "%", ha='center', va='bottom')
    
    ax.patch.set_visible(False)
    ax.set_axisbelow(True)
    #ax.text(1000,2.47,'Read', fontsize = 12, color = 'black', \
    #    bbox=dict(facecolor='white', edgecolor='black', pad=10.0))
    
def ksinglelatencyline(testname):
    col_names = ['Size']
    final_matrix = np.expand_dims(lsizes, axis=1)
    for d in kdirs2:
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
        elif d == 'ukl-ret-bp':
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

def kplotdataline(pd, pArr, ax, title, pos, yt):
    tmax = 0
    for d in kdirs2:
        pdval = pd[d]
        pdlow = pd[d+"_low"].to_numpy()
        pdhi = pd[d+"_hi"].to_numpy()
        max = np.amax(pdval)
        if tmax < max:
            tmax = max
        ax.plot(lsizes, pdval, label=d, linewidth=1.5)#, \
                 #color=colors[kdirs.index(d)])
    tmax = math.ceil(tmax*2)/2
    ax.set_xlim(xmin=0)
    ax.set_ylim(ymin=0)
    #ax.legend()
    ax.set_ylim(ymax=tmax)
    ytix_steps = 100/yt
    ytix = np.arange(0, 101, ytix_steps)
    ax.set_title(title, fontsize = 12, color = 'black')
    ax.set_xlim(xmin=0)
    if pos == -1:
        ax.set_xlabel("Bytes", fontsize = 12, color = 'black')
    ax.set_ylabel("Time (us)", fontsize = 12, color = 'black')
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmax=8200)
    ax.set_xticks(xtix)
    if pos == 1:
        ax.legend(bbox_to_anchor=(0,1.17,1,0.2),loc="lower left",
            mode="expand", borderaxespad=0, ncol=3)
    ax.grid()
    ax2 = ax.twinx()
    ax2.fill_between(lsizes, pArr, label = 'Improvement', \
         color = 'darkorange', alpha=0.3, facecolor='darkorange')
    ax2.set_ylim(0,100)
    ax2.set_yticks(ytix)
    ax2.set_ylabel('UKL_Bypass Improv.',  fontsize=8)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.set_zorder(ax.get_zorder()-1)
        
    ax.text(lsizes[0]+370, (pArr[0]/100)*tmax,'%d' % pArr[0] + "%", ha='center', va='bottom')
    ax.text(lsizes[8], (pArr[8]/100)*tmax,'%d' % pArr[8] + "%", ha='center', va='bottom')
    ax.text(lsizes[16], (pArr[16]/100)*tmax,'%d' % pArr[16] + "%", ha='center', va='bottom')
    ax.text(lsizes[24], (pArr[24]/100)*tmax,'%d' % pArr[24] + "%", ha='center', va='bottom')
    ax.text(lsizes[32]-320, (pArr[32]/100)*tmax,'%d' % pArr[32] + "%", ha='center', va='bottom')
    
    ax.patch.set_visible(False)
    ax.set_axisbelow(True)
    #ax.text(1000,2.47,'Read', fontsize = 12, color = 'black', \
    #    bbox=dict(facecolor='white', edgecolor='black', pad=10.0))


fig, [ax1, ax2, ax3, ax4] = plt.subplots(4,1,figsize=(5,8), dpi= 100, facecolor='w', \
                       edgecolor='k')

pd1, pArr1 = singlelatencyline("read")
plotdataline(pd1, pArr1, ax1, "read() Latency", 1, 4)

pd1, pArr1 = singlelatencyline("write")
plotdataline(pd1, pArr1, ax2, "write() Latency", 0, 3)

pd1, pArr1 = ksinglelatencyline("send")
kplotdataline(pd1, pArr1, ax3, "sendto() Latency", 0, 3)

pd1, pArr1 = ksinglelatencyline("recv")
kplotdataline(pd1, pArr1, ax4, "recvfrom() Latency", -1, 3)

fig.tight_layout()
fig.savefig('read-write-send-recv.pdf')
