import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

fig, [ax, axm, axd] = plt.subplots(3,1,figsize=(5,9), dpi= 100, facecolor='w', \
                       edgecolor='k')
#---------------------------------
tmp = pd.read_csv('Linux',delim_whitespace=True, \
                  skipfooter=3, engine='python')

cummLat = tmp['Value'].to_numpy()
lat = np.diff(cummLat, axis=0)
lat = np.insert(lat, 0, cummLat[0])
nzLat = lat[lat != 0] # Getting non zero latencies
nzCummLat = cummLat[lat != 0]

ptile = tmp['Percentile'].to_numpy()

tc = tmp['TotalCount'].to_numpy()
nzTc = tc[lat != 0]
count = np.diff(tc, axis=0)
count = np.insert(count, 0, tc[0])
nzCount = count[lat != 0] # Getting freq. for non zero latencies
total = np.sum(nzCount)

Nw = np.multiply(nzLat, total) # multiply each latency with total freq
fd = np.divide(nzCount,Nw) # divide each freq. with Nw

check = np.multiply(nzLat,fd) # Multiply each latency with prob.
#print(np.sum(check)) # this should be 1

# probability distribution
ax.bar(nzCummLat, fd, width=0.03, label = "Probablity Density", color = '#b2abd2')
plt.xlim(xmin=0,xmax=5)
plt.xlabel('Time Latency (ms)',  fontsize=12)#, weight='bold')
ax.set_ylabel('Probablity Density',  fontsize=12)#, weight='bold')

ax.set_xlim(0,3.5)
ax.set_ylim(0,2.5)


# CDF
ax2 = ax.twinx()
norTc = np.divide(tc,np.sum(count))
ax2.plot(cummLat, norTc, label = 'CDF', \
         color = '#fdb863', linewidth=3)
ax2.set_ylim(0,1.0)
plt.ylabel('CDF',  fontsize=12)#, weight='bold')

plt.rc('legend',fontsize=9)
lines1,l1 = ax.get_legend_handles_labels()
lines3, l3 = plt.gca().get_legend_handles_labels()
lg = ax.legend(lines1 +lines3, l1 +l3,bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
       ncol=3, mode="expand", borderaxespad=0., prop={'size':9})

axes = plt.gca()
y_min, y_max = axes.get_ylim()
mid = (y_max - y_min)/2

# Average
totalLat = np.multiply(nzCummLat,nzCount)
totalLatSum = np.sum(totalLat)
totalCountSum = np.sum(nzCount)
AveLat = totalLatSum/totalCountSum

plt.axvline(x=AveLat, color='brown', linestyle='--', linewidth=3)
plt.text(AveLat+0.075,mid, \
         'Average (' + str(round(AveLat,2))  + 'ms)', \
         verticalalignment='center', rotation=90, \
         fontsize = 9, color = 'black',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))
            
# 99 percentile
indArr99 = np.where(ptile > 0.99)
ind99 = indArr99[0][0]
pt99 = cummLat[ind99]
#print(pt99)

plt.axvline(x=pt99, color='#5e3c99', linestyle='--', linewidth=3)
plt.text(pt99+0.075,mid,\
         '99 % Tail Latency (' + str(round(pt99,2)) + 'ms)', \
         verticalalignment='center', rotation=90, \
         fontsize = 9, color = 'black',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))
            
ax.grid()
plt.text(0.2,0.86,'Linux', fontsize = 10, color = 'black', \
        bbox=dict(facecolor='white', edgecolor='black', pad=10.0))
#---------------------------------
tmp = pd.read_csv('ukl-ret-bp',delim_whitespace=True, \
                  skipfooter=3, engine='python')

cummLat = tmp['Value'].to_numpy()
lat = np.diff(cummLat, axis=0)
lat = np.insert(lat, 0, cummLat[0])
nzLat = lat[lat != 0] # Getting non zero latencies
nzCummLat = cummLat[lat != 0]

ptile = tmp['Percentile'].to_numpy()

tc = tmp['TotalCount'].to_numpy()
nzTc = tc[lat != 0]
count = np.diff(tc, axis=0)
count = np.insert(count, 0, tc[0])
nzCount = count[lat != 0] # Getting freq. for non zero latencies
total = np.sum(nzCount)

Nw = np.multiply(nzLat, total) # multiply each latency with total freq
fd = np.divide(nzCount,Nw) # divide each freq. with Nw

check = np.multiply(nzLat,fd) # Multiply each latency with prob.
#print(np.sum(check)) # this should be 1

# probability distribution
axm.bar(nzCummLat, fd, width=0.03, label = "Probablity Density", color = '#b2abd2')
plt.xlim(xmin=0,xmax=5)
plt.xlabel('Time Latency (ms)',  fontsize=12)#, weight='bold')
axm.set_ylabel('Probablity Density',  fontsize=12)#, weight='bold')

axm.set_xlim(0,3.5)
axm.set_ylim(0,2.5)


# CDF
axm2 = axm.twinx()
norTc = np.divide(tc,np.sum(count))
axm2.plot(cummLat, norTc, label = 'CDF', \
         color = '#fdb863', linewidth=3)
axm2.set_ylim(0,1.0)
plt.ylabel('CDF',  fontsize=12)#, weight='bold')

#plt.rc('legend',fontsize=17)
#lines1,l1 = axm.get_legend_handles_labels()
#lines3, l3 = plt.gca().get_legend_handles_labels()
#lg = axm.legend(lines1 +lines3, l1 +l3,bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#       ncol=3, mode="expand", borderaxespad=0., prop={'size':13})

axes = plt.gca()
y_min, y_max = axes.get_ylim()
mid = (y_max - y_min)/2

# Average
totalLat = np.multiply(nzCummLat,nzCount)
totalLatSum = np.sum(totalLat)
totalCountSum = np.sum(nzCount)
AveLat = totalLatSum/totalCountSum

plt.axvline(x=AveLat, color='brown', linestyle='--', linewidth=3)
plt.text(AveLat+0.075,mid, \
         'Average (' + str(round(AveLat,2))  + 'ms)', \
         verticalalignment='center', rotation=90, \
         fontsize = 9, color = 'black',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))
            
# 99 percentile
indArr99 = np.where(ptile > 0.99)
ind99 = indArr99[0][0]
pt99 = cummLat[ind99]
#print(pt99)

plt.axvline(x=pt99, color='#5e3c99', linestyle='--', linewidth=3)
plt.text(pt99+0.075,mid,\
         '99 % Tail Latency (' + str(round(pt99,2)) + 'ms)', \
         verticalalignment='center', rotation=90, \
         fontsize = 9, color = 'black',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))
            
axm.grid()
plt.text(0.2,0.86,'UKL_RET_BYP', fontsize = 10, color = 'black', \
        bbox=dict(facecolor='white', edgecolor='black', pad=10.0))
#---------------------------------
tmp = pd.read_csv('ukl-ret-bp-sc',delim_whitespace=True, \
                  skipfooter=3, engine='python')

cummLat = tmp['Value'].to_numpy()
lat = np.diff(cummLat, axis=0)
lat = np.insert(lat, 0, cummLat[0])
nzLat = lat[lat != 0] # Getting non zero latencies
nzCummLat = cummLat[lat != 0]

ptile = tmp['Percentile'].to_numpy()

tc = tmp['TotalCount'].to_numpy()
nzTc = tc[lat != 0]
count = np.diff(tc, axis=0)
count = np.insert(count, 0, tc[0])
nzCount = count[lat != 0] # Getting freq. for non zero latencies
total = np.sum(nzCount)

Nw = np.multiply(nzLat, total) # multiply each latency with total freq
fd = np.divide(nzCount,Nw) # divide each freq. with Nw

check = np.multiply(nzLat,fd) # Multiply each latency with prob.
#print(np.sum(check)) # this should be 1

# probability distribution
axd.bar(nzCummLat, fd, width=0.03, label = "Probablity Density", color = '#b2abd2')
plt.xlim(xmin=0,xmax=5)
plt.xlabel('Time Latency (ms)',  fontsize=12)#, weight='bold')
axd.set_ylabel('Probablity Density',  fontsize=12)#, weight='bold')

axd.set_xlim(0,3.5)
axd.set_ylim(0,2.5)
ax.set_xlim(0,3.5)
axm.set_xlim(0,3.5)

# CDF
axd2 = axd.twinx()
norTc = np.divide(tc,np.sum(count))
axd2.plot(cummLat, norTc, label = 'CDF', \
         color = '#fdb863', linewidth=3)
axd2.set_ylim(0,1.0)
plt.ylabel('CDF',  fontsize=12)#, weight='bold')

#plt.rc('legend',fontsize=17)
#lines1,l1 = axd.get_legend_handles_labels()
#lines3, l3 = plt.gca().get_legend_handles_labels()
#lg = axd.legend(lines1 +lines3, l1 +l3,bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#       ncol=3, mode="expand", borderaxespad=0., prop={'size':13})

axes = plt.gca()
y_min, y_max = axes.get_ylim()
mid = (y_max - y_min)/2

# Average
totalLat = np.multiply(nzCummLat,nzCount)
totalLatSum = np.sum(totalLat)
totalCountSum = np.sum(nzCount)
AveLat = totalLatSum/totalCountSum

plt.axvline(x=AveLat, color='brown', linestyle='--', linewidth=3)
plt.text(AveLat+0.075,mid, \
         'Average (' + str(round(AveLat,2))  + 'ms)', \
         verticalalignment='center', rotation=90, \
         fontsize = 9, color = 'black',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))
            
# 99 percentile
indArr99 = np.where(ptile > 0.99)
ind99 = indArr99[0][0]
pt99 = cummLat[ind99]
#print(pt99)

plt.axvline(x=pt99, color='#5e3c99', linestyle='--', linewidth=3)
plt.text(pt99+0.075,mid,\
         '99 % Tail Latency (' + str(round(pt99,2)) + 'ms)', \
         verticalalignment='center', rotation=90, \
         fontsize = 9, color = 'black', \
         bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))
            
axd.grid()
plt.text(0.2,0.79,'UKL_RET_BYP\n   (shortcut)', fontsize = 10, color = 'black', \
        bbox=dict(facecolor='white', edgecolor='black', pad=10.0),wrap=True)
#----------------------

fig.tight_layout()
plt.savefig("redis.pdf")
plt.show()
