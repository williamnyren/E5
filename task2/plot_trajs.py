# %%
import numpy as np
import matplotlib.pyplot as plt
import time

import numpy as np
import matplotlib.pyplot as plt
import time


file_path = '/home/nyrenw/chalmers/FKA121/E5/task2/'
fnames_high = ['0_high.dat', '1_high.dat', '2_high.dat', '3_high.dat', '4_high.dat']
fnames_low = ['0_low.dat', '1_low.dat', '2_low.dat', '3_low.dat', '4_low.dat']
#with open(file) as f:
#    lines = f.read() ##Assume the sample file has 3 lines
#    str_settings = lines.split('\n', 1)[0]
#f.close()
#setting = str_settings.split(",")
#dt = float(setting[0])
#n_timesteps = int(setting[1])


# Read normal gaussian from results.dat
data_high = []
data_low = []
time_avg_high = np.zeros(np.genfromtxt(file_path + fnames_high[0], delimiter=',', skip_header=1).shape)
time_avg_low = np.zeros(np.genfromtxt(file_path + fnames_low[0], delimiter=',', skip_header=1).shape)

fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(16, 16))
dt = 0.001
for i, fname_high in enumerate(fnames_high):
    data_high = np.genfromtxt(file_path + fnames_high[i], delimiter=',', skip_header=1)
    time_avg_high += data_high

    data_low = np.genfromtxt(file_path + fnames_low[i], delimiter=',', skip_header=1)
    time_avg_low += data_low

    t = np.arange(len(data_low[:, 0]))*dt
    ax[0].plot(t, data_low[:, 0])
    ax[1].plot(t, data_low[:, 1])


    t = np.arange(len(data_high[:, 0]))*dt
    ax[2].plot(t, data_high[:, 0])
    ax[3].plot(t, data_high[:, 1])
time_avg_low = time_avg_low/5.0
time_avg_high = time_avg_high/5.0
ax[0].set_xlim((0, 6))
ax[0].set_ylabel(r"r_low []")
ax[1].set_xlim((0, 6))
ax[1].set_ylabel(r"v_low []")
ax[2].set_xlim((0, 6))
ax[2].set_ylabel(r"r_high []")
ax[3].set_xlim((0, 6))
ax[3].set_ylabel(r"v_high []")
ax[3].set_xlabel(r"time [ms]")

plt.savefig('traj.png', dpi=600)
plt.show()
fig, ax = plt.subplots(ncols=4,nrows=1,figsize=(16,16))
ax[0].plot(t,time_avg_low[:,0],label='time avg low')
ax[1].plot(t,time_avg_low[:,1],label='time avg low')
ax[0].plot(t,time_avg_high[:,0],label='time avg low')
ax[1].plot(t,time_avg_high[:,1],label='time avg low')
plt.savefig('time_avg_traj.png',dpi=600)
plt.show()

# %%

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(  np.arange(len(v))*dt, v[:]*1e3, alpha = 0.2, color='red', marker='.')
    ##time.sleep(time_duration)

ax.set_ylabel('Velocity [mm/s]')
ax.set_xlabel('t [ms]')
#ax.set_xlim(-0.1, 0.1)
#ax.set_ylim(-0.1, 0.1)
plt.show()
# %%
