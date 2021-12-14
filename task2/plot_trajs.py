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

fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(16, 16))
dt = 0.001
for i, fname_high in enumerate(fnames_high):
    data_high = np.genfromtxt(file_path + fnames_high[i], delimiter=',', skip_header=1)

    data_low = np.genfromtxt(file_path + fnames_low[i], delimiter=',', skip_header=1)

    t = np.arange(len(data_low[:, 0]))*dt
    ax[0].plot(t, data_low[:, 0]*1e3)
    ax[1].plot(t, data_low[:, 1]*1e3)


    t = np.arange(len(data_high[:, 0]))*dt
    ax[2].plot(t, data_high[:, 0]*1e3)
    ax[3].plot(t, data_high[:, 1]*1e3)
ax[0].set_xlim((0, 6))
ax[0].set_ylabel(r"r_low  [$\mu$m]", size=15)
ax[1].set_xlim((0, 6))
ax[1].set_ylabel(r"v_low  [$\mu$m/ms]", size=15)
ax[2].set_xlim((0, 6))
ax[2].set_ylabel(r"r_high  [$\mu$m]", size=15)
ax[3].set_xlim((0, 6))
ax[3].set_ylabel(r"v_high  [$\mu$m/ms]", size=15)
ax[3].set_xlabel(r"time  [ms]", size=15)
plt.savefig('traj.png', dpi=600)
ax[0].grid()
ax[1].grid()
ax[2].grid()
ax[3].grid()
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
