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
data_high.append(np.genfromtxt(file_path + fnames_high, delimiter=',', skip_header=1))

data_low = []
data_low.append(np.genfromtxt(file_path + fnames_low, delimiter=',', skip_header=1))

r = data[:n_timesteps, 0]

v = data[:n_timesteps, 1]

fig = plt.figure()
ax = fig.add_subplot()
ax.plot( np.arange(len(r))*dt, r[:]*1e6, alpha = 0.2, color='blue', marker='.')
    ##time.sleep(time_duration)

ax.set_xlabel('t [ms]')
ax.set_ylabel(r'Position [nm]')
#ax.set_xlim(-0.1, 0.1)
#ax.set_ylim(-0.1, 0.1)
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
