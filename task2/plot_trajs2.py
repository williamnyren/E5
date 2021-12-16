# %%
import numpy as np
import matplotlib.pyplot as plt
import time

import numpy as np
import matplotlib.pyplot as plt
import time

from numpy.core.fromnumeric import size


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



## ------- HIGH ---------
time_avg_high = np.zeros(np.genfromtxt(file_path + fnames_high[0], delimiter=',', skip_header=1).shape)

sigma_sq_high = np.zeros(np.genfromtxt(file_path + fnames_high[0], delimiter=',', skip_header=1).shape)

## ------- LOW ---------
time_avg_low = np.zeros(np.genfromtxt(file_path + fnames_low[0], delimiter=',', skip_header=1).shape)

sigma_sq_low = np.zeros(np.genfromtxt(file_path + fnames_high[0], delimiter=',', skip_header=1).shape)



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

time_avg_low /= 5.0
time_avg_high /= 5.0

x_max = 1
x_min = -0.005
font_size = 25
ax[0].set_xlim((0, x_max))
ax[0].set_ylabel(r"$x_{low}$  [$\mu$m]", size=font_size)
ax[1].set_xlim((0, x_max))
ax[1].set_ylabel(r"$v_{low}$  [$\mu$m/ms]", size=font_size)
ax[2].set_xlim((0, x_max))
ax[2].set_ylabel(r"$x_{high}$  [$\mu$m]", size=font_size)
ax[3].set_xlim((0, x_max))
ax[3].set_ylabel(r"$v_{high}$ [$\mu$m/ms]", size=font_size)
ax[3].set_xlabel(r"time [ms]", size=font_size)

ax[0].grid()
ax[1].grid()
ax[2].grid()
ax[3].grid()
plt.savefig('traj.png', dpi=600)
plt.show()

# %%

for i, fname_high in enumerate(fnames_high):
    data_high = np.genfromtxt(file_path + fnames_high[i], delimiter=',', skip_header=1)
    sigma_sq_high += (data_high - time_avg_high)**2

    data_low = np.genfromtxt(file_path + fnames_low[i], delimiter=',', skip_header=1)
    sigma_sq_low += (data_low - time_avg_low)**2

sigma_sq_low /= 5.0
sigma_sq_high /= 5.0



fig, ax = plt.subplots(nrows=4,ncols=1,figsize=(16,16))
ax[0].plot(t, time_avg_low[:, 0], color='red', label='time avg low')
ax[2].plot(t, time_avg_low[:, 1], color='blue', label='time avg low')
ax[1].plot(t, sigma_sq_low[:, 0], color='red', label=r"low")
ax[3].plot(t, sigma_sq_low[:, 1], color='blue', label=r"low")
ax[0].set_xlim((x_max, x_max))
ax[0].set_xlim((x_min, x_max))
ax[0].set_ylabel(r"$\langle \mu_{x} \rangle$  [$\mu$m]", size=font_size)
ax[1].set_xlim((x_min, x_max))
ax[1].set_ylabel(r"$\langle \sigma_{x}^{2} \rangle$  [$(\mu$m)]", size=font_size)
ax[2].set_xlim((x_min, x_max))
ax[2].set_ylabel(r"$\langle \mu_{v} \rangle$  [$(\mu$m/ms)$^{2}$]", size=font_size)
ax[3].set_xlim((x_min, x_max))
ax[3].set_ylabel(r"$\langle \sigma_{v}^{2} \rangle$  [$(\mu$m/ms)$^{2}$]", size=font_size)
ax[0].set_title(r"Case LOW")
ax[0].grid()
ax[1].grid()
ax[2].grid()
ax[3].grid()
plt.savefig('time_avg_low.png',dpi=600)
plt.show()

# %%

fig, ax = plt.subplots(nrows=4,ncols=1,figsize=(16,16))
ax[0].plot(t, time_avg_high[:, 0], color='red', label='time avg high')
ax[2].plot(t, time_avg_high[:, 1], color='blue',label='time avg high')
ax[1].plot(t, sigma_sq_high[:, 0], color='red', label=r"$\sigma^{2}$")
ax[3].plot(t, sigma_sq_high[:, 1], color='blue',label=r"$\langle \sigma^{2} \rangle$")

ax[0].set_xlim((x_min, x_max))
ax[0].set_xlim((x_min, x_max))
ax[0].set_ylabel(r"$\langle \mu_{x} \rangle$  [$\mu$m]", size=font_size)
ax[1].set_xlim((x_min, x_max))
ax[1].set_ylabel(r"$\langle \sigma_{x}^{2} \rangle$  [$(\mu$m)]", size=font_size)
ax[2].set_xlim((x_min, x_max))
ax[2].set_ylabel(r"$\langle \mu_{v} \rangle$  [$(\mu$m/ms)$^{2}$]", size=font_size)
ax[3].set_xlim((x_min, x_max))
ax[3].set_ylabel(r"$\langle \sigma_{v}^{2} \rangle$  [$(\mu$m/ms)$^{2}$]", size=font_size)
ax[3].set_xlabel(r"time [ms]", size=font_size)
ax[0].set_title(r"Case HIGH")
ax[0].grid()
ax[1].grid()
ax[2].grid()
ax[3].grid()
plt.savefig('time_avg_high.png',dpi=600)
plt.show()
# %%