# %%
import numpy as np
import matplotlib.pyplot as plt
import time

import numpy as np
import matplotlib.pyplot as plt
import time

from numpy.core.fromnumeric import size


file_path = '/home/nyrenw/chalmers/FKA121/E5/task2/'
fname_high = 'trajs_high.dat'
fname_low = 'trajs_low.dat'
#with open(file) as f:
#    lines = f.read() ##Assume the sample file has 3 lines
#    str_settings = lines.split('\n', 1)[0]
#f.close()
#setting = str_settings.split(",")
#dt = float(setting[0])
#n_timesteps = int(setting[1])



data_high = np.genfromtxt(file_path + fname_high, delimiter=',', skip_header=0)
data_low = np.genfromtxt(file_path + fname_low, delimiter=',', skip_header=0)
n_timesteps = data_high[0,0]
n_trajs = data_high[0,1]
positions_high = data_high[1:,0]
velocities_high = data_high[1:,1]
positions_low = data_low[1:,0]
velocities_low = data_low[1:,1]

pos_high = np.split(positions_high, n_timesteps)
vel_high = np.split(velocities_high, n_timesteps)
pos_low = np.split(positions_low, n_timesteps)
vel_low = np.split(velocities_low, n_timesteps)

pos_avg_high = np.zeros_like(pos_high[:,0])
vel_avg_high = np.zeros_like(pos_high[:,0])
pos_avg_low = np.zeros_like(pos_high[:,0])
vel_avg_low = np.zeros_like(pos_high[:,0])

fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(16, 16))
dt = 0.001
for i in range(n_trajs):
    pos_avg_high += pos_high[:,i]
    vel_avg_high += vel_high[:,i]

    pos_avg_low += pos_low[:,i]
    vel_avg_low += vel_low[:,i]
    
    t = np.arange(len(data_low[:, 0]))*dt
    ax[0].plot(t, pos_low[:, 0],alpha=0.1)
    ax[1].plot(t, vel_low[:, 1],alpha=0.1)


    t = np.arange(len(data_high[:, 0]))*dt
    ax[2].plot(t, pos_high[:, i],alpha=0.1)
    ax[3].plot(t, vel_high[:, i],alpha=0.1)

pos_avg_low /= n_trajs
vel_avg_low /= n_trajs
pos_avg_high /= n_trajs
vel_avg_high /= n_trajs

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