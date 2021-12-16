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

pos_high = np.split(positions_high, n_trajs)
vel_high = np.split(velocities_high, n_trajs)
pos_low = np.split(positions_low, n_trajs)
vel_low = np.split(velocities_low, n_trajs)

pos_avg_high = np.zeros_like(pos_high[0])
vel_avg_high = np.zeros_like(pos_high[0])
pos_avg_low = np.zeros_like(pos_high[0])
vel_avg_low = np.zeros_like(pos_high[0])
# %%
fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(16, 16))
dt = 0.001
n_trajs = int(n_trajs)
n_timesteps = int(n_timesteps)
print(n_timesteps, n_trajs)
t = np.arange(n_timesteps)*dt
for i in range(n_trajs):
    pos_avg_high += pos_high[i][:]
    vel_avg_high += vel_high[i][:]

    pos_avg_low += pos_low[i][:]
    vel_avg_low += vel_low[i][:]
    
    ax[0].plot(t, pos_low[i][:],alpha=0.1)
    ax[1].plot(t, vel_low[i][:],alpha=0.1)


    ax[2].plot(t, pos_high[i][:],alpha=0.1)
    ax[3].plot(t, vel_high[i][:],alpha=0.1)

pos_avg_low /= n_trajs
vel_avg_low /= n_trajs
pos_avg_high /= n_trajs
vel_avg_high /= n_trajs


sigma_pos_low = np.zeros_like(pos_avg_high)
sigma_v_low = np.zeros_like(pos_avg_high)
sigma_pos_high = np.zeros_like(pos_avg_high)
sigma_v_high = np.zeros_like(pos_avg_high)

for i in range(n_trajs):
    sigma_pos_low += (pos_low[i][:] - pos_avg_low[:])**2
    sigma_v_low += (vel_low[i][:] - vel_avg_low[:])**2

    sigma_pos_high += (pos_high[i][:] - pos_avg_high[:])**2
    sigma_v_high += (vel_high[i][:] - vel_avg_high[:])**2

sigma_pos_low /= n_trajs
sigma_v_low /=  n_trajs
sigma_pos_high /= n_trajs
sigma_v_high /= n_trajs




ax[0].plot(t, pos_avg_low - np.sqrt(sigma_pos_low),color=  '#90db24'
, linewidth=2)
ax[1].plot(t, vel_avg_low - np.sqrt(sigma_v_low),  color= '#a11de2'
 , linewidth=2)

ax[0].plot(t, pos_avg_low + np.sqrt(sigma_pos_low), color=  '#90db24'
, linewidth=2)
ax[1].plot(t, vel_avg_low + np.sqrt(sigma_v_low),   color= '#a11de2'
 , linewidth=2)




ax[2].plot(t, pos_avg_high - np.sqrt(sigma_pos_high), 
color=  '#90db24'
, linewidth=2)

ax[3].plot(t, vel_avg_high - np.sqrt(sigma_v_high), 
color= '#a11de2'
 , linewidth=2)

ax[2].plot(t, pos_avg_high + np.sqrt(sigma_pos_high), 
color=  '#90db24'
, linewidth=2)

ax[3].plot(t, vel_avg_high + np.sqrt(sigma_v_high), 
color= '#a11de2'
 , linewidth=2)

ax[0].plot(t, pos_avg_low, color= 'green', linewidth=2)
ax[1].plot(t, vel_avg_low, color= 'blue' , linewidth=2)
ax[2].plot(t, pos_avg_high, color= 'green', linewidth=2)
ax[3].plot(t, vel_avg_high, color= 'blue' , linewidth=2)

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
