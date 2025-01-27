# plot a figure that the neural state keep tracking to the madximum point of the posterior function.
# plot out the neural represented information during the delay
import context
import seaborn as sns
import matplotlib.pyplot as plt
from core.agent import Agent
import numpy as np
from core.tools import mean_se, save_dic, load_dic, find_nearest
from core.diff_drift import Diff_Drift, plot_traj
import sys
from core.ddm import fit_ddm

try:
    out_rnn_dir = sys.argv[2]
    out_fig_path = sys.argv[3]
except:
    out_rnn_dir = './figs/fig_data/rnn_info_change.json'
    out_fig_path = './figs/fig_collect/rnn_info_change.pdf'

try:
    if sys.argv[1] == 'Y': # set false so it will not generate data
        gen_data = True
    else:
        gen_data = False
except:
    gen_data = False

def find_split_point(arr):
    '''
    if abs(arr[i+1] - arr[i]) > 180, we call point i+1 as split points. 0 is a split point
    output:
      split_points (array)
    '''
    arr_diff = np.abs(np.diff(arr))
    split_points, = np.where(arr_diff > 180)
    split_points = np.insert(split_points + 1, 0, 0).astype(int)
    return split_points

prod_intervals = 1000
box_space = 20 # if equal to 5 then 0, 5, 10, 15, ..., 360. One box is an interval for example (0, 5). There should be 360 // box_sapce boxes
box_repeat = 1
common_color = [40, 130, 220, 310]

model_dir_list = ["../core/model/color_reproduction_delay_unit_25/",]
sub_dir_list = ["/model_6/noise_delta"]
rule_name="color_reproduction_delay_unit"

if gen_data:
    ddf = Diff_Drift()
    box_id = np.array([180])
    init_color = np.tile(box_id, box_repeat) # for every centroid we repeat trial multiple times
    batch_size = init_color.shape[0]
    n_time = prod_intervals // 20
    n_sub = len(model_dir_list)

    rnn_colors_tot = np.zeros((n_time, batch_size * n_sub))

    for j in range(n_sub):
        ########## RNN unit_90
        model_dir, sub_dir = model_dir_list[j], sub_dir_list[j]

        sub = Agent(model_dir + sub_dir, rule_name)
        ddf.read_rnn_agent(sub)
        rnn_time, rnn_colors = ddf.traj_fix_start(init_color, prod_intervals=prod_intervals, sigma_x=0, sigma_rec=0)
        rnn_colors_tot[:, j * batch_size: (j+1) * batch_size] = rnn_colors

    save_dic({'rnn_time': rnn_time, 'rnn_colors': rnn_colors_tot}, out_rnn_dir)

def plot_color_traj(time, colors, ax):
    new_color_start = colors[0, :]

    for i in range(colors.shape[-1]): # loop over all batches
        split_points = find_split_point(colors[:, i])
        for j in range(len(split_points) - 1):
            ax.plot(rnn_time[split_points[j] : split_points[j+1]], colors[split_points[j] : split_points[j+1], i])
        ax.plot(time[split_points[-1]:], colors[split_points[-1]:, i])

    ax.set_xlim([0, 1000])
    ax.tick_params(direction='in')

    ax.set_ylabel('Information Value (degree)')
    ax.set_xlabel('Delay Time (ms)')

sns.set()
sns.set_style("ticks")
data = load_dic(out_rnn_dir)
rnn_time, rnn_colors = np.array(data['rnn_time']), np.array(data['rnn_colors'])

##### Posterior
from core.bay_drift_loss_torch import Posterior
from core.bay_drift_loss_torch import prior_func
sigma_n = 0.006
sigma_s = 10.0 / 360 * 2 * np.pi
batch_size_post = 100
c_center = np.array([40, 130, 220, 310]) # 4 peaks
c_center = c_center / 360 * 2 * np.pi - np.pi # convert to rad

post = Posterior(c_center, sigma_s, sigma_n)
post.norm_table()

init_s = 0 # generate stimulus which is 180.
mesh_s = np.linspace(-np.pi, np.pi, batch_size_post, endpoint=False)
n_time = prod_intervals // 20
posterior_table = np.zeros((batch_size_post, n_time))

for i_t in range(n_time):
    for i_s, i_s_value in enumerate(mesh_s):
        posterior_table[i_s, i_t] = post.forward(init_s, i_s_value, i_t * 20)

fig = plt.figure(figsize=(5, 5))
ax_rnn = fig.add_subplot(111)

ax_rnn.imshow(posterior_table)

#prior_table = prior_func(mesh_s, c_center, sigma_s)
#ax_rnn.plot(prior_table)
#plot_color_traj(rnn_time, rnn_colors, ax_rnn)

fig.savefig(out_fig_path, format='pdf')
#plt.show()
