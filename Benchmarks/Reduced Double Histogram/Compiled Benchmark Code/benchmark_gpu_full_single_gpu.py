from corrLib_single_gpu import g2ToFile, g2ToFile_nominibatching, g2ToFile_nostreams
import numpy as np
import time
import scipy
import os

current_dir = os.getcwd()
mat_directory = os.path.abspath(current_dir + "/../../Benchmark Results").replace("\\","/") + "/"
data_folder = os.path.abspath(current_dir + "/../../Benchmark Files").replace("\\","/") + "/"

mat_file = mat_directory+"g2_out"
benchmark_mat = mat_directory+"reduced_double_histogram_single_gpu_benchmark"

bin_width = 82.3e-12*12
pulse_spacing = 100e-6
max_pulse_distance = 4
half_tau_bins = np.array([1,2,5,10,20,50,100,200,500,1000,2000,5000,10000,20000,50000,100000,200000,500000])
#half_tau_bins = np.array([16000,32000,64000,128000,256000])
calc_bins = half_tau_bins * 2 + 1
time_taken = np.zeros(len(half_tau_bins))
for i in range(len(half_tau_bins)):
    max_time = half_tau_bins[i] * bin_width
    start_time = time.time()
    g2ToFile(data_folder,mat_file,max_time,bin_width,pulse_spacing,max_pulse_distance)
    time_taken[i] = time.time()-start_time

    time.sleep(5)

scipy.io.savemat(benchmark_mat,{'num_bins':calc_bins,'time':time_taken})