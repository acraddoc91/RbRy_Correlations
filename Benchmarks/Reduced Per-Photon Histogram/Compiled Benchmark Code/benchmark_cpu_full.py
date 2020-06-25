from corrLib import g2ToFile
import numpy as np
import time
import os
import scipy

current_dir = os.getcwd()
mat_directory = os.path.abspath(current_dir + "/../../Benchmark Results").replace("\\","/") + "/"
data_folder = os.path.abspath(current_dir + "/../../Benchmark Files").replace("\\","/") + "/"

mat_file = mat_directory+"g2_out"
benchmark_mat = mat_directory+"reduced_per_photon_histogram_cpu_benchmark"

bin_width = 82.3e-12*12
pulse_spacing = 100e-6
max_pulse_distance = 4
half_tau_bins = np.array([1,2,5,10,20,50,100,200,500,1000,2000,5000,10000,20000,50000,100000,200000,500000])
calc_bins = half_tau_bins * 2 + 1
time_taken = np.zeros(len(half_tau_bins))
for i in range(len(half_tau_bins)):
    max_time = half_tau_bins[i] * bin_width
    start_time = time.time()
    g2ToFile(data_folder,mat_file,max_time,bin_width,pulse_spacing,max_pulse_distance)
    time_taken[i] = time.time()-start_time

scipy.io.savemat(benchmark_mat,{'num_bins':calc_bins,'time':time_taken,'device':'Threadripper 1950x','thread_num':32})