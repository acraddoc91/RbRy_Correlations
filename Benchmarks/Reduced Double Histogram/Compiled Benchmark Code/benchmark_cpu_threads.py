from corrLib import g2ToFile_cpu_thread_benchmark
import numpy as np
import time
import scipy
import os

current_dir = os.getcwd()
mat_directory = os.path.abspath(current_dir + "/../../Benchmark Results").replace("\\","/") + "/"
data_folder = os.path.abspath(current_dir + "/../../Benchmark Files").replace("\\","/") + "/"

mat_file = mat_directory+"g2_out"
benchmark_mat = mat_directory+"cpu_multithreading_benchmark"

bin_width = 82.3e-12*12
half_tau_bins = 1000
max_time = half_tau_bins * bin_width
pulse_spacing = 100e-6
max_pulse_distance = 4

#cpu_threads = np.array(range(48))+1
cpu_threads = [i for i in range(1,41)]

time_taken = np.zeros(len(cpu_threads))
for i in range(len(cpu_threads)):
    start_time = time.time()
    g2ToFile_cpu_thread_benchmark(data_folder,mat_file,max_time,bin_width,pulse_spacing,max_pulse_distance,cpu_threads[i])
    time_taken[i] = time.time()-start_time


scipy.io.savemat(benchmark_mat,{'num_threads':cpu_threads,'time':time_taken,'device':'Threadripper 1950x'})