from corrLib import g2ToFile_cpu_thread_benchmark
import numpy as np
import time
import scipy

mat_directory = "Z:/Data/Processed Correlations/For Sandy/"
data_directory = "C:/Users/Ryd Berg/Downloads/"

data_folder = data_directory+"g2_benchmark/"
mat_file = mat_directory+"g2_benchmark_cpu"
benchmark_mat = mat_directory+"g2_benchmark_cpu_multithread"

bin_width = 82.3e-12*12
half_tau_bins = 1000
max_time = half_tau_bins * bin_width
pulse_spacing = 100e-6
max_pulse_distance = 4

cpu_threads = np.array(range(48))+1

time_taken = np.zeros(len(cpu_threads))
for i in range(len(cpu_threads)):
    start_time = time.time()
    g2ToFile_cpu_thread_benchmark(data_folder,mat_file,max_time,bin_width,pulse_spacing,max_pulse_distance,cpu_threads[i])
    time_taken[i] = time.time()-start_time


scipy.io.savemat(benchmark_mat,{'num_threads':cpu_threads,'time':time_taken,'device':'Threadripper 1950x'})