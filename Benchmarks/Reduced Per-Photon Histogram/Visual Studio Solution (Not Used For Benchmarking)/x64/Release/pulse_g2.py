mat_directory = "Z:/Data/Processed Correlations/For Sandy/"
data_folder= "Z:/Data/Correlation Data/2018/November/12/Test/"

from corrLib import g2ToFile_pulse,g2ToFile_pulse_2

max_tau_2 = 500e-6
min_tau_2 = -500e-6
max_tau_1 = 500e-6
min_tau_1 = 0
bin_width = 100e-9
#g2ToFile_pulse(data_folder, mat_directory + "Test_old", max_tau, bin_width)
g2ToFile_pulse_2(data_folder, mat_directory + "Test_new", min_tau_1, max_tau_1, min_tau_2, max_tau_2, bin_width)