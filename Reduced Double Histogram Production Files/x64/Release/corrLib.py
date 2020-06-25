import ctypes
import _ctypes
import multiprocessing
import os
import numpy as np
import scipy.io
#import wurlitzer
import time

tagger_resolution = 82.3e-12*2
num_gpu = 2

working_directory = os.getcwd()
lib_name = 'pythonDLL.dll'

def file_list_to_ctypes(file_list,folder):
    len_list = len(file_list)
    str_array_type = ctypes.c_char_p * len_list
    str_array = str_array_type()
    for i, file in enumerate(file_list):
        str_array[i] = (folder+file).encode('utf-8')
    return str_array

def g2ToFile(folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance):
    file_list = os.listdir(folder_name)
    int_bin_width = round(bin_width / tagger_resolution) * tagger_resolution
    int_max_time = round(max_time / int_bin_width) * int_bin_width
    int_pulse_spacing = round(pulse_spacing / int_bin_width) * int_bin_width
    max_bin  = int(round(int_max_time/int_bin_width))
    num_files = len(file_list)
    numer_list = [int(0)]*(2*max_bin + 1)
    denom_ctypes = ctypes.c_int(0)
    ctypes_file_list = file_list_to_ctypes(file_list, folder_name)
    lib = ctypes.CDLL(working_directory + '/' + lib_name)
    lib.getG2Correlations.argtypes = [ctypes.c_char_p * num_files, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.py_object, ctypes.POINTER(ctypes.c_int)]
    start_time = time.time()
    #with wurlitzer.sys_pipes():
    lib.getG2Correlations(ctypes_file_list, num_files, int_max_time, int_bin_width, int_pulse_spacing, max_pulse_distance, numer_list, ctypes.byref(denom_ctypes))
    print("Finished in " + str(time.time()-start_time) + "s")
    
    if os.name == 'nt':
        _ctypes.FreeLibrary(lib._handle)
    else:
        _ctypes.dlclose(lib._handle)
    tau = np.arange(-max_bin,max_bin+1) * int_bin_width
    scipy.io.savemat(file_out_name,{'numer':np.array(numer_list),'denom':denom_ctypes.value,'tau':tau})

def g2ToFile_nostreams(folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance):
    file_list = os.listdir(folder_name)
    int_bin_width = round(bin_width / tagger_resolution) * tagger_resolution
    int_max_time = round(max_time / int_bin_width) * int_bin_width
    int_pulse_spacing = round(pulse_spacing / int_bin_width) * int_bin_width
    max_bin  = int(round(int_max_time/int_bin_width))
    num_files = len(file_list)
    numer_list = [int(0)]*(2*max_bin + 1)
    denom_ctypes = ctypes.c_int(0)
    ctypes_file_list = file_list_to_ctypes(file_list, folder_name)
    lib = ctypes.CDLL(working_directory + '/' + lib_name)
    lib.getG2Correlations_nostreams.argtypes = [ctypes.c_char_p * num_files, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.py_object, ctypes.POINTER(ctypes.c_int)]
    start_time = time.time()
    #with wurlitzer.sys_pipes():
    lib.getG2Correlations_nostreams(ctypes_file_list, num_files, int_max_time, int_bin_width, int_pulse_spacing, max_pulse_distance, numer_list, ctypes.byref(denom_ctypes))
    print("Finished in " + str(time.time()-start_time) + "s")
    
    if os.name == 'nt':
        _ctypes.FreeLibrary(lib._handle)
    else:
        _ctypes.dlclose(lib._handle)
    tau = np.arange(-max_bin,max_bin+1) * int_bin_width
    scipy.io.savemat(file_out_name,{'numer':np.array(numer_list),'denom':denom_ctypes.value,'tau':tau})

def g2ToFile_nominibatching(folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance):
    file_list = os.listdir(folder_name)
    int_bin_width = round(bin_width / tagger_resolution) * tagger_resolution
    int_max_time = round(max_time / int_bin_width) * int_bin_width
    int_pulse_spacing = round(pulse_spacing / int_bin_width) * int_bin_width
    max_bin  = int(round(int_max_time/int_bin_width))
    num_files = len(file_list)
    numer_list = [int(0)]*(2*max_bin + 1)
    denom_ctypes = ctypes.c_int(0)
    ctypes_file_list = file_list_to_ctypes(file_list, folder_name)
    lib = ctypes.CDLL(working_directory + '/' + lib_name)
    lib.getG2Correlations_nominibatching.argtypes = [ctypes.c_char_p * num_files, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.py_object, ctypes.POINTER(ctypes.c_int)]
    start_time = time.time()
    #with wurlitzer.sys_pipes():
    lib.getG2Correlations_nominibatching(ctypes_file_list, num_files, int_max_time, int_bin_width, int_pulse_spacing, max_pulse_distance, numer_list, ctypes.byref(denom_ctypes))
    print("Finished in " + str(time.time()-start_time) + "s")
    
    if os.name == 'nt':
        _ctypes.FreeLibrary(lib._handle)
    else:
        _ctypes.dlclose(lib._handle)
    tau = np.arange(-max_bin,max_bin+1) * int_bin_width
    scipy.io.savemat(file_out_name,{'numer':np.array(numer_list),'denom':denom_ctypes.value,'tau':tau})

def g3ToFile(folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance):
    file_list = os.listdir(folder_name)
    int_bin_width = round(bin_width / tagger_resolution) * tagger_resolution
    int_max_time = round(max_time / int_bin_width) * int_bin_width
    int_pulse_spacing = round(pulse_spacing / int_bin_width) * int_bin_width
    max_bin  = int(round(int_max_time/int_bin_width))
    num_files = len(file_list)
    numer_list = [int(0)]*(2*max_bin + 1)*(2*max_bin + 1)
    denom_ctypes = ctypes.c_int(0)
    ctypes_file_list = file_list_to_ctypes(file_list, folder_name)
    lib = ctypes.CDLL(working_directory + '/' + lib_name)
    lib.getG3Correlations.argtypes = [ctypes.c_char_p * num_files, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.py_object, ctypes.POINTER(ctypes.c_int)]
    start_time = time.time()
    #with wurlitzer.sys_pipes():
    lib.getG3Correlations(ctypes_file_list, num_files, int_max_time, int_bin_width, int_pulse_spacing, max_pulse_distance, numer_list, ctypes.byref(denom_ctypes))
    print("Finished in " + str(time.time()-start_time) + "s")
    
    if os.name == 'nt':
        _ctypes.FreeLibrary(lib._handle)
    else:
        _ctypes.dlclose(lib._handle)
    tau = np.arange(-max_bin,max_bin+1) * int_bin_width
    scipy.io.savemat(file_out_name,{'numer':np.array(numer_list).reshape((2*max_bin + 1),(2*max_bin + 1)),'denom':denom_ctypes.value,'tau':tau})
    
def g2ToFile_channelPair(folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance,channel_1,channel_2):
    file_list = os.listdir(folder_name)
    int_bin_width = round(bin_width / tagger_resolution) * tagger_resolution
    int_max_time = round(max_time / int_bin_width) * int_bin_width
    int_pulse_spacing = round(pulse_spacing / int_bin_width) * int_bin_width
    max_bin  = int(round(int_max_time/int_bin_width))
    num_files = len(file_list)
    numer_list = [int(0)]*(2*max_bin + 1)
    denom_ctypes = ctypes.c_int(0)
    ctypes_file_list = file_list_to_ctypes(file_list, folder_name)
    lib = ctypes.CDLL(working_directory + '/' + lib_name)
    lib.getG2Correlations_channelPair.argtypes = [ctypes.c_char_p * num_files, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.py_object, ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.c_int]
    start_time = time.time()
    #with wurlitzer.sys_pipes():
    lib.getG2Correlations_channelPair(ctypes_file_list, num_files, int_max_time, int_bin_width, int_pulse_spacing, max_pulse_distance, numer_list, ctypes.byref(denom_ctypes),channel_1,channel_2)
    print("Finished in " + str(time.time()-start_time) + "s")
    
    if os.name == 'nt':
        _ctypes.FreeLibrary(lib._handle)
    else:
        _ctypes.dlclose(lib._handle)
    tau = np.arange(-max_bin,max_bin+1) * int_bin_width
    scipy.io.savemat(file_out_name,{'numer':np.array(numer_list),'denom':denom_ctypes.value,'tau':tau})

def count_tags(folder_name,max_time,bin_width,pulse_spacing,max_pulse_distance):
	file_list = os.listdir(folder_name)
	int_bin_width = round(bin_width / tagger_resolution) * tagger_resolution
	int_max_time = round(max_time / int_bin_width) * int_bin_width
	int_pulse_spacing = round(pulse_spacing / int_bin_width) * int_bin_width
	num_files = len(file_list)
	ctypes_file_list = file_list_to_ctypes(file_list, folder_name)
	lib = ctypes.CDLL(working_directory + '/' + lib_name)
	lib.count_tags.argtypes = [ctypes.c_char_p * num_files, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int]
	lib.count_tags(ctypes_file_list, num_files, int_max_time, int_bin_width, int_pulse_spacing, max_pulse_distance)
	
def g2ToFile_cpu(folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance,cpu_threads):
    file_list = os.listdir(folder_name)
    int_bin_width = round(bin_width / tagger_resolution) * tagger_resolution
    int_max_time = round(max_time / int_bin_width) * int_bin_width
    int_pulse_spacing = round(pulse_spacing / int_bin_width) * int_bin_width
    max_bin  = int(round(int_max_time/int_bin_width))
    num_files = len(file_list)
    numer_list = [int(0)]*(2*max_bin + 1)
    denom_ctypes = ctypes.c_int(0)
    ctypes_file_list = file_list_to_ctypes(file_list, folder_name)
    lib = ctypes.CDLL(working_directory + '/' + lib_name)
    lib.getG2Correlations_cpu.argtypes = [ctypes.c_char_p * num_files, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.py_object, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    start_time = time.time()
    #with wurlitzer.sys_pipes():
    lib.getG2Correlations_cpu(ctypes_file_list, num_files, int_max_time, int_bin_width, int_pulse_spacing, max_pulse_distance, numer_list, ctypes.byref(denom_ctypes), cpu_threads)
    print("Finished in " + str(time.time()-start_time) + "s")
    
    #if os.name == 'nt':
    #    _ctypes.FreeLibrary(lib._handle)
    #else:
    #    _ctypes.dlclose(lib._handle)
    tau = np.arange(-max_bin,max_bin+1) * int_bin_width
    scipy.io.savemat(file_out_name,{'numer':np.array(numer_list),'denom':denom_ctypes.value,'tau':tau})

def g2ToFile_cpu_thread_benchmark(folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance,cpu_threads):
    file_list = os.listdir(folder_name)
    int_bin_width = round(bin_width / tagger_resolution) * tagger_resolution
    int_max_time = round(max_time / int_bin_width) * int_bin_width
    int_pulse_spacing = round(pulse_spacing / int_bin_width) * int_bin_width
    max_bin  = int(round(int_max_time/int_bin_width))
    num_files = len(file_list)
    numer_list = [int(0)]*(2*max_bin + 1)
    denom_ctypes = ctypes.c_int(0)
    ctypes_file_list = file_list_to_ctypes(file_list, folder_name)
    lib = ctypes.CDLL(working_directory + '/' + lib_name)
    lib.getG2Correlations_cpu_thread_benchmark.argtypes = [ctypes.c_char_p * num_files, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.py_object, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    start_time = time.time()
    #with wurlitzer.sys_pipes():
    lib.getG2Correlations_cpu_thread_benchmark(ctypes_file_list, num_files, int_max_time, int_bin_width, int_pulse_spacing, max_pulse_distance, numer_list, ctypes.byref(denom_ctypes), cpu_threads)
    print("Finished in " + str(time.time()-start_time) + "s")
    
    #if os.name == 'nt':
    #    _ctypes.FreeLibrary(lib._handle)
    #else:
    #    _ctypes.dlclose(lib._handle)
    tau = np.arange(-max_bin,max_bin+1) * int_bin_width
    #scipy.io.savemat(file_out_name,{'numer':np.array(numer_list),'denom':denom_ctypes.value,'tau':tau})