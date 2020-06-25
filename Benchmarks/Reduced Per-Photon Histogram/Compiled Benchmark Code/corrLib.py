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
lib_name = 'pythonDLLCPU.so'

def file_list_to_ctypes(file_list,folder):
    len_list = len(file_list)
    str_array_type = ctypes.c_char_p * len_list
    str_array = str_array_type()
    for i, file in enumerate(file_list):
        str_array[i] = (folder+file).encode('utf-8')
    return str_array

def g2ToFile(folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance,calc_norm=True):
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
    lib.getG2Correlations.argtypes = [ctypes.c_char_p * num_files, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int64, ctypes.py_object, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    start_time = time.time()
    #with wurlitzer.sys_pipes():
    lib.getG2Correlations(ctypes_file_list, num_files, int_max_time, int_bin_width, int_pulse_spacing, max_pulse_distance, numer_list, ctypes.byref(denom_ctypes),calc_norm,32,1)
    print("Finished in " + str(time.time()-start_time) + "s")
    time.sleep(1)
    if os.name == 'nt':
        _ctypes.FreeLibrary(lib._handle)
    else:
        _ctypes.dlclose(lib._handle)
    tau = np.arange(-max_bin,max_bin+1) * int_bin_width
    scipy.io.savemat(file_out_name,{'numer':np.array(numer_list),'denom':denom_ctypes.value,'tau':tau})

def g2ToFile_cpu_thread_benchmark(folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance,file_threads,proc_threads,calc_norm=True):
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
    lib.getG2Correlations.argtypes = [ctypes.c_char_p * num_files, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int64, ctypes.py_object, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    start_time = time.time()
    #with wurlitzer.sys_pipes():
    lib.getG2Correlations(ctypes_file_list, num_files, int_max_time, int_bin_width, int_pulse_spacing, max_pulse_distance, numer_list, ctypes.byref(denom_ctypes),calc_norm,file_threads,proc_threads)
    print("Finished in " + str(time.time()-start_time) + "s")
    time.sleep(1)
    if os.name == 'nt':
        _ctypes.FreeLibrary(lib._handle)
    else:
        _ctypes.dlclose(lib._handle)
    tau = np.arange(-max_bin,max_bin+1) * int_bin_width
    scipy.io.savemat(file_out_name,{'numer':np.array(numer_list),'denom':denom_ctypes.value,'tau':tau})

def g2ToDict(folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance,calc_norm=True,update=False,disp_counts=False):
    #Convert various parameters to their integer values for the DLL
    int_bin_width = round(bin_width / tagger_resolution) * tagger_resolution
    int_max_time = round(max_time / int_bin_width) * int_bin_width
    int_pulse_spacing = round(pulse_spacing / int_bin_width) * int_bin_width
    max_bin  = int(round(int_max_time/int_bin_width))
    numer_list = [int(0)]*(2*max_bin + 1)
    #Calculate what tau should look like
    tau = np.arange(-max_bin,max_bin+1) * int_bin_width
    
    #Get list of files to process from the data folder
    dir_file_list = os.listdir(folder_name)
    #Previous values to add to new calculated total
    old_denom = 0
    old_numer = []
    updating = False
    #Check if the output file already exists, and if we should be updating the old file
    if (os.path.isfile(file_out_name) or os.path.isfile(file_out_name+".mat")) and update:
        #Load old matrix
        old_mat = scipy.io.loadmat(file_out_name)
        #Try and fetch old file list, try included for backwards compatibility
        try:
            #Grab old file list, denominator and numerator
            #Strip removes whitespace which seems to occasionally be a problem
            old_file_list = [filename.strip() for filename in old_mat['file_list']]
            old_denom = old_mat['denom_g2'][0][0]
            old_numer = old_mat['numer_g2'][0]
            old_tau = old_mat['tau'][0]
            if not np.array_equal(tau,old_tau):
                print("Can't update as the new and old tau values are different")
                raise Exception("Different tau values")
            #If so only process files that are different from last processing
            file_list = [filename for filename in dir_file_list if filename not in old_file_list]
            updating = True
        except:
            #Throw an error if the old stuff couldn't be grabbed and just default to non-update behaviour
            print("Error thrown, falling back on using whole file list")
            file_list = dir_file_list
    else:
        #Otherwise process everything
        file_list = dir_file_list
    
    num_files = len(file_list)
    #Convert things to C versions for the DLL
    denom_ctypes = ctypes.c_int(0)
    ctypes_file_list = file_list_to_ctypes(file_list, folder_name)
    #Setup the DLL
    lib = ctypes.CDLL(working_directory + '/' + lib_name)
    if(disp_counts):
        lib.getG2Correlations_with_rate_calc.argtypes = [ctypes.c_char_p * num_files, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int64, ctypes.py_object, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        start_time = time.time()
        #Call the DLL
        lib.getG2Correlations_with_rate_calc(ctypes_file_list, num_files, int_max_time, int_bin_width, int_pulse_spacing, max_pulse_distance, numer_list, ctypes.byref(denom_ctypes),calc_norm,4,4)
    else:
        lib.getG2Correlations.argtypes = [ctypes.c_char_p * num_files, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int64, ctypes.py_object, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        start_time = time.time()
        #Call the DLL
        lib.getG2Correlations(ctypes_file_list, num_files, int_max_time, int_bin_width, int_pulse_spacing, max_pulse_distance, numer_list, ctypes.byref(denom_ctypes),calc_norm,4,4)
    print("Finished in " + str(time.time()-start_time) + "s")
    
    time.sleep(1)
    #This is required in Windows as otherwise the DLL can't be re-used without rebooting the computer
    if os.name == 'nt':
        _ctypes.FreeLibrary(lib._handle)
    else:
        _ctypes.dlclose(lib._handle)
    #Check if we have old values to update
    output_dict = {}
    if updating:
        try:
            #Try and add the newly calculated values to the old ones may fuck up if you ask for numerators of different sizes
            output_dict = {'numer_g2':np.array(numer_list)+old_numer,'denom_g2':denom_ctypes.value+old_denom,'tau':tau,'file_list':dir_file_list}
        except:
            print("Could not update values")
    else:
        output_dict = {'numer_g2':np.array(numer_list),'denom_g2':denom_ctypes.value,'tau':tau,'file_list':dir_file_list}
    return output_dict

def g3ToFile(folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance,calc_norm=True):
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
    lib.getG3Correlations.argtypes = [ctypes.c_char_p * num_files, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int64, ctypes.py_object, ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.c_int]
    start_time = time.time()
    #with wurlitzer.sys_pipes():
    lib.getG3Correlations(ctypes_file_list, num_files, int_max_time, int_bin_width, int_pulse_spacing, max_pulse_distance, numer_list, ctypes.byref(denom_ctypes),calc_norm,4,4)
    print("Finished in " + str(time.time()-start_time) + "s")
    time.sleep(1)
    if os.name == 'nt':
        _ctypes.FreeLibrary(lib._handle)
    else:
        _ctypes.dlclose(lib._handle)
    tau = np.arange(-max_bin,max_bin+1) * int_bin_width
    scipy.io.savemat(file_out_name,{'numer':np.array(numer_list).reshape((2*max_bin + 1),(2*max_bin + 1)),'denom':denom_ctypes.value,'tau':tau})

def g3ToDict(folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance,calc_norm=True,update=False):
    #Convert various parameters to their integer values for the DLL
    int_bin_width = round(bin_width / tagger_resolution) * tagger_resolution
    int_max_time = round(max_time / int_bin_width) * int_bin_width
    int_pulse_spacing = round(pulse_spacing / int_bin_width) * int_bin_width
    max_bin  = int(round(int_max_time/int_bin_width))
    numer_list = [int(0)]*(2*max_bin + 1)*(2*max_bin + 1)
    #Calculate what tau should look like
    tau = np.arange(-max_bin,max_bin+1) * int_bin_width

    #Get list of files to process from the data folder
    dir_file_list = os.listdir(folder_name)
    #Previous values to add to new calculated total
    old_denom = 0
    old_numer = []
    updating = False
    #Check if the output file already exists, and if we should be updating the old file
    if (os.path.isfile(file_out_name) or os.path.isfile(file_out_name+".mat")) and update:
        #Load old matrix
        old_mat = scipy.io.loadmat(file_out_name)
        #Try and fetch old file list, try included for backwards compatibility
        try:
            #Grab old file list, denominator and numerator
            old_file_list = [filename.strip() for filename in old_mat['file_list']]
            old_denom = old_mat['denom_g3'][0][0]
            old_numer = old_mat['numer_g3']
            old_tau = old_mat['tau'][0]
            if not np.array_equal(tau,old_tau):
                print("Can't update as the new and old tau values are different")
                raise Exception("Different tau values")
            #If so only process files that are different from last processing
            file_list = [filename for filename in dir_file_list if (filename not in old_file_list)]
            updating = True
        except:
            #Throw an error if the old stuff couldn't be grabbed and just default to non-update behaviour
            print("Error thrown, falling back on using whole file list")
            file_list = dir_file_list
    else:
        #Otherwise process everything
        file_list = dir_file_list

    num_files = len(file_list)
    
    #Convert things to C versions for the DLL
    denom_ctypes = ctypes.c_int(0)
    ctypes_file_list = file_list_to_ctypes(file_list, folder_name)
    #Setup the DLL
    lib = ctypes.CDLL(working_directory + '/' + lib_name)
    lib.getG3Correlations.argtypes = [ctypes.c_char_p * num_files, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int64, ctypes.py_object, ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.c_int]
    start_time = time.time()
    #Call the DLL
    lib.getG3Correlations(ctypes_file_list, num_files, int_max_time, int_bin_width, int_pulse_spacing, max_pulse_distance, numer_list, ctypes.byref(denom_ctypes),calc_norm,4,4)
    print("Finished in " + str(time.time()-start_time) + "s")
    time.sleep(1)
    #This is required in Windows as otherwise the DLL can't be re-used without rebooting the computer
    if os.name == 'nt':
        _ctypes.FreeLibrary(lib._handle)
    else:
        _ctypes.dlclose(lib._handle)

    #Check if we have old values to update
    output_dict = {}
    if updating:
        try:
            #Try and add the newly calculated values to the old ones may fuck up if you ask for numerators of different sizes
            output_dict = {'numer_g3':np.array(numer_list).reshape((2*max_bin + 1),(2*max_bin + 1))+old_numer,'denom_g3':denom_ctypes.value +old_denom,'tau':tau,'file_list':dir_file_list}
        except:
            print("Could not update values")
    else:
        output_dict = {'numer_g3':np.array(numer_list).reshape((2*max_bin + 1),(2*max_bin + 1)),'denom_g3':denom_ctypes.value,'tau':tau,'file_list':dir_file_list}
    return output_dict

def g2ToFile_pulse(folder_name, file_out_name, min_tau_1, max_tau_1, min_tau_2, max_tau_2, bin_width):
    file_list = os.listdir(folder_name)
    int_bin_width = round(bin_width / tagger_resolution) * tagger_resolution

    int_min_tau_1 = round(min_tau_1 / int_bin_width) * int_bin_width
    min_tau_1_bin  = int(round(int_min_tau_1/int_bin_width))
    
    int_min_tau_2 = round(min_tau_2 / int_bin_width) * int_bin_width
    min_tau_2_bin  = int(round(int_min_tau_2/int_bin_width))

    int_max_tau_1 = round(max_tau_1 / int_bin_width) * int_bin_width
    max_tau_1_bin  = int(round(int_max_tau_1/int_bin_width))

    int_max_tau_2 = round(max_tau_2 / int_bin_width) * int_bin_width
    max_tau_2_bin  = int(round(int_max_tau_2/int_bin_width))

    num_files = len(file_list)
    numer_list = [int(0)]*(max_tau_1_bin - min_tau_1_bin + 1)*(max_tau_2_bin - min_tau_2_bin + 1)

    denom_ctypes = ctypes.c_int(0)
    ctypes_file_list = file_list_to_ctypes(file_list, folder_name)
    lib = ctypes.CDLL(working_directory + '/' + lib_name)
    lib.getG2Correlations_pulse.argtypes = [ctypes.c_char_p * num_files, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.py_object, ctypes.POINTER(ctypes.c_int),ctypes.c_int]
    start_time = time.time()
    #with wurlitzer.sys_pipes():
    lib.getG2Correlations_pulse(ctypes_file_list, num_files, int_min_tau_1, int_max_tau_1, int_min_tau_2, int_max_tau_2, int_bin_width, numer_list, ctypes.byref(denom_ctypes),4)
    print("Finished in " + str(time.time()-start_time) + "s")
    time.sleep(1)
    if os.name == 'nt':
        _ctypes.FreeLibrary(lib._handle)
    else:
        _ctypes.dlclose(lib._handle)
    tau_1 = np.arange(min_tau_1_bin,max_tau_1_bin+1) * int_bin_width
    tau_2 = np.arange(min_tau_2_bin,max_tau_2_bin+1) * int_bin_width
    scipy.io.savemat(file_out_name,{'numer':np.array(numer_list).reshape(max_tau_2_bin-min_tau_2_bin+1,max_tau_1_bin-min_tau_1_bin+1),'tau_1':tau_1,'tau_2':tau_2})
    
def processFiles(g2_proccessing,g3_proccessing,folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance,calc_norm,update,disp_counts=False):
    dict = {}
    if g2_proccessing:
        g2_dict = g2ToDict(folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance,calc_norm,update,disp_counts)
        dict = {**dict, **g2_dict}
    if g3_proccessing:
        g3_dict = g3ToDict(folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance,calc_norm,update)
        dict = {**dict, **g3_dict}
    scipy.io.savemat(file_out_name,dict)