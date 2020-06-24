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

def processFiles(g2_proccessing,g3_proccessing,folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance,update):
    dict = {}
    if g2_proccessing:
        g2_dict = g2ToFile(folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance,update)
        dict = {**dict, **g2_dict}
    if g3_proccessing:
        g3_dict = g3ToFile(folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance,update)
        dict = {**dict, **g3_dict}
    scipy.io.savemat(file_out_name,dict)

def processEtaFiles(g2_proccessing,g3_proccessing,folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance,update):
    dict = {}
    if g2_proccessing:
        g2_dict = g2ToFile(folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance,update)
        dict = {**dict, **g2_dict}
    if g3_proccessing:
        g3_dict = g3ToFile_single_tau1(folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance,update,0)
        dict = {**dict, **g3_dict}
    scipy.io.savemat(file_out_name,dict)

def file_list_to_ctypes(file_list,folder):
    len_list = len(file_list)
    str_array_type = ctypes.c_char_p * len_list
    str_array = str_array_type()
    for i, file in enumerate(file_list):
        str_array[i] = (folder+file).encode('utf-8')
    return str_array

def g2ToFile(folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance,update=False):
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
    lib.getG2Correlations.argtypes = [ctypes.c_char_p * num_files, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.py_object, ctypes.POINTER(ctypes.c_int)]
    start_time = time.time()
    #with wurlitzer.sys_pipes():
    #Call the DLL
    lib.getG2Correlations(ctypes_file_list, num_files, int_max_time, int_bin_width, int_pulse_spacing, max_pulse_distance, numer_list, ctypes.byref(denom_ctypes))
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

def g3ToFile(folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance,update=False):
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
    lib.getG3Correlations.argtypes = [ctypes.c_char_p * num_files, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.py_object, ctypes.POINTER(ctypes.c_int)]
    start_time = time.time()
    #with wurlitzer.sys_pipes():
    #Call the DLL
    lib.getG3Correlations(ctypes_file_list, num_files, int_max_time, int_bin_width, int_pulse_spacing, max_pulse_distance, numer_list, ctypes.byref(denom_ctypes))
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

def g3ToFile_single_tau1(folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance,tau1,update=False):
    #Convert various parameters to their integer values for the DLL
    int_bin_width = round(bin_width / tagger_resolution) * tagger_resolution
    int_max_time = round(max_time / int_bin_width) * int_bin_width
    int_pulse_spacing = round(pulse_spacing / int_bin_width) * int_bin_width
    int_tau1 = round(tau1/int_bin_width)* int_bin_width
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
            old_denom = old_mat['denom_g3_single'][0][0]
            old_numer = old_mat['numer_g3_single'][0]
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
    lib.getG3Correlations_single_tau1.argtypes = [ctypes.c_char_p * num_files, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.py_object, ctypes.POINTER(ctypes.c_int),ctypes.c_double]
    start_time = time.time()
    #with wurlitzer.sys_pipes():
    #Call the DLL
    lib.getG3Correlations_single_tau1(ctypes_file_list, num_files, int_max_time, int_bin_width, int_pulse_spacing, max_pulse_distance, numer_list, ctypes.byref(denom_ctypes),int_tau1)
    print("Finished in " + str(time.time()-start_time) + "s")
    
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
            output_dict = {'numer_g3_single':np.array(numer_list)+old_numer,'denom_g3_single':denom_ctypes.value+old_denom,'tau':tau,'file_list':dir_file_list}
        except:
            print("Could not update values")
    else:
        output_dict = {'numer_g3_single':np.array(numer_list),'denom_g3_single':denom_ctypes.value,'tau':tau,'file_list':dir_file_list}
    return output_dict