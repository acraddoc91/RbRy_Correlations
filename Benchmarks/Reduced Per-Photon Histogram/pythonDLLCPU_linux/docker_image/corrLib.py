import ctypes
import _ctypes
import multiprocessing
import os
import numpy as np
import scipy.io
import time
import psutil

tagger_resolution = 82.3e-12*2
#num_cpu = multiprocessing.cpu_count()

def max_threads_allowed(array_size_list, bit_size):
    num_cpu = multiprocessing.cpu_count()
    available_memory = psutil.virtual_memory().available
    
    size_in_memory = bit_size/4
    for array_dim_size in array_size_list:
        size_in_memory *= array_dim_size
    
    max_num_arrays_from_memory = int(np.floor(available_memory/size_in_memory))
    max_threads = min(num_cpu,max_num_arrays_from_memory)

    return max_threads

working_directory = "/usr/local/lib"
lib_name = 'pythonDLLCPU.so'

def file_list_to_ctypes(file_list,folder):
    len_list = len(file_list)
    str_array_type = ctypes.c_char_p * len_list
    str_array = str_array_type()
    for i, file in enumerate(file_list):
        str_array[i] = (folder+file).encode('utf-8')
    return str_array

def g2ToDict_pairwise(dir_file_list,folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance,calc_norm=True,update=False,disp_counts=False,pairwise_channel_list=[[3,5]],offset_list=[[3,0],[5,0],[8,0]]):
    
    #Convert various parameters to their integer values for the DLL
    int_bin_width = round(bin_width / tagger_resolution) * tagger_resolution
    int_max_time = round(max_time / int_bin_width) * int_bin_width
    int_pulse_spacing = round(pulse_spacing / int_bin_width) * int_bin_width
    max_bin  = int(round(int_max_time/int_bin_width))
    numer_list = [int(0)]*(2*max_bin + 1) * len(pairwise_channel_list)
    denom_list = [int(0)] * len(pairwise_channel_list)
    #Calculate what tau should look like
    tau = np.arange(-max_bin,max_bin+1) * int_bin_width
    
    #Previous values to add to new calculated total
    old_denom = []
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
            old_denom = old_mat['denom_g2'][0]
            old_numer = old_mat['numer_g2']
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

    max_threads = max_threads_allowed([len(numer_list)],32)
    if max_threads > 0:

        #Convert things to C versions for the DLL
        ctypes_file_list = file_list_to_ctypes(file_list, folder_name)
        #Setup the DLL
        lib = ctypes.CDLL(working_directory + '/' + lib_name)
        lib.getG2Correlations_pairwise.argtypes = [ctypes.c_char_p * num_files, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.py_object, ctypes.py_object, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.py_object, ctypes.c_bool, ctypes.py_object]
        start_time = time.time()
        #Call the DLL
        lib.getG2Correlations_pairwise(ctypes_file_list, num_files, int_max_time, int_bin_width, int_pulse_spacing, max_pulse_distance, numer_list, denom_list,calc_norm,max_threads,1,pairwise_channel_list, disp_counts, offset_list)
        print("Finished g2 in " + str(time.time()-start_time) + "s")
        
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
                output_dict = {'numer_g2':np.reshape(np.array(numer_list),(len(pairwise_channel_list),len(tau)))+old_numer,'denom_g2':denom_list+old_denom,'tau':tau,'file_list':dir_file_list, 'pairwise_channel_list': pairwise_channel_list}
            except:
                print("Could not update values")
        else:
            output_dict = {'numer_g2':np.reshape(np.array(numer_list),(len(pairwise_channel_list),len(tau))),'denom_g2':denom_list,'tau':tau,'file_list':dir_file_list, 'pairwise_channel_list': pairwise_channel_list}
        return output_dict
    else:
        return 1

def g3ToDict_tripwise(dir_file_list,folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance,calc_norm=True,update=False,disp_counts=False,tripwise_channel_list=[[3,5,8]],offset_list=[[3,0],[5,0],[8,0]]):
    #Convert various parameters to their integer values for the DLL
    int_bin_width = round(bin_width / tagger_resolution) * tagger_resolution
    int_max_time = round(max_time / int_bin_width) * int_bin_width
    int_pulse_spacing = round(pulse_spacing / int_bin_width) * int_bin_width
    max_bin  = int(round(int_max_time/int_bin_width))
    numer_list = [int(0)]*(2*max_bin + 1)*(2*max_bin + 1) * len(tripwise_channel_list)
    denom_list = [int(0)] * len(tripwise_channel_list)
    #Calculate what tau should look like
    tau = np.arange(-max_bin,max_bin+1) * int_bin_width

    #Previous values to add to new calculated total
    old_denom = []
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
            old_denom = old_mat['denom_g3'][0]
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
    
    max_threads = max_threads_allowed([len(numer_list)],32)
    if max_threads > 0:

        #Convert things to C versions for the DLL
        denom_ctypes = ctypes.c_int(0)
        ctypes_file_list = file_list_to_ctypes(file_list, folder_name)
        #Setup the DLL
        lib = ctypes.CDLL(working_directory + '/' + lib_name)
        lib.getG3Correlations_tripletwise.argtypes = [ctypes.c_char_p * num_files, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.py_object, ctypes.py_object, ctypes.c_int, ctypes.c_int,ctypes.c_int, ctypes.py_object, ctypes.c_bool, ctypes.py_object]
        start_time = time.time()
        #Call the DLL
        lib.getG3Correlations_tripletwise(ctypes_file_list, num_files, int_max_time, int_bin_width, int_pulse_spacing, max_pulse_distance, numer_list, denom_list,calc_norm,max_threads,1, tripwise_channel_list, disp_counts, offset_list)
        print("Finished g3 in " + str(time.time()-start_time) + "s")
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
                output_dict = {'numer_g3':np.array(numer_list).reshape(len(tripwise_channel_list),len(tau),len(tau))+old_numer,'denom_g3':denom_list +old_denom,'tau':tau,'file_list':dir_file_list, 'tripwise_channel_list': tripwise_channel_list}
            except:
                print("Could not update values")
        else:
            output_dict = {'numer_g3':np.array(numer_list).reshape(len(tripwise_channel_list),len(tau),len(tau)),'denom_g3':denom_list,'tau':tau,'file_list':dir_file_list, 'tripwise_channel_list': tripwise_channel_list}
        return output_dict
    else:
        print("Couldn't start enough threads to run g3, too little memory or too few CPU available")
        return 1

def g3ToDict_tripwise_with_taus(dir_file_list,folder_name,file_out_name,min_time_2,max_time_2,min_time_3,max_time_3,bin_width,pulse_spacing,max_pulse_distance,calc_norm=True,update=False,disp_counts=False,tripwise_channel_list=[[3,5,8]],offset_list=[[3,0],[5,0],[8,0]]):
    #Convert various parameters to their integer values for the DLL
    int_bin_width = round(bin_width / tagger_resolution) * tagger_resolution
    int_max_time_2 = round(max_time_2 / int_bin_width) * int_bin_width
    int_min_time_2 = round(min_time_2 / int_bin_width) * int_bin_width
    int_max_time_3 = round(max_time_3 / int_bin_width) * int_bin_width
    int_min_time_3 = round(min_time_3 / int_bin_width) * int_bin_width
    int_pulse_spacing = round(pulse_spacing / int_bin_width) * int_bin_width
    max_bin_2  = int(round(int_max_time_2/int_bin_width))
    min_bin_2  = int(round(int_min_time_2/int_bin_width))
    max_bin_3  = int(round(int_max_time_3/int_bin_width))
    min_bin_3  = int(round(int_min_time_3/int_bin_width))
    numer_list = [int(0)] * (max_bin_2 - min_bin_2 + 1) * (max_bin_3 - min_bin_3 + 1) * len(tripwise_channel_list)
    denom_list = [int(0)] * len(tripwise_channel_list)
    #Calculate what tau should look like
    tau_2 = np.arange(min_bin_2,max_bin_2+1) * int_bin_width
    tau_3 = np.arange(min_bin_3,max_bin_3+1) * int_bin_width

    #Previous values to add to new calculated total
    old_denom = []
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
            old_denom = old_mat['denom_g3'][0]
            old_numer = old_mat['numer_g3']
            old_tau_2 = old_mat['tau_2'][0]
            old_tau_3 = old_mat['tau_3'][0]
            if not np.array_equal(tau_2,old_tau_2) and not np.array_equal(tau_3,old_tau_3):
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
    
    max_threads = max_threads_allowed([len(numer_list)],32)
    if max_threads > 0:

        #Convert things to C versions for the DLL
        denom_ctypes = ctypes.c_int(0)
        ctypes_file_list = file_list_to_ctypes(file_list, folder_name)
        #Setup the DLL
        lib = ctypes.CDLL(working_directory + '/' + lib_name)
        lib.getG3Correlations_tripletwise_with_tau.argtypes = [ctypes.c_char_p * num_files, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.py_object, ctypes.py_object, ctypes.c_int, ctypes.c_int,ctypes.c_int, ctypes.py_object, ctypes.c_bool, ctypes.py_object]
        start_time = time.time()
        #Call the DLL
        lib.getG3Correlations_tripletwise_with_tau(ctypes_file_list, num_files, int_min_time_2, int_max_time_2, int_min_time_3, int_max_time_3, int_bin_width, int_pulse_spacing, max_pulse_distance, numer_list, denom_list,calc_norm,max_threads,1, tripwise_channel_list, disp_counts, offset_list)
        print("Finished g3 in " + str(time.time()-start_time) + "s")
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
                output_dict = {'numer_g3':np.array(numer_list).reshape(len(tripwise_channel_list),len(tau_3),len(tau_2))+old_numer,'denom_g3':denom_list +old_denom,'tau_2':tau_2, 'tau_3': tau_3,'file_list':dir_file_list, 'tripwise_channel_list': tripwise_channel_list}
            except:
                print("Could not update values")
        else:
            output_dict = {'numer_g3':np.array(numer_list).reshape(len(tripwise_channel_list),len(tau_3),len(tau_2)),'denom_g3':denom_list,'tau_2':tau_2, 'tau_3': tau_3,'file_list':dir_file_list, 'tripwise_channel_list': tripwise_channel_list}
            #output_dict = {'numer_g3':np.array(numer_list),'denom_g3':denom_list,'tau_2':tau_2, 'tau_3': tau_3,'file_list':dir_file_list, 'tripwise_channel_list': tripwise_channel_list}
        return output_dict
    else:
        print("Couldn't start enough threads to run g3, too little memory or too few CPU available")
        return 1

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
    
def processFiles(g2_proccessing,g3_proccessing,folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance,calc_norm,update,disp_counts=False,pairwise_channel_list=[[3,5]],triplewise_channel_list=[[3,5,8]],offset_list=[[3,0],[5,0],[8,0]]):
    dict = {}
    #Get list of files to process from the data folder
    dir_file_list = os.listdir(folder_name)

    #Keep track of whether we can save the dict
    good_to_save = True
    if g2_proccessing:
        if g3_proccessing:
            g2_dict = g2ToDict_pairwise(dir_file_list,folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance,calc_norm,update,False,pairwise_channel_list,offset_list)
            #Check if the g2 processing completed or failed because of a memory or CPU error
            if g2_dict != 1:
                dict = {**dict, **g2_dict}
            else:
                good_to_save = False
        else:
            g2_dict = g2ToDict_pairwise(dir_file_list,folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance,calc_norm,update,disp_counts,pairwise_channel_list,offset_list)
            #Check if the g2 processing completed or failed because of a memory or CPU error
            if g2_dict != 1:
                dict = {**dict, **g2_dict}
            else:
                good_to_save = False
    if g3_proccessing:
        g3_dict = g3ToDict_tripwise(dir_file_list,folder_name,file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance,calc_norm,update,disp_counts,triplewise_channel_list,offset_list)
        #Check if the g3 processing completed or failed because of a memory or CPU error
        if g3_dict != 1:
            dict = {**dict, **g3_dict}
        else:
                good_to_save = False
    #Only save if it's safe too
    if good_to_save:
        scipy.io.savemat(file_out_name,dict)

def processFiles_with_taus(g2_proccessing,g3_proccessing,folder_name,file_out_name,max_time_g2,min_time_2,max_time_2,min_time_3,max_time_3,bin_width,pulse_spacing,max_pulse_distance,calc_norm,update,disp_counts=False,pairwise_channel_list=[[3,5]],triplewise_channel_list=[[3,5,8]],offset_list=[[3,0],[5,0],[8,0]]):
    dict = {}
    #Get list of files to process from the data folder
    dir_file_list = os.listdir(folder_name)

    #Keep track of whether we can save the dict
    good_to_save = True
    if g2_proccessing:
        if g3_proccessing:
            g2_dict = g2ToDict_pairwise(dir_file_list,folder_name,file_out_name,max_time_2,bin_width,pulse_spacing,max_pulse_distance,calc_norm,update,False,pairwise_channel_list,offset_list)
            #Check if the g2 processing completed or failed because of a memory or CPU error
            if g2_dict != 1:
                dict = {**dict, **g2_dict}
            else:
                good_to_save = False
        else:
            g2_dict = g2ToDict_pairwise(dir_file_list,folder_name,file_out_name,max_time_2,bin_width,pulse_spacing,max_pulse_distance,calc_norm,update,disp_counts,pairwise_channel_list,offset_list)
            #Check if the g2 processing completed or failed because of a memory or CPU error
            if g2_dict != 1:
                dict = {**dict, **g2_dict}
            else:
                good_to_save = False
    if g3_proccessing:
        g3_dict = g3ToDict_tripwise_with_taus(dir_file_list,folder_name,file_out_name,min_time_2,max_time_2,min_time_3,max_time_3,bin_width,pulse_spacing,max_pulse_distance,calc_norm,update,disp_counts,triplewise_channel_list,offset_list)
        #Check if the g3 processing completed or failed because of a memory or CPU error
        if g3_dict != 1:
            dict = {**dict, **g3_dict}
        else:
                good_to_save = False
    #Only save if it's safe too
    if good_to_save:
        scipy.io.savemat(file_out_name,dict)

def countTags(folder_name):
    file_list = os.listdir(folder_name)
    num_files = len(file_list)
    ctypes_file_list = file_list_to_ctypes(file_list, folder_name)
    lib = ctypes.CDLL(working_directory + '/' + lib_name)
    lib.getCounts.argtypes = [ctypes.c_char_p * num_files, ctypes.c_int, ctypes.c_int]
    start_time = time.time()
    lib.getCounts(ctypes_file_list, num_files,16)
    print("Finished in " + str(time.time()-start_time) + "s")
    time.sleep(1)

def countTags_and_return(folder_name):
    file_list = os.listdir(folder_name)
    num_files = len(file_list)
    ctypes_file_list = file_list_to_ctypes(file_list, folder_name)
    lib = ctypes.CDLL(working_directory + '/' + lib_name)
    channel_list = [3,5,8]
    tot_time = ctypes.c_double(0)
    mask_time = ctypes.c_double(0)
    tot_counts = [0] * len(channel_list)
    mask_counts = [0] * len(channel_list)
    mask_block_counts = [0] * len(channel_list)
    
    lib.getCounts_and_return.argtypes = [ctypes.c_char_p * num_files, ctypes.c_int, ctypes.c_int, ctypes.py_object, ctypes.py_object, ctypes.py_object, ctypes.py_object, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
    start_time = time.time()
    lib.getCounts_and_return(ctypes_file_list, num_files,16, channel_list, mask_counts, tot_counts, mask_block_counts, ctypes.byref(tot_time), ctypes.byref(mask_time))
    print("Finished in " + str(time.time()-start_time) + "s")
    time.sleep(1)
    return (tot_time.value,mask_time.value,tot_counts,mask_counts,mask_block_counts)