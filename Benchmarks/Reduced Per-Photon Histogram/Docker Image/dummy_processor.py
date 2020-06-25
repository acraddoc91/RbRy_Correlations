import corrLib
import sys

if __name__=="__main__":
        g2_proccessing = bool(sys.argv[2])
        g3_proccessing = bool(sys.argv[3])
        folder_name = sys.argv[4]
        file_out_name = sys.argv[5]
        max_time = float(sys.argv[6])
        bin_width = float(sys.argv[7])
        pulse_spacing = float(sys.argv[8])
        max_pulse_distance = int(sys.argv[9])
        calc_norm = bool(sys.argv[10])
        update = bool(sys.argv[11])
        dummy = folder_name.split("/Data")
        docker_folder_name = "/NAS/Data"+dummy[1]
        dummy = file_out_name.split("/Data")
        docker_file_out_name = "/NAS/Data"+dummy[1]
        corrLib.processFiles(g2_proccessing,g3_proccessing,docker_folder_name,docker_file_out_name,max_time,bin_width,pulse_spacing,max_pulse_distance,calc_norm,update)
