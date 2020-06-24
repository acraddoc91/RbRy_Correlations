// pythonDLLCPU.cpp : Defines the exported functions for the DLL application.
//

#if defined(_MSC_VER)
    //  Microsoft
    #define EXPORT __declspec(dllexport)
    #define IMPORT __declspec(dllimport)
#elif defined(__GNUC__)
    //  GCC
    #define EXPORT __attribute__((visibility("default")))
    #define IMPORT
#else
    //  do nothing and hope for the best?
    #define EXPORT
    #define IMPORT
    #pragma warning Unknown dynamic link import/export semantics.
#endif

#ifdef _MSC_VER
	#include "stdafx.h"
#endif

#include "H5Cpp.h"
#include <vector>
#include <H5Exception.h>
#include <map>
#include <Python.h>
#include <math.h>
#include <omp.h>

#if defined(_MSC_VER)
	//Define integers explicitly to prevent problems on different platforms
	#define int8 __int8
	#define int16 __int16
	#define int32 __int32
	#define int64 __int64
#elif defined(__GNUC__)
	#include <inttypes.h>
	#define int8 int8_t
	#define int16 int16_t
	#define int32 int32_t
	#define int64 int64_t
#endif

const int64 max_tags_length = 2000000;
const int64 max_clock_tags_length = 1000000;
const int32 max_channels = 3;
const size_t return_size = 3;
const int32 file_block_size = 16;
const double tagger_resolution = 82.3e-12;
//const int32 offset[3] = { 0, 219, 211 };
//const int64 offset[3] = { 0, -23, 299 };
//const int64 offset[3] = { 25, 0, 311 };
//const int64 offset[3] = {181, 157, 0};

struct shotData {
	bool file_load_completed;
	std::vector<int16> channel_list;
	std::map<int16, int16> channel_map;
	std::vector<int64> start_tags;
	std::vector<int64> end_tags;
	std::vector<int64> photon_tags;
	std::vector<int64> clock_tags;
	std::vector<std::vector<int64>> sorted_photon_tags;
	std::vector<std::vector<int64>> sorted_photon_bins;
	std::vector<std::vector<int64>> sorted_clock_tags;
	std::vector<std::vector<int64>> sorted_clock_bins;
	std::vector<int64> sorted_photon_tag_pointers;
	std::vector<int64> sorted_clock_tag_pointers;

	shotData() : sorted_photon_tags(max_channels, std::vector<int64>(max_tags_length, 0)), sorted_photon_bins(max_channels, std::vector<int64>(max_tags_length, 0)), sorted_photon_tag_pointers(max_channels, 0), sorted_clock_tags(2, std::vector<int64>(max_clock_tags_length, 0)), sorted_clock_bins(2, std::vector<int64>(max_clock_tags_length, 0)), sorted_clock_tag_pointers(2, 0) {}
};

void calculateDenom_g3(shotData *shot_data, int64 *max_bin, int64 *pulse_spacing, int64 *max_pulse_distance, int32 *denom, int32 shot_file_num) {
	
	int64 start_clock = shot_data->sorted_clock_bins[1][0];
	int64 end_clock = shot_data->sorted_clock_bins[0][shot_data->sorted_clock_tag_pointers[0] - 1];

	for (int32 channel_1 = 0; channel_1 < shot_data->channel_list.size(); channel_1++) {
		for (int32 channel_2 = channel_1 + 1; channel_2 < shot_data->channel_list.size(); channel_2++) {
			for (int32 channel_3 = channel_2 + 1; channel_3 < shot_data->channel_list.size(); channel_3++) {
				std::vector<std::vector<int32>> denom_counts(*max_pulse_distance * 2 + 1, std::vector<int32>(*max_pulse_distance * 2 + 1,0));
				#pragma omp parallel for
				for (int64 pulse_dist_1 = -*max_pulse_distance; pulse_dist_1 <= *max_pulse_distance; pulse_dist_1++) {
					for (int64 pulse_dist_2 = -*max_pulse_distance; pulse_dist_2 <= *max_pulse_distance; pulse_dist_2++) {
						if ((pulse_dist_1 != 0) && (pulse_dist_2 != 0) && (pulse_dist_1 != pulse_dist_2)) {
							int64 tau_1 = *pulse_spacing * pulse_dist_1;
							int64 tau_2 = *pulse_spacing * pulse_dist_2;
							int64 i = 0;
							int64 j = 0;
							int64 k = 0;
							while ((i < shot_data->sorted_photon_tag_pointers[channel_1]) && (j < shot_data->sorted_photon_tag_pointers[channel_2]) && (k < shot_data->sorted_photon_tag_pointers[channel_3])) {
								//Check if we're outside the window of interest
								int32 out_window = (shot_data->sorted_photon_bins[channel_1][i] < (*max_bin + *max_pulse_distance * *pulse_spacing + start_clock)) || (shot_data->sorted_photon_bins[channel_1][i] > (end_clock - (*max_bin + *max_pulse_distance * *pulse_spacing)));
								//chan_1 > chan_2
								int32 c1gc2 = shot_data->sorted_photon_bins[channel_1][i] > (shot_data->sorted_photon_bins[channel_2][j] - tau_1);
								//Chan_1 > chan_3
								int32 c1gc3 = shot_data->sorted_photon_bins[channel_1][i] > (shot_data->sorted_photon_bins[channel_3][k] - tau_2);
								//Chan_1 == chan_2
								int32 c1ec2 = shot_data->sorted_photon_bins[channel_1][i] == (shot_data->sorted_photon_bins[channel_2][j] - tau_1);
								//Chan_1 == chan_3
								int32 c1ec3 = shot_data->sorted_photon_bins[channel_1][i] == (shot_data->sorted_photon_bins[channel_3][k] - tau_2);

								//Increment the running total if all three are equal and we're in the window
								denom_counts[pulse_dist_1 + *max_pulse_distance][pulse_dist_2 + *max_pulse_distance] += !out_window && c1ec2 && c1ec3;

								//Increment j if chan_2 < chan_1 or all three are equal
								j += c1gc2 || (c1ec2 && c1ec3);
								//Increment k if chan_3 < chan_1 or all three are equal
								k += c1gc3 || (c1ec2 && c1ec3);

								//Increment i if we're out the window or if chan_1 <= chan_2 and chan_1 <= chan_3
								i += out_window || (!c1gc2 && !c1gc3);
							}
						}
					}
				}
				for (int64 pulse_dist_1 = -*max_pulse_distance; pulse_dist_1 <= *max_pulse_distance; pulse_dist_1++) {
					for (int64 pulse_dist_2 = -*max_pulse_distance; pulse_dist_2 <= *max_pulse_distance; pulse_dist_2++) {
						denom[0] += denom_counts[pulse_dist_1 + *max_pulse_distance][pulse_dist_2 + *max_pulse_distance];
					}
				}
			}
		}
	}

}

void calculateDenom_g3_for_channel_trip(shotData *shot_data, int64 *max_bin, int64 *pulse_spacing, int64 *max_pulse_distance, int32 *denom, int32 shot_file_num, int32 channel_1, int32 channel_2, int32 channel_3) {
	
	int64 start_clock = shot_data->sorted_clock_bins[1][0];
	int64 end_clock = shot_data->sorted_clock_bins[0][shot_data->sorted_clock_tag_pointers[0] - 1];
	std::vector<std::vector<int32>> denom_counts(*max_pulse_distance * 2 + 1, std::vector<int32>(*max_pulse_distance * 2 + 1,0));
	#pragma omp parallel for
	for (int64 pulse_dist_1 = -*max_pulse_distance; pulse_dist_1 <= *max_pulse_distance; pulse_dist_1++) {
		for (int64 pulse_dist_2 = -*max_pulse_distance; pulse_dist_2 <= *max_pulse_distance; pulse_dist_2++) {
			if ((pulse_dist_1 != 0) && (pulse_dist_2 != 0) && (pulse_dist_1 != pulse_dist_2)) {
				int64 tau_1 = *pulse_spacing * pulse_dist_1;
				int64 tau_2 = *pulse_spacing * pulse_dist_2;
				int64 i = 0;
				int64 j = 0;
				int64 k = 0;
				while ((i < shot_data->sorted_photon_tag_pointers[channel_1]) && (j < shot_data->sorted_photon_tag_pointers[channel_2]) && (k < shot_data->sorted_photon_tag_pointers[channel_3])) {
					//Check if we're outside the window of interest
					int32 out_window = (shot_data->sorted_photon_bins[channel_1][i] < (*max_bin + *max_pulse_distance * *pulse_spacing + start_clock)) || (shot_data->sorted_photon_bins[channel_1][i] > (end_clock - (*max_bin + *max_pulse_distance * *pulse_spacing)));
					//chan_1 > chan_2
					int32 c1gc2 = shot_data->sorted_photon_bins[channel_1][i] > (shot_data->sorted_photon_bins[channel_2][j] - tau_1);
					//Chan_1 > chan_3
					int32 c1gc3 = shot_data->sorted_photon_bins[channel_1][i] > (shot_data->sorted_photon_bins[channel_3][k] - tau_2);
					//Chan_1 == chan_2
					int32 c1ec2 = shot_data->sorted_photon_bins[channel_1][i] == (shot_data->sorted_photon_bins[channel_2][j] - tau_1);
					//Chan_1 == chan_3
					int32 c1ec3 = shot_data->sorted_photon_bins[channel_1][i] == (shot_data->sorted_photon_bins[channel_3][k] - tau_2);

					//Increment the running total if all three are equal and we're in the window
					denom_counts[pulse_dist_1 + *max_pulse_distance][pulse_dist_2 + *max_pulse_distance] += !out_window && c1ec2 && c1ec3;

					//Increment j if chan_2 < chan_1 or all three are equal
					j += c1gc2 || (c1ec2 && c1ec3);
					//Increment k if chan_3 < chan_1 or all three are equal
					k += c1gc3 || (c1ec2 && c1ec3);

					//Increment i if we're out the window or if chan_1 <= chan_2 and chan_1 <= chan_3
					i += out_window || (!c1gc2 && !c1gc3);
				}
			}
		}
	}
	for (int64 pulse_dist_1 = -*max_pulse_distance; pulse_dist_1 <= *max_pulse_distance; pulse_dist_1++) {
		for (int64 pulse_dist_2 = -*max_pulse_distance; pulse_dist_2 <= *max_pulse_distance; pulse_dist_2++) {
			denom[0] += denom_counts[pulse_dist_1 + *max_pulse_distance][pulse_dist_2 + *max_pulse_distance];
		}
	}
}

int64 first_above_binary_search(std::vector<int64> *a, int64 N, int64 b) {
	int64 L = 0;
	int64 R = N - 1;
	int64 return_val;
	if ((*a)[N - 1] < b) {
		return_val = N;
	}
	else {
		while (L <= R) {
			int64 m = (int64)floor((L + R) / 2);
			if ((*a)[m] < b) {
				L = m + 1;
			}
			else if (m == 0) {
				return_val = m;
				L = R + 1;
			}
			else if (((*a)[m] > b) && !((*a)[m - 1] < b)) {
				R = m - 1;
			}
			else {
				return_val = m;
				L = R + 1;
			}
		}
	}
	return return_val;
}

int64 first_below_binary_search(std::vector<int64> *a, int64 N, int64 b) {
	int64 L = 0;
	int64 R = N - 1;
	int64 return_val;
	if ((*a)[0] > b) {
		return_val = -1;
	}
	else {
		while (L <= R) {
			int64 m = (int64)floor((L + R) / 2);
			if (m == N - 1) {
				return m;
				L = R + 1;
			}
			else if (((*a)[m] < b) && !((*a)[m + 1] > b)) {
				L = m + 1;
			}
			else if ((*a)[m] > b) {
				R = m - 1;
			}
			else {
				return_val = m;
				L = R + 1;
			}
		}
	}
	return return_val;
}

void calculateNumer_g3_for_channel_trip(shotData *shot_data, int64 *max_bin, int64 *pulse_spacing, int64 *max_pulse_distance, int32 *coinc, int32 shot_file_num, int32 num_cpu_threads_proc, int32 channel_1, int32 channel_2, int32 channel_3) {

	//Get the start and stop clock bin
	int64 start_clock = shot_data->sorted_clock_bins[1][0];
	int64 end_clock = shot_data->sorted_clock_bins[0][shot_data->sorted_clock_tag_pointers[0] - 1];

	//Make sure we've actually got some tags in channel 1
	if((shot_data->sorted_photon_tag_pointers[channel_1] > 0) && (shot_data->sorted_photon_tag_pointers[channel_2] > 0) && (shot_data->sorted_photon_tag_pointers[channel_3] > 0)){
		//Make sure we've actually got some tags in channel 1
		//Let's find out which indicies from channel 1 we can discard due to them being outside of the window of interest
		int64 low_index;
		int64 high_index;
		//Figure out which indices in the first thread we can ignore
		#pragma omp parallel for
		for (int8 i = 0; i < 2; i++) {
			if (i == 0) {
				low_index = first_above_binary_search(&(shot_data->sorted_photon_bins[channel_1]), shot_data->sorted_photon_tag_pointers[channel_1], *max_bin + *max_pulse_distance * *pulse_spacing + start_clock);
			}
			else {
				high_index = first_below_binary_search(&(shot_data->sorted_photon_bins[channel_1]), shot_data->sorted_photon_tag_pointers[channel_1], end_clock - (*max_bin + *max_pulse_distance * *pulse_spacing));
			}
		}

		//Split the remaining indices between the work threads we have
		int64 indices_per_thread = (high_index - low_index) / num_cpu_threads_proc;
		//Vector to hold the relevant indices on channel 2 and 3, for each tag on channel 1, that falls with the max/min tau
		std::vector<std::vector<int64>> channel_2_indices(high_index - low_index + 1, std::vector<int64>(2));
		std::vector<std::vector<int64>> channel_3_indices(high_index - low_index + 1, std::vector<int64>(2));
		#pragma omp parallel for num_threads(num_cpu_threads_proc)
		for (int32 thread = 0; thread < num_cpu_threads_proc; thread++) {
			//Find out form this thread what the first and last indices to work on are
			int64 first_index = thread*indices_per_thread + low_index;
			int64 last_index;
			if (thread == num_cpu_threads_proc - 1) {
				last_index = high_index;
			}
			else {
				last_index = first_index + indices_per_thread - 1;
			}

			//Do a binary search to find the first and last relevant tag on channel 2 and 3 for the first tag on channel 1 that the thread is working on
			int64 lower_pointer_2 = first_above_binary_search(&(shot_data->sorted_photon_bins[channel_2]), shot_data->sorted_photon_tag_pointers[channel_2], shot_data->sorted_photon_bins[channel_1][first_index] - *max_bin);
			int64 upper_pointer_2 = first_below_binary_search(&(shot_data->sorted_photon_bins[channel_2]), shot_data->sorted_photon_tag_pointers[channel_2], shot_data->sorted_photon_bins[channel_1][first_index] + *max_bin);
			int64 lower_pointer_3 = first_above_binary_search(&(shot_data->sorted_photon_bins[channel_3]), shot_data->sorted_photon_tag_pointers[channel_3], shot_data->sorted_photon_bins[channel_1][first_index] - *max_bin);
			int64 upper_pointer_3 = first_below_binary_search(&(shot_data->sorted_photon_bins[channel_3]), shot_data->sorted_photon_tag_pointers[channel_3], shot_data->sorted_photon_bins[channel_1][first_index] + *max_bin);

			//Save the relevant tags on channel 2 and 3
			channel_2_indices[first_index - low_index][0] = lower_pointer_2;
			channel_2_indices[first_index - low_index][1] = upper_pointer_2;
			channel_3_indices[first_index - low_index][0] = lower_pointer_3;
			channel_3_indices[first_index - low_index][1] = upper_pointer_3;

			for (int64 i = first_index; i <= last_index; i++) {
				//Find the first tag on channel 2 that is within the max/min tau of the current channel 1 tag
				bool going = true;
				int64 j = lower_pointer_2;
				while (going) {
					if (shot_data->sorted_photon_bins[channel_2][j] < shot_data->sorted_photon_bins[channel_1][i] - *max_bin) {
						j++;
						lower_pointer_2 = j;
					}
					else if (shot_data->sorted_photon_bins[channel_2][j] >= shot_data->sorted_photon_bins[channel_1][i] - *max_bin) {
						going = false;
						lower_pointer_2 = j;
					}
					if (j > shot_data->sorted_photon_tag_pointers[channel_2]) {
						going = false;
						lower_pointer_2 = j;
					}
				}
				//Find the last tag on channel 2 that is within the max/min tau of the current channel 1 tag
				j = upper_pointer_2;
				//Ensure j can't be negative
				if(j < 0){
					j = 0;
				}
				going = true;
				while (going) {
					if (shot_data->sorted_photon_bins[channel_2][j] <= shot_data->sorted_photon_bins[channel_1][i] + *max_bin) {
						j++;
						upper_pointer_2 = j;
					}
					else if (shot_data->sorted_photon_bins[channel_2][j] > shot_data->sorted_photon_bins[channel_1][i] + *max_bin) {
						going = false;
						upper_pointer_2 = j - 1;
					}
					if (j > shot_data->sorted_photon_tag_pointers[channel_2]) {
						going = false;
						upper_pointer_2 = shot_data->sorted_photon_tag_pointers[channel_2] - 1;
					}
				}
				//Save the relevant tags to the vector for later
				channel_2_indices[i - low_index][0] = lower_pointer_2;
				channel_2_indices[i - low_index][1] = upper_pointer_2;

				//Find the first tag on channel 3 that is within the max/min tau of the current channel 1 tag
				going = true;
				j = lower_pointer_3;
				while (going) {
					if (shot_data->sorted_photon_bins[channel_3][j] < shot_data->sorted_photon_bins[channel_1][i] - *max_bin) {
						j++;
						lower_pointer_3 = j;
					}
					else if (shot_data->sorted_photon_bins[channel_3][j] >= shot_data->sorted_photon_bins[channel_1][i] - *max_bin) {
						going = false;
						lower_pointer_3 = j;
					}
					if (j > shot_data->sorted_photon_tag_pointers[channel_3]) {
						going = false;
						lower_pointer_3 = j;
					}
				}
				//Find the last tag on channel 2 that is within the max/min tau of the current channel 1 tag
				j = upper_pointer_3;
				//Ensure j can't be negative
				if(j < 0){
					j = 0;
				}
				going = true;
				while (going) {
					if (shot_data->sorted_photon_bins[channel_3][j] <= shot_data->sorted_photon_bins[channel_1][i] + *max_bin) {
						j++;
						upper_pointer_3 = j;
					}
					else if (shot_data->sorted_photon_bins[channel_3][j] > shot_data->sorted_photon_bins[channel_1][i] + *max_bin) {
						going = false;
						upper_pointer_3 = j - 1;
					}
					if (j > shot_data->sorted_photon_tag_pointers[channel_3]) {
						going = false;
						upper_pointer_3 = shot_data->sorted_photon_tag_pointers[channel_3] - 1;
					}
				}
				//Save the relevant tags to the vector for later
				channel_3_indices[i - low_index][0] = lower_pointer_3;
				channel_3_indices[i - low_index][1] = upper_pointer_3;
			}
			//Loop through all the tags on the thread has worked on to find the tau bin which the tags on channel 2 fall into
			for (int64 i = first_index; i <= last_index; i++) {
				for (int64 j = channel_2_indices[i - low_index][0]; j <= channel_2_indices[i - low_index][1]; j++) {
					for (int64 k = channel_3_indices[i - low_index][0]; k <= channel_3_indices[i - low_index][1]; k++) {

						int64 id_x = shot_data->sorted_photon_bins[channel_2][j] - shot_data->sorted_photon_bins[channel_1][i] + *max_bin;
						int64 id_y = shot_data->sorted_photon_bins[channel_3][k] - shot_data->sorted_photon_bins[channel_1][i] + *max_bin;
						int64 tot_id = id_y * (2 * (*max_bin) + 1) + id_x;
						coinc[tot_id + thread * ((*max_bin * 2 + 1) * (*max_bin * 2 + 1)) + shot_file_num * num_cpu_threads_proc  * ((*max_bin * 2 + 1) * (*max_bin * 2 + 1))]++;

					}
				}
			}
		}
	}
}

void calculateNumer_g3_for_channel_trip_with_taus(shotData *shot_data, int64 *min_bin_2, int64 *max_bin_2, int64 *min_bin_3, int64 *max_bin_3, int64 *pulse_spacing, int64 *max_pulse_distance, int32 *coinc, int32 shot_file_num, int32 num_cpu_threads_proc, int32 channel_1, int32 channel_2, int32 channel_3) {

	//Get the start and stop clock bin
	int64 start_clock = shot_data->sorted_clock_bins[1][0];
	int64 end_clock = shot_data->sorted_clock_bins[0][shot_data->sorted_clock_tag_pointers[0] - 1];

	//Make sure we've actually got some tags in channel 1
	if((shot_data->sorted_photon_tag_pointers[channel_1] > 0) && (shot_data->sorted_photon_tag_pointers[channel_2] > 0) && (shot_data->sorted_photon_tag_pointers[channel_3] > 0)){
		//Make sure we've actually got some tags in channel 1
		//Let's find out which indicies from channel 1 we can discard due to them being outside of the window of interest
		int64 low_index;
		int64 high_index;

		//Work out overall minimum bin
		int64 min_bin = 0;
		if (*min_bin_2 < *min_bin_3){
			min_bin = *min_bin_2;
		}
		else{
			min_bin = *min_bin_3;
		}
		//And overall maximum bin
		int64 max_bin = 0;
		if (*max_bin_2 > *max_bin_3){
			max_bin = *max_bin_2;
		}
		else{
			max_bin = *max_bin_3;
		}


		//Figure out which indices in the first thread we can ignore
		#pragma omp parallel for
		for (int8 i = 0; i < 2; i++) {
			if (i == 0) {
				low_index = first_above_binary_search(&(shot_data->sorted_photon_bins[channel_1]), shot_data->sorted_photon_tag_pointers[channel_1], -min_bin + *max_pulse_distance * *pulse_spacing + start_clock);
			}
			else {
				high_index = first_below_binary_search(&(shot_data->sorted_photon_bins[channel_1]), shot_data->sorted_photon_tag_pointers[channel_1], end_clock - (max_bin + *max_pulse_distance * *pulse_spacing));
			}
		}

		//Split the remaining indices between the work threads we have
		int64 indices_per_thread = (high_index - low_index) / num_cpu_threads_proc;
		//Vector to hold the relevant indices on channel 2 and 3, for each tag on channel 1, that falls with the max/min tau
		std::vector<std::vector<int64>> channel_2_indices(high_index - low_index + 1, std::vector<int64>(2));
		std::vector<std::vector<int64>> channel_3_indices(high_index - low_index + 1, std::vector<int64>(2));
		#pragma omp parallel for num_threads(num_cpu_threads_proc)
		for (int32 thread = 0; thread < num_cpu_threads_proc; thread++) {
			//Find out form this thread what the first and last indices to work on are
			int64 first_index = thread*indices_per_thread + low_index;
			int64 last_index;
			if (thread == num_cpu_threads_proc - 1) {
				last_index = high_index;
			}
			else {
				last_index = first_index + indices_per_thread - 1;
			}

			//Do a binary search to find the first and last relevant tag on channel 2 and 3 for the first tag on channel 1 that the thread is working on
			int64 lower_pointer_2 = first_above_binary_search(&(shot_data->sorted_photon_bins[channel_2]), shot_data->sorted_photon_tag_pointers[channel_2], shot_data->sorted_photon_bins[channel_1][first_index] + *min_bin_2);
			int64 upper_pointer_2 = first_below_binary_search(&(shot_data->sorted_photon_bins[channel_2]), shot_data->sorted_photon_tag_pointers[channel_2], shot_data->sorted_photon_bins[channel_1][first_index] + *max_bin_2);
			int64 lower_pointer_3 = first_above_binary_search(&(shot_data->sorted_photon_bins[channel_3]), shot_data->sorted_photon_tag_pointers[channel_3], shot_data->sorted_photon_bins[channel_1][first_index] + *min_bin_3);
			int64 upper_pointer_3 = first_below_binary_search(&(shot_data->sorted_photon_bins[channel_3]), shot_data->sorted_photon_tag_pointers[channel_3], shot_data->sorted_photon_bins[channel_1][first_index] + *max_bin_3);

			//Save the relevant tags on channel 2 and 3
			channel_2_indices[first_index - low_index][0] = lower_pointer_2;
			channel_2_indices[first_index - low_index][1] = upper_pointer_2;
			channel_3_indices[first_index - low_index][0] = lower_pointer_3;
			channel_3_indices[first_index - low_index][1] = upper_pointer_3;

			for (int64 i = first_index; i <= last_index; i++) {
				//Find the first tag on channel 2 that is within the max/min tau of the current channel 1 tag
				bool going = true;
				int64 j = lower_pointer_2;
				while (going) {
					if (shot_data->sorted_photon_bins[channel_2][j] < shot_data->sorted_photon_bins[channel_1][i] + *min_bin_2) {
						j++;
						lower_pointer_2 = j;
					}
					else if (shot_data->sorted_photon_bins[channel_2][j] >= shot_data->sorted_photon_bins[channel_1][i] + *min_bin_2) {
						going = false;
						lower_pointer_2 = j;
					}
					if (j > shot_data->sorted_photon_tag_pointers[channel_2]) {
						going = false;
						lower_pointer_2 = j;
					}
				}
				//Find the last tag on channel 2 that is within the max/min tau of the current channel 1 tag
				j = upper_pointer_2;
				//Ensure j can't be negative
				if(j < 0){
					j = 0;
				}
				going = true;
				while (going) {
					if (shot_data->sorted_photon_bins[channel_2][j] <= shot_data->sorted_photon_bins[channel_1][i] + *max_bin_2) {
						j++;
						upper_pointer_2 = j;
					}
					else if (shot_data->sorted_photon_bins[channel_2][j] > shot_data->sorted_photon_bins[channel_1][i] + *max_bin_2) {
						going = false;
						upper_pointer_2 = j - 1;
					}
					if (j > shot_data->sorted_photon_tag_pointers[channel_2]) {
						going = false;
						upper_pointer_2 = shot_data->sorted_photon_tag_pointers[channel_2] - 1;
					}
				}
				//Save the relevant tags to the vector for later
				channel_2_indices[i - low_index][0] = lower_pointer_2;
				channel_2_indices[i - low_index][1] = upper_pointer_2;

				//Find the first tag on channel 3 that is within the max/min tau of the current channel 1 tag
				going = true;
				j = lower_pointer_3;
				while (going) {
					if (shot_data->sorted_photon_bins[channel_3][j] < shot_data->sorted_photon_bins[channel_1][i] + *min_bin_3) {
						j++;
						lower_pointer_3 = j;
					}
					else if (shot_data->sorted_photon_bins[channel_3][j] >= shot_data->sorted_photon_bins[channel_1][i] + *min_bin_3) {
						going = false;
						lower_pointer_3 = j;
					}
					if (j > shot_data->sorted_photon_tag_pointers[channel_3]) {
						going = false;
						lower_pointer_3 = j;
					}
				}
				//Find the last tag on channel 2 that is within the max/min tau of the current channel 1 tag
				j = upper_pointer_3;
				//Ensure j can't be negative
				if(j < 0){
					j = 0;
				}
				going = true;
				while (going) {
					if (shot_data->sorted_photon_bins[channel_3][j] <= shot_data->sorted_photon_bins[channel_1][i] + *max_bin_3) {
						j++;
						upper_pointer_3 = j;
					}
					else if (shot_data->sorted_photon_bins[channel_3][j] > shot_data->sorted_photon_bins[channel_1][i] + *max_bin_3) {
						going = false;
						upper_pointer_3 = j - 1;
					}
					if (j > shot_data->sorted_photon_tag_pointers[channel_3]) {
						going = false;
						upper_pointer_3 = shot_data->sorted_photon_tag_pointers[channel_3] - 1;
					}
				}
				//Save the relevant tags to the vector for later
				channel_3_indices[i - low_index][0] = lower_pointer_3;
				channel_3_indices[i - low_index][1] = upper_pointer_3;
			}
			//Loop through all the tags on the thread has worked on to find the tau bin which the tags on channel 2 fall into
			for (int64 i = first_index; i <= last_index; i++) {
				for (int64 j = channel_2_indices[i - low_index][0]; j <= channel_2_indices[i - low_index][1]; j++) {
					for (int64 k = channel_3_indices[i - low_index][0]; k <= channel_3_indices[i - low_index][1]; k++) {

						int64 id_x = shot_data->sorted_photon_bins[channel_2][j] - shot_data->sorted_photon_bins[channel_1][i] - *min_bin_2;
						int64 id_y = shot_data->sorted_photon_bins[channel_3][k] - shot_data->sorted_photon_bins[channel_1][i] - *min_bin_3;
						int64 tot_id = id_y * (*max_bin_2 - *min_bin_2 + 1) + id_x;
						coinc[tot_id + thread * ((*max_bin_2 - *min_bin_2 + 1) * (*max_bin_3 - *min_bin_3 + 1)) + shot_file_num * num_cpu_threads_proc  * ((*max_bin_2 - *min_bin_2 + 1) * (*max_bin_3 - *min_bin_3 + 1))]++;

					}
				}
			}
		}
	}
}

void calculateNumer_g2_pulse(shotData *shot_data, int64 *min_bin_1, int64 *max_bin_1, int64 *min_bin_2, int64 *max_bin_2, int32 *coinc, int32 shot_file_num) {
	//Get the start and stop clock bin
	int64 start_clock = shot_data->sorted_clock_bins[1][0];
	int64 end_clock = shot_data->sorted_clock_bins[0][shot_data->sorted_clock_tag_pointers[0] - 1];

	//Let's find out which indicies from channel 1 we can discard due to them being outside of the window of interest
	int64 low_index;
	int64 high_index;

	//Check which min and max bin is bigger
	int64 max_bin_check;
	int64 min_bin_check;
	if (*min_bin_1 < *min_bin_2) {
		min_bin_check = *min_bin_1;
	}
	else {
		min_bin_check = *min_bin_2;
	}
	if (*max_bin_1 < *max_bin_2) {
		max_bin_check = *max_bin_2;
	}
	else {
		max_bin_check = *min_bin_1;
	}

	//Figure out which indices in the first list we can ignore
	#pragma omp parallel for
	for (int32 i = 0; i < 2; i++) {
		if (i == 0) {
			low_index = first_above_binary_search(&(shot_data->sorted_clock_bins[1]), shot_data->sorted_clock_tag_pointers[1], -min_bin_check + start_clock);
		}
		else {
			high_index = first_below_binary_search(&(shot_data->sorted_clock_bins[1]), shot_data->sorted_clock_tag_pointers[1], end_clock - max_bin_check);
		}
	}

	for (int32 channel_1 = 0; channel_1 < shot_data->channel_list.size(); channel_1++) {
		for (int32 channel_2 = channel_1 + 1; channel_2 < shot_data->channel_list.size(); channel_2++) {

			//Vector to hold the relevant indices on channel 2 and 3, for each tag on channel 1, that falls with the max/min tau
			std::vector<std::vector<int64>> channel_1_indices(high_index - low_index + 1, std::vector<int64>(2));
			std::vector<std::vector<int64>> channel_2_indices(high_index - low_index + 1, std::vector<int64>(2));
			//Find out form this thread what the first and last indices to work on are
			int64 first_index = low_index;
			int64 last_index = high_index;

			//Do a binary search to find the first and last relevant tag on channel 2 and 3 for the first tag on channel 1 that the thread is working on
			int64 lower_pointer_1 = first_above_binary_search(&(shot_data->sorted_photon_bins[channel_1]), shot_data->sorted_photon_tag_pointers[channel_1], shot_data->sorted_clock_bins[1][first_index] + *min_bin_1);
			int64 upper_pointer_1 = first_below_binary_search(&(shot_data->sorted_photon_bins[channel_1]), shot_data->sorted_photon_tag_pointers[channel_1], shot_data->sorted_clock_bins[1][first_index] + *max_bin_1);
			int64 lower_pointer_2 = first_above_binary_search(&(shot_data->sorted_photon_bins[channel_2]), shot_data->sorted_photon_tag_pointers[channel_2], shot_data->sorted_clock_bins[1][first_index] + *min_bin_2);
			int64 upper_pointer_2 = first_below_binary_search(&(shot_data->sorted_photon_bins[channel_2]), shot_data->sorted_photon_tag_pointers[channel_2], shot_data->sorted_clock_bins[1][first_index] + *max_bin_2);

			//Save the relevant tags on channel 2 and 3
			channel_1_indices[first_index - low_index][0] = lower_pointer_1;
			channel_1_indices[first_index - low_index][1] = upper_pointer_1;
			channel_2_indices[first_index - low_index][0] = lower_pointer_2;
			channel_2_indices[first_index - low_index][1] = upper_pointer_2;

			for (int64 i = first_index; i <= last_index; i++) {
				//Find the first tag on channel 2 that is within the max/min tau of the current channel 1 tag
				bool going = true;
				int64 j = lower_pointer_1;
				while (going) {
					if (shot_data->sorted_photon_bins[channel_1][j] < shot_data->sorted_clock_bins[1][i] + *min_bin_1) {
						j++;
						lower_pointer_1 = j;
					}
					else if (shot_data->sorted_photon_bins[channel_1][j] >= shot_data->sorted_clock_bins[1][i] + *min_bin_1) {
						going = false;
						lower_pointer_1 = j;
					}
					if (j > shot_data->sorted_photon_tag_pointers[channel_1]) {
						going = false;
						lower_pointer_1 = j;
					}
				}
				//Find the last tag on channel 2 that is within the max/min tau of the current channel 1 tag
				j = upper_pointer_1;
				//Ensure j can't be negative
				if(j < 0){
					j = 0;
				}
				going = true;
				while (going) {
					if (shot_data->sorted_photon_bins[channel_1][j] <= shot_data->sorted_clock_bins[1][i] + *max_bin_1) {
						j++;
						upper_pointer_1 = j;
					}
					else if (shot_data->sorted_photon_bins[channel_1][j] > shot_data->sorted_clock_bins[1][i] + *max_bin_1) {
						going = false;
						upper_pointer_1 = j - 1;
					}
					if (j > shot_data->sorted_photon_tag_pointers[channel_1]) {
						going = false;
						upper_pointer_1 = shot_data->sorted_photon_tag_pointers[channel_1] - 1;
					}
				}
				//Save the relevant tags to the vector for later
				channel_1_indices[i - low_index][0] = lower_pointer_1;
				channel_1_indices[i - low_index][1] = upper_pointer_1;

				//Find the first tag on channel 3 that is within the max/min tau of the current channel 1 tag
				going = true;
				j = lower_pointer_2;
				while (going) {
					if (shot_data->sorted_photon_bins[channel_2][j] < shot_data->sorted_clock_bins[1][i] + *min_bin_2) {
						j++;
						lower_pointer_2 = j;
					}
					else if (shot_data->sorted_photon_bins[channel_2][j] >= shot_data->sorted_clock_bins[1][i] + *min_bin_2) {
						going = false;
						lower_pointer_2 = j;
					}
					if (j > shot_data->sorted_photon_tag_pointers[channel_2]) {
						going = false;
						lower_pointer_2 = j;
					}
				}
				//Find the last tag on channel 2 that is within the max/min tau of the current channel 1 tag
				j = upper_pointer_2;
				//Ensure j can't be negative
				if(j < 0){
					j = 0;
				}
				going = true;
				while (going) {
					if (shot_data->sorted_photon_bins[channel_2][j] <= shot_data->sorted_clock_bins[1][i] + *max_bin_2) {
						j++;
						upper_pointer_2 = j;
					}
					else if (shot_data->sorted_photon_bins[channel_2][j] > shot_data->sorted_clock_bins[1][i] + *max_bin_2) {
						going = false;
						upper_pointer_2 = j - 1;
					}
					if (j > shot_data->sorted_photon_tag_pointers[channel_2]) {
						going = false;
						upper_pointer_2 = shot_data->sorted_photon_tag_pointers[channel_2] - 1;
					}
				}
				//Save the relevant tags to the vector for later
				channel_2_indices[i - low_index][0] = lower_pointer_2;
				channel_2_indices[i - low_index][1] = upper_pointer_2;
			}
			//Loop through all the tags on the thread has worked on to find the tau bin which the tags on channel 2 fall into
			for (int64 i = first_index; i <= last_index; i++) {
				for (int64 j = channel_1_indices[i - low_index][0]; j <= channel_1_indices[i - low_index][1]; j++) {
					for (int64 k = channel_2_indices[i - low_index][0]; k <= channel_2_indices[i - low_index][1]; k++) {

						int64 id_x = shot_data->sorted_photon_bins[channel_1][j] - shot_data->sorted_clock_bins[1][i] - *min_bin_1;
						int64 id_y = shot_data->sorted_photon_bins[channel_2][k] - shot_data->sorted_clock_bins[1][i] - *min_bin_2;
						int64 tot_id = id_y * (*max_bin_1 - *min_bin_1+1) + id_x;
						coinc[tot_id + ((*max_bin_1 - *min_bin_1 + 1) * (*max_bin_2 - *min_bin_2 + 1)) * shot_file_num]++;

					}
				}
			}
		}
	}
}

void calculateNumer_g2_for_channel_pair(shotData *shot_data, int64 *max_bin, int64 *pulse_spacing, int64 *max_pulse_distance, int32 *coinc, int32 shot_file_num, int32 num_cpu_threads_proc, int32 channel_1, int32 channel_2) {
	
	//Make sure we've actually got some tags in channel 1
	if((shot_data->sorted_photon_tag_pointers[channel_1] > 0) && (shot_data->sorted_photon_tag_pointers[channel_2] > 0)){

		//Get the start and stop clock bin
		int64 start_clock = shot_data->sorted_clock_bins[1][0];
		int64 end_clock = shot_data->sorted_clock_bins[0][shot_data->sorted_clock_tag_pointers[0] - 1];

		//Let's find out which indicies from channel 1 we can discard due to them being outside of the window of interest
		int64 low_index;
		int64 high_index;
		//Figure out which indices in the first thread we can ignore
		#pragma omp parallel for
		for (int32 i = 0; i < 2; i++) {
			if (i == 0) {
				low_index = first_above_binary_search(&(shot_data->sorted_photon_bins[channel_1]), shot_data->sorted_photon_tag_pointers[channel_1], *max_bin + *max_pulse_distance * *pulse_spacing + start_clock);
			}
			else {
				high_index = first_below_binary_search(&(shot_data->sorted_photon_bins[channel_1]), shot_data->sorted_photon_tag_pointers[channel_1], end_clock - (*max_bin + *max_pulse_distance * *pulse_spacing));
			}
		}

		//Split the remaining indices between the work threads we have
		int64 indices_per_thread = (high_index - low_index) / num_cpu_threads_proc;

		//Vector to hold the relevant indices on channel 2, for each tag on channel 1, that falls with the max/min tau
		std::vector<std::vector<int64>> channel_2_indices(high_index - low_index + 1, std::vector<int64>(2));
		#pragma omp parallel for num_threads(num_cpu_threads_proc)
		for (int32 thread = 0; thread < num_cpu_threads_proc; thread++) {
			
			//Find out form this thread what the first and last indices to work on are
			int64 first_index = thread*indices_per_thread + low_index;
			int64 last_index;
			if (thread == num_cpu_threads_proc - 1) {
				last_index = high_index;
			}
			else {
				last_index = first_index + indices_per_thread-1;
			}

			//Do a binary search to find the first and last relevant tag on channel 2 for the first tag on channel 1 that the thread is working on
			int64 lower_pointer = first_above_binary_search(&(shot_data->sorted_photon_bins[channel_2]), shot_data->sorted_photon_tag_pointers[channel_2], shot_data->sorted_photon_bins[channel_1][first_index] - *max_bin);;
			int64 upper_pointer = first_below_binary_search(&(shot_data->sorted_photon_bins[channel_2]), shot_data->sorted_photon_tag_pointers[channel_2], shot_data->sorted_photon_bins[channel_1][first_index] + *max_bin);

			//Save the relevant tags on channel 2
			channel_2_indices[first_index-low_index][0] = lower_pointer;
			channel_2_indices[first_index-low_index][1] = upper_pointer;

			//Loop over all the channel 1 indices this thread needs to work on
			for (int64 i = first_index + 1; i <= last_index; i++) {

				//Find the first tag on channel 2 that is within the max/min tau of the current channel 1 tag
				bool going = true;
				int64 j = lower_pointer;
				while (going) {
					if (shot_data->sorted_photon_bins[channel_2][j] < shot_data->sorted_photon_bins[channel_1][i] - *max_bin) {
						j++;
						lower_pointer = j;
					}
					else if (shot_data->sorted_photon_bins[channel_2][j] >= shot_data->sorted_photon_bins[channel_1][i] - *max_bin) {
						going = false;
						lower_pointer = j;
					}
					if (j > shot_data->sorted_photon_tag_pointers[channel_2]) {
						going = false;
						lower_pointer = j;
					}
				}
				//Find the last tag on channel 2 that is within the max/min tau of the current channel 1 tag
				j = upper_pointer;
				//Ensure j can't be negative
				if(j < 0){
					j = 0;
				}
				going = true;
				while (going) {
					if (shot_data->sorted_photon_bins[channel_2][j] <= shot_data->sorted_photon_bins[channel_1][i] + *max_bin) {
						j++;
						upper_pointer = j;
					}
					else if (shot_data->sorted_photon_bins[channel_2][j] > shot_data->sorted_photon_bins[channel_1][i] + *max_bin) {
						going = false;
						upper_pointer = j - 1;
					}
					if (j > shot_data->sorted_photon_tag_pointers[channel_2]) {
						going = false;
						upper_pointer = shot_data->sorted_photon_tag_pointers[channel_2] - 1;
					}
				}
				//Save the relevant tags to the vector for later
				channel_2_indices[i - low_index][0] = lower_pointer;
				channel_2_indices[i - low_index][1] = upper_pointer;
			}
			//Loop through all the tags on the thread has worked on to find the tau bin which the tags on channel 2 fall into
			for (int64 i = first_index; i <= last_index; i++) {
				for (int64 j = channel_2_indices[i - low_index][0]; j <= channel_2_indices[i - low_index][1]; j++) {
					int64 id_x = shot_data->sorted_photon_bins[channel_2][j] - shot_data->sorted_photon_bins[channel_1][i] + *max_bin;
					coinc[id_x + thread * ((*max_bin * 2 + 1)) + shot_file_num * num_cpu_threads_proc * ((*max_bin * 2 + 1))]++;
				}
			}
		}
	}
}

void calculateDenom_g2_for_channel_pair(shotData *shot_data, int64 *max_bin, int64 *pulse_spacing, int64 *max_pulse_distance, int32 *denom, int32 shot_file_num, int32 channel_1, int32 channel_2) {

	if((shot_data->sorted_photon_tag_pointers[channel_1] > 0) && (shot_data->sorted_photon_tag_pointers[channel_2] > 0)){

		int64 start_clock = shot_data->sorted_clock_bins[1][0];
		int64 end_clock = shot_data->sorted_clock_bins[0][shot_data->sorted_clock_tag_pointers[0] - 1];

		std::vector<int32> denom_counts(*max_pulse_distance * 2 + 1, 0);
		#pragma omp parallel for
		for (int64 pulse_dist = -*max_pulse_distance; pulse_dist <= *max_pulse_distance; pulse_dist++) {
			if (pulse_dist != 0) {
				int64 tau = *pulse_spacing * pulse_dist;
				int64 i = 0;
				int64 j = 0;
				while ((i < shot_data->sorted_photon_tag_pointers[channel_1]) && (j < shot_data->sorted_photon_tag_pointers[channel_2])) {
					//Check if we're outside the window of interest
					int32 out_window = (shot_data->sorted_photon_bins[channel_1][i] < (*max_bin + *max_pulse_distance * *pulse_spacing + start_clock)) || (shot_data->sorted_photon_bins[channel_1][i] > (end_clock - (*max_bin + *max_pulse_distance * *pulse_spacing)));
					//chan_1 > chan_2
					int32 c1gc2 = shot_data->sorted_photon_bins[channel_1][i] >(shot_data->sorted_photon_bins[channel_2][j] - tau);
					//Check if we have a common element increment
					int32 c1ec2 = shot_data->sorted_photon_bins[channel_1][i] == (shot_data->sorted_photon_bins[channel_2][j] - tau);
					//Increment running total if channel 1 equals channel 2
					denom_counts[pulse_dist + *max_pulse_distance] += !out_window && c1ec2;
					//Increment channel 1 if it is greater than channel 2, equal to channel 2 or ouside of the window
					i += (!c1gc2 || out_window);
					j += (c1gc2 || c1ec2);
				}
			}
		}
		for (int64 pulse_dist = -*max_pulse_distance; pulse_dist <= *max_pulse_distance; pulse_dist++) {
			denom[0] += denom_counts[pulse_dist + *max_pulse_distance];
		}
	}
}

void getChannelList(char* filename, std::vector<int16>* channel_list, std::map<int16, int16>* channel_map){
	//Open up file
	H5::H5File file(filename, H5F_ACC_RDONLY);
	//Open up "Inform" group
	H5::Group inform_group(file.openGroup("Inform"));
	//Grab channel list
	H5::DataSet chan_dset(inform_group.openDataSet("ChannelList"));
	H5::DataSpace chan_dspace = chan_dset.getSpace();
	hsize_t chan_length[1];
	chan_dspace.getSimpleExtentDims(chan_length, NULL);
	channel_list->resize(chan_length[0]);
	chan_dset.read(&(*channel_list)[0u], H5::PredType::NATIVE_UINT16, chan_dspace);
	chan_dspace.close();
	chan_dset.close();
	//Close Inform group
	inform_group.close();
	//Close file
	file.close();
	//Populate channel map
	for (int16 i = 0; i < channel_list->size(); i++) {
		(*channel_map)[(*channel_list)[i]] = i;
	}
}

//Function grabs all tags and channel list from file
void fileToShotData(shotData *shot_data, char* filename) {
	//Open up file
	H5::H5File file(filename, H5F_ACC_RDONLY);
	//Open up "Tags" group
	H5::Group tag_group(file.openGroup("Tags"));
	//Find out how many tag sets there are, should be 4 if not something is fucky
	hsize_t numTagsSets = tag_group.getNumObjs();
	if (numTagsSets != 4) {
		printf("There should be 4 sets of Tags, found %i\n", numTagsSets);
		delete filename;
		exit;
	}
	//Read tags to shotData structure
	//First the clock tags
	H5::DataSet clock_dset(tag_group.openDataSet("ClockTags0"));
	H5::DataSpace clock_dspace = clock_dset.getSpace();
	hsize_t clock_length[1];
	clock_dspace.getSimpleExtentDims(clock_length, NULL);
	shot_data->clock_tags.resize(clock_length[0]);
	clock_dset.read(&(*shot_data).clock_tags[0u], H5::PredType::NATIVE_UINT64, clock_dspace);
	clock_dspace.close();
	clock_dset.close();
	//Then start tags
	H5::DataSet start_dset(tag_group.openDataSet("StartTag"));
	H5::DataSpace start_dspace = start_dset.getSpace();
	hsize_t start_length[1];
	start_dspace.getSimpleExtentDims(start_length, NULL);
	shot_data->start_tags.resize(start_length[0]);
	start_dset.read(&(*shot_data).start_tags[0u], H5::PredType::NATIVE_UINT64, start_dspace);
	start_dspace.close();
	start_dset.close();
	//Then end tags
	H5::DataSet end_dset(tag_group.openDataSet("EndTag"));
	H5::DataSpace end_dspace = end_dset.getSpace();
	hsize_t end_length[1];
	end_dspace.getSimpleExtentDims(end_length, NULL);
	shot_data->end_tags.resize(end_length[0]);
	end_dset.read(&(*shot_data).end_tags[0u], H5::PredType::NATIVE_UINT64, end_dspace);
	end_dspace.close();
	end_dset.close();
	//Finally photon tags
	H5::DataSet photon_dset(tag_group.openDataSet("TagWindow0"));
	H5::DataSpace photon_dspace = photon_dset.getSpace();
	hsize_t photon_length[1];
	photon_dspace.getSimpleExtentDims(photon_length, NULL);
	shot_data->photon_tags.resize(photon_length[0]);
	photon_dset.read(&(*shot_data).photon_tags[0u], H5::PredType::NATIVE_UINT64, photon_dspace);
	photon_dspace.close();
	photon_dset.close();
	//And close tags group
	tag_group.close();
	//Open up "Inform" group
	H5::Group inform_group(file.openGroup("Inform"));
	//Grab channel list
	H5::DataSet chan_dset(inform_group.openDataSet("ChannelList"));
	H5::DataSpace chan_dspace = chan_dset.getSpace();
	hsize_t chan_length[1];
	chan_dspace.getSimpleExtentDims(chan_length, NULL);
	shot_data->channel_list.resize(chan_length[0]);
	chan_dset.read(&(*shot_data).channel_list[0u], H5::PredType::NATIVE_UINT16, chan_dspace);
	chan_dspace.close();
	chan_dset.close();
	//Close Inform group
	inform_group.close();
	//Close file
	file.close();

	//Populate channel map
	for (int16 i = 0; i < shot_data->channel_list.size(); i++) {
		shot_data->channel_map[shot_data->channel_list[i]] = i;
	}
}

//Reads relevant information for a block of files into shot_block
void populateBlock(std::vector<shotData> *shot_block, std::vector<char *> *filelist, int32 block_num, int32 num_devices, int32 block_size) {
	//Loop over the block size
	for (int32 i = 0; i < block_size * num_devices; i++) {
		//Default to assuming the block is corrupted
		(*shot_block)[i].file_load_completed = false;
		//Figure out the file id within the filelist
		int32 file_id = block_num * block_size * num_devices + i;
		//Check the file_id isn't out of range of the filelist
		if (file_id < filelist->size()) {
			//Try to load file to shot_block
			try {
				fileToShotData(&(*shot_block)[i], (*filelist)[file_id]);
				(*shot_block)[i].file_load_completed = true;
			}
			//Will catch if the file is corrupted, print corrupted filenames to command window
			catch (...) {
				printf("%s appears corrupted\n", (*filelist)[file_id]);
			}
		}
	}
}

//Process the time tags, assigning them to the correct channel, binning them appropriately and removing tags which do not fall in the clock mask
void sortTags(shotData *shot_data) {
	int64 i;
	int64 high_count = 0;
	//Loop over all tags in clock_tags
	for (i = 0; i < shot_data->clock_tags.size(); i++) {
		//Check if clock tag is a high word
		if (shot_data->clock_tags[i] & 1) {
			//Up the high count
			high_count++;
		}
		else {
			//Determine whether it is the rising (start) or falling (end) slope
			int8 slope = ((shot_data->clock_tags[i] >> 28) & 1);
			//Put tag in appropriate clock tag vector and increment the pointer for said vector
			shot_data->sorted_clock_tags[slope][shot_data->sorted_clock_tag_pointers[slope]] = ((shot_data->clock_tags[i] >> 1) & 0x7FFFFFF) + (high_count << 27) - ((shot_data->start_tags[1] >> 1) & 0x7FFFFFF);
			shot_data->sorted_clock_tag_pointers[slope]++;
		}
	}
	high_count = 0;
	//Clock pointer
	int64 clock_pointer = 0;
	//Loop over all tags in photon_tags
	for (i = 0; i < shot_data->photon_tags.size(); i++) {
		//Check if photon tag is a high word
		if (shot_data->photon_tags[i] & 1) {
			//Up the high count
			high_count++;
		}
		else {
			//Figure out if it fits within the mask
			int64 time_tag = ((shot_data->photon_tags[i] >> 1) & 0x7FFFFFF) + (high_count << 27) - ((shot_data->start_tags[1] >> 1) & 0x7FFFFFF);
			bool valid = true;
			while (valid) {
				//printf("%i\t%i\t%i\t", time_tag, shot_data->sorted_clock_tags[1][clock_pointer], shot_data->sorted_clock_tags[0][clock_pointer - 1]);
				//Increment dummy pointer if channel tag is greater than current start tag
				if ((time_tag >= shot_data->sorted_clock_tags[1][clock_pointer]) & (clock_pointer < shot_data->sorted_clock_tag_pointers[1])) {
					//printf("up clock pointer\n");
					clock_pointer++;
				}
				//Make sure clock_pointer is greater than 0, preventing an underflow error
				else if (clock_pointer > 0) {
					//Check if tag is lower than previous end tag i.e. startTags[j-1] < channeltags[i] < endTags[j-1]
					if (time_tag <= shot_data->sorted_clock_tags[0][clock_pointer - 1]) {
						//printf("add tag tot data\n");
						//Determine the index for given tag
						int32 channel_index;
						//Bin tag and assign to appropriate vector
						channel_index = shot_data->channel_map.find(((shot_data->photon_tags[i] >> 29) & 7) + 1)->second;
						shot_data->sorted_photon_tags[channel_index][shot_data->sorted_photon_tag_pointers[channel_index]] = time_tag;
						shot_data->sorted_photon_tag_pointers[channel_index]++;
						//printf("%i\t%i\t%i\n", channel_index, time_tag, shot_data->sorted_photon_tag_pointers[channel_index]);
					}
					//Break the valid loop
					valid = false;
				}
				// If tag is smaller than the first start tag
				else {
					valid = false;
				}
			}
		}
	}
}

//Converts our tags to bins with a given bin width
void tagsToBins(shotData *shot_data, double bin_width, std::vector<int64> *offset) {
	int64 tagger_bins_per_bin_width = (int64)round(bin_width / tagger_resolution);
	#pragma omp parallel for
	for (int32 channel = 0; channel < shot_data->sorted_photon_bins.size(); channel++) {
	#pragma omp parallel for
		for (int32 i = 0; i < shot_data->sorted_photon_tag_pointers[channel]; i++) {
			shot_data->sorted_photon_bins[channel][i] = (shot_data->sorted_photon_tags[channel][i] + (*offset)[channel]) / tagger_bins_per_bin_width;
		}
	}
	for (int32 slope = 0; slope <= 1; slope++) {
	#pragma omp parallel for
		for (int32 i = 0; i < shot_data->sorted_clock_tag_pointers[slope]; i++) {
			shot_data->sorted_clock_bins[slope][i] = shot_data->sorted_clock_tags[slope][i] / tagger_bins_per_bin_width;
		}
	}
}

//Sorts photons and bins them for each file in a block
void sortAndBinBlock(std::vector<shotData> *shot_block, double bin_width, int32 num_devices, int32 block_size, std::vector<int64> *offset) {
	#pragma omp parallel for
	for (int32 shot_file_num = 0; shot_file_num < (block_size * num_devices); shot_file_num++) {
		if ((*shot_block)[shot_file_num].file_load_completed) {
			sortTags(&(*shot_block)[shot_file_num]);
			tagsToBins(&(*shot_block)[shot_file_num], bin_width, offset);
		}
	}
}

void countTags(shotData *shot_data, std::vector<int32> *tot_counts, std::vector<int32> *masked_counts, std::vector<int32> *mask_block_counts, std::vector<int16> *channel_vec) {
	int64 high_count = 0;
	//Clock pointer
	int64 clock_pointer = 0;
	//Loop over all tags in photon_tags
	for (int64 i = 0; i < shot_data->photon_tags.size(); i++) {
		//Check if photon tag is a high word
		if (shot_data->photon_tags[i] & 1) {
			//Up the high count
			high_count++;
		}
		else {
			//Figure out if it fits within the mask
			int64 time_tag = ((shot_data->photon_tags[i] >> 1) & 0x7FFFFFF) + (high_count << 27) - ((shot_data->start_tags[1] >> 1) & 0x7FFFFFF);
			//Figure out the channel index, we'll use -1 as a placeholder for if the channel index doesn't appear in the list we've given the program
			int channel_index = -1;
			//Loop over the channel list to see if the channel pops up in the list
			for(int channel = 0; channel < channel_vec->size(); channel++){
				if((*channel_vec)[channel] == (((shot_data->photon_tags[i] >> 29) & 7) + 1)){
					channel_index = channel;
					//Break so we're not needlessly looping
					break;
				}
			}
			//If the channel is in the channel list
			if(channel_index >= 0){
				//Increment the total counts related to the channel
				(*tot_counts)[channel_index]++;
				//Check if tag is in or outside of mask block, i.e. after the first start clock and before the last end clock
				if((time_tag >= shot_data->sorted_clock_tags[1][0]) && (time_tag <= shot_data->sorted_clock_tags[0][shot_data->sorted_clock_tag_pointers[0]-1])){
					(*mask_block_counts)[channel_index]++;
				}
				bool valid = true;
				while (valid) {
					//Increment dummy pointer if channel tag is greater than current start tag
					if ((time_tag >= shot_data->sorted_clock_tags[1][clock_pointer]) & (clock_pointer < shot_data->sorted_clock_tag_pointers[1])) {
						//printf("up clock pointer\n");
						clock_pointer++;
					}
					//Make sure clock_pointer is greater than 0, preventing an underflow error
					else if (clock_pointer > 0) {
						//Check if tag is lower than previous end tag i.e. startTags[j-1] < channeltags[i] < endTags[j-1]
						if (time_tag <= shot_data->sorted_clock_tags[0][clock_pointer - 1]) {
							//Increment the masked counts for the channel if it lies within the mask
							(*masked_counts)[channel_index]++;

						}
						//Break the valid loop
						valid = false;
					}
					// If tag is smaller than the first start tag
					else {
						valid = false;
					}
				}
			}
		}
	}
}

extern "C" void EXPORT getG3Correlations_tripletwise(char **file_list, int32 file_list_length, double max_time, double bin_width, double pulse_spacing, int64 max_pulse_distance, PyObject *numer, PyObject *denom, bool calc_norm, int32 num_cpu_threads_files, int32 num_cpu_threads_proc, PyObject *corr_channel_list, bool disp_counts, PyObject *offset_list) {

	std::vector<char *> filelist(file_list_length);
	//Grab filename and stick it into filelist vector
	for (int32 i = 0; i < file_list_length; i++) {
		filelist[i] = file_list[i];
	}
	//Figure out max bin and the pulse spacing in multiples of the bin width
	int64 max_bin = (int64)round(max_time / bin_width);
	int64 bin_pulse_spacing = (int64)round(pulse_spacing / bin_width);

	//Get length of list of channel pairs
	int corr_channel_list_length = PyObject_Length(corr_channel_list);

	//Make a coincidence vector the appropriate size
	std::vector<int32 *> coinc(corr_channel_list_length);
	for(int channel_list_id  = 0; channel_list_id  < corr_channel_list_length; channel_list_id ++){
		coinc[channel_list_id] = (int32*)malloc((((2 * (max_bin)+1) * (2 * (max_bin)+1))) * num_cpu_threads_files * num_cpu_threads_proc * sizeof(int32));

		for (int32 id = 0; id < (((2 * (max_bin)+1) * (2 * (max_bin)+1))) * num_cpu_threads_files * num_cpu_threads_proc; id++) {
			coinc[channel_list_id][id] = 0;
		}
	}
	//Figure out how many blocks we need to chunk the files into
	int32 blocks_req;
	if (file_list_length < (num_cpu_threads_files)) {
		blocks_req = 1;
	}
	else if ((file_list_length % (num_cpu_threads_files)) == 0) {
		blocks_req = file_list_length / (num_cpu_threads_files);
	}
	else {
		blocks_req = file_list_length / (num_cpu_threads_files)+1;
	}

	//Get the channel list for the first file
	std::vector<int16> channel_vec;
	std::map<int16, int16> channel_map;
	if(file_list_length > 0){
		getChannelList(filelist[0], &channel_vec, &channel_map);
	}

	//Convert python channel lists into channel lists the correlations calculations understand
	std::vector<int16> corr_channel_1_vec(corr_channel_list_length,-1);
	std::vector<int16> corr_channel_2_vec(corr_channel_list_length,-1);
	std::vector<int16> corr_channel_3_vec(corr_channel_list_length,-1);
	for(int i = 0; i < corr_channel_list_length; i++){
		PyObject* channel_trip = PyList_GetItem(corr_channel_list,i);
		//Check if the channels exist in our map
		if((channel_map.find(PyLong_AsLong(PyList_GetItem(channel_trip, 0))) != channel_map.end()) && (channel_map.find(PyLong_AsLong(PyList_GetItem(channel_trip, 1))) != channel_map.end()) && (channel_map.find(PyLong_AsLong(PyList_GetItem(channel_trip, 2))) != channel_map.end())){
			//If both channels exist add them to be calculated
			corr_channel_1_vec[i] = (channel_map.find(PyLong_AsLong(PyList_GetItem(channel_trip, 0)))->second);
			corr_channel_2_vec[i] = (channel_map.find(PyLong_AsLong(PyList_GetItem(channel_trip, 1)))->second);
			corr_channel_3_vec[i] = (channel_map.find(PyLong_AsLong(PyList_GetItem(channel_trip, 2)))->second);
		}
		else{
			printf("Couldn't find %i, %i & %i channel trip\n", PyLong_AsLong(PyList_GetItem(channel_trip, 0)), PyLong_AsLong(PyList_GetItem(channel_trip, 1)), PyLong_AsLong(PyList_GetItem(channel_trip, 2)));
		}
	}

	//Convert offset pyobject to an offset vector
	std::vector<int64> offset_vec(channel_vec.size(),0);
	for(int i = 0; i < PyObject_Length(offset_list); i++){
		PyObject *offset_pair = PyList_GetItem(offset_list,i);
		//Lookup channel and put it into offset_vec
		if(channel_map.find(PyLong_AsLong(PyList_GetItem(offset_pair, 0))) != channel_map.end()){
			offset_vec[channel_map.find(PyLong_AsLong(PyList_GetItem(offset_pair, 0)))->second] = PyLong_AsLong(PyList_GetItem(offset_pair, 1));
		}
	}

	//Vector containing the masked counts and total counts for each channel
	std::vector<std::vector<int32>> masked_counts(num_cpu_threads_files, std::vector<int32>(channel_vec.size(),0));
	std::vector<std::vector<int32>> tot_counts(num_cpu_threads_files, std::vector<int32>(channel_vec.size(),0));
	std::vector<std::vector<int32>> masked_block_counts(num_cpu_threads_files, std::vector<int32>(channel_vec.size(),0));
	//The masked and total time
	double tot_time = 0;
	double mask_block_time = 0;

	std::vector<std::vector<int32>> denom_counts(corr_channel_list_length,std::vector<int32>(num_cpu_threads_files,0));

	printf("Chunking %i files into %i blocks\n", file_list_length, blocks_req);
	printf("Max Time\tBin Width\tPulse Spacing\tMax Pulse Distance\n");
	printf("%fus\t%fns\t%fus\t%i\n", max_time * 1e6, bin_width * 1e9, pulse_spacing * 1e6, max_pulse_distance);
	//Processes files in blocks
	for (int32 block_num = 0; block_num < blocks_req; block_num++) {
		//Allocate a vector to hold a block of shot_data
		std::vector<shotData> shot_block(num_cpu_threads_files);

		//Populate the shot_block with data from file
		populateBlock(&shot_block, &filelist, block_num, 1, num_cpu_threads_files);

		//Sort tags and convert them to bins
		sortAndBinBlock(&shot_block, bin_width, 1, num_cpu_threads_files, &offset_vec);

		if(disp_counts){
			//Get the start and stop clock bin
			for (int32 shot_file_num = 0; shot_file_num < num_cpu_threads_files; shot_file_num++) {
				if ((shot_block)[shot_file_num].file_load_completed) {
					int64 start_tag = ((shot_block[shot_file_num].start_tags[0] >> 1) << 27) + ((shot_block[shot_file_num].start_tags[1] >> 1) & 0x7FFFFFF);
					int64 end_tag = ((shot_block[shot_file_num].end_tags[0] >> 1) << 27) + ((shot_block[shot_file_num].end_tags[1] >> 1) & 0x7FFFFFF);
					tot_time += (double)(end_tag-start_tag) * tagger_resolution;
					mask_block_time += (double)(shot_block[shot_file_num].sorted_clock_tags[0][shot_block[shot_file_num].sorted_clock_tag_pointers[0] - 1] - shot_block[shot_file_num].sorted_clock_tags[1][0]) * tagger_resolution;
				}
			}
		}

		//Processes files
		#pragma omp parallel for num_threads(num_cpu_threads_files)
		for (int32 shot_file_num = 0; shot_file_num < num_cpu_threads_files; shot_file_num++) {
			if ((shot_block)[shot_file_num].file_load_completed) {
				for(int32 channel_list_id = 0; channel_list_id < corr_channel_list_length; channel_list_id++){
					if((corr_channel_1_vec[channel_list_id] != -1) && (corr_channel_2_vec[channel_list_id] != -1) && (corr_channel_3_vec[channel_list_id] != -1)){
						calculateNumer_g3_for_channel_trip(&(shot_block[shot_file_num]), &max_bin, &bin_pulse_spacing, &max_pulse_distance, &coinc[channel_list_id][0], shot_file_num,num_cpu_threads_proc, corr_channel_1_vec[channel_list_id], corr_channel_2_vec[channel_list_id], corr_channel_3_vec[channel_list_id]);
						if (calc_norm) {
							calculateDenom_g3_for_channel_trip(&(shot_block[shot_file_num]), &max_bin, &bin_pulse_spacing, &max_pulse_distance, &(denom_counts[channel_list_id][shot_file_num]), shot_file_num, corr_channel_1_vec[channel_list_id], corr_channel_2_vec[channel_list_id], corr_channel_3_vec[channel_list_id]);
						}
					}
				}
				if(disp_counts){
					countTags(&(shot_block[shot_file_num]), &(tot_counts[shot_file_num]), &(masked_counts[shot_file_num]), &(masked_block_counts[shot_file_num]), &channel_vec);
				}
			}
		}
		printf("Finished block %i of %i\n", block_num + 1, blocks_req);

	}

	//Collapse streamed coincidence counts down to regular numerator and denominator
	for(int32 channel_list_id = 0; channel_list_id < corr_channel_list_length; channel_list_id++){
		for (int32 i = 0; i < num_cpu_threads_files; i++) {
			for (int32 thread = 0; thread < num_cpu_threads_proc; thread++) {
				for (int32 j = 0; j < (((2 * (max_bin)+1) * (2 * (max_bin)+1))); j++) {
					PyList_SetItem(numer, j + channel_list_id * (((2 * (max_bin)+1) * (2 * (max_bin)+1))), PyLong_FromLong(PyLong_AsLong(PyList_GetItem(numer, j)) + coinc[channel_list_id][j + thread * ((2 * (max_bin)+1) * (2 * (max_bin)+1)) + i * num_cpu_threads_proc * ((2 * (max_bin)+1) * (2 * (max_bin)+1))]));
				}
			}
			PyList_SetItem(denom, channel_list_id, PyLong_FromLong(PyLong_AsLong(PyList_GetItem(denom, channel_list_id)) + denom_counts[channel_list_id][i]));
		}
	}
	for(int32 channel_list_id = 0; channel_list_id < corr_channel_list_length; channel_list_id++){
		free(coinc[channel_list_id]);
	}

	if(disp_counts){
		//Collapse the counts down
		std::vector<int32> tot_counts_collapse(channel_vec.size(),0);
		std::vector<int32> masked_counts_collapse(channel_vec.size(),0);
		std::vector<int32> masked_block_counts_collapse(channel_vec.size(),0);
		for(int channel = 0; channel < channel_vec.size(); channel++){
			for (int32 shot_file_num = 0; shot_file_num < (num_cpu_threads_files); shot_file_num++){
				tot_counts_collapse[channel] += tot_counts[shot_file_num][channel];
				masked_counts_collapse[channel] += masked_counts[shot_file_num][channel];
				masked_block_counts_collapse[channel] += masked_block_counts[shot_file_num][channel];
			}
		}

		printf("Tot time\tMasked block time\n");
		printf("%f\t%f\n",tot_time, mask_block_time);
		printf("Count info:\n");
		for(int i = 0; i < channel_vec.size(); i++){
			printf("Channel %i:\n", channel_vec[i]);
			printf("Singles Counts:\n");
			printf("Tot\t\tMasked\t\tUnmasked block\tUnmasked non-block\n");
			printf("%i\t\t%i\t\t%i\t\t%i\n", tot_counts_collapse[i], masked_counts_collapse[i], masked_block_counts_collapse[i] - masked_counts_collapse[i], tot_counts_collapse[i] - masked_block_counts_collapse[i]);
			printf("Rates (s^-1):\n");
			printf("Tot\t\tMasked\t\tUnmasked block\tUnmasked non-block\n");
			printf("%.1f\t\t%.1f\t\t%.1f\t\t%.1f\n", ((double)tot_counts_collapse[i]) / tot_time, ((double)masked_counts_collapse[i]) / mask_block_time, (double)(masked_block_counts_collapse[i] - masked_counts_collapse[i]) / (mask_block_time), (double)(tot_counts_collapse[i] - masked_block_counts_collapse[i]) / (tot_time - mask_block_time));
		}
	}
	printf("\n\n");
}

extern "C" void EXPORT getG3Correlations_tripletwise_with_tau(char **file_list, int32 file_list_length, double min_time_2, double max_time_2, double min_time_3, double max_time_3, double bin_width, double pulse_spacing, int64 max_pulse_distance, PyObject *numer, PyObject *denom, bool calc_norm, int32 num_cpu_threads_files, int32 num_cpu_threads_proc, PyObject *corr_channel_list, bool disp_counts, PyObject *offset_list) {

	std::vector<char *> filelist(file_list_length);
	//Grab filename and stick it into filelist vector
	for (int32 i = 0; i < file_list_length; i++) {
		filelist[i] = file_list[i];
	}
	//Figure out max bin and the pulse spacing in multiples of the bin width
	int64 max_bin_2 = (int64)round(max_time_2 / bin_width);
	int64 min_bin_2 = (int64)round(min_time_2 / bin_width);
	int64 max_bin_3 = (int64)round(max_time_3 / bin_width);
	int64 min_bin_3 = (int64)round(min_time_3 / bin_width);
	int64 bin_pulse_spacing = (int64)round(pulse_spacing / bin_width);

	//Overall max_bin
	int64 max_bin = 0;
	if (max_bin_2 > max_bin_3){
		max_bin = max_bin_2;
	}
	else{
		max_bin = max_bin_3;
	}

	//Get length of list of channel pairs
	int corr_channel_list_length = PyObject_Length(corr_channel_list);

	//Make a coincidence vector the appropriate size
	std::vector<int32 *> coinc(corr_channel_list_length);
	for(int channel_list_id  = 0; channel_list_id  < corr_channel_list_length; channel_list_id ++){
		coinc[channel_list_id] = (int32*)malloc(((max_bin_2 - min_bin_2 + 1) * (max_bin_3 - min_bin_3 + 1)) * num_cpu_threads_files * num_cpu_threads_proc * sizeof(int32));

		for (int32 id = 0; id < ((max_bin_2 - min_bin_2 + 1) * (max_bin_3 - min_bin_3 + 1)) * num_cpu_threads_files * num_cpu_threads_proc; id++) {
			coinc[channel_list_id][id] = 0;
		}
	}
	//Figure out how many blocks we need to chunk the files into
	int32 blocks_req;
	if (file_list_length < (num_cpu_threads_files)) {
		blocks_req = 1;
	}
	else if ((file_list_length % (num_cpu_threads_files)) == 0) {
		blocks_req = file_list_length / (num_cpu_threads_files);
	}
	else {
		blocks_req = file_list_length / (num_cpu_threads_files)+1;
	}

	//Get the channel list for the first file
	std::vector<int16> channel_vec;
	std::map<int16, int16> channel_map;
	if(file_list_length > 0){
		getChannelList(filelist[0], &channel_vec, &channel_map);
	}

	//Convert python channel lists into channel lists the correlations calculations understand
	std::vector<int16> corr_channel_1_vec(corr_channel_list_length,-1);
	std::vector<int16> corr_channel_2_vec(corr_channel_list_length,-1);
	std::vector<int16> corr_channel_3_vec(corr_channel_list_length,-1);
	for(int i = 0; i < corr_channel_list_length; i++){
		PyObject* channel_trip = PyList_GetItem(corr_channel_list,i);
		//Check if the channels exist in our map
		if((channel_map.find(PyLong_AsLong(PyList_GetItem(channel_trip, 0))) != channel_map.end()) && (channel_map.find(PyLong_AsLong(PyList_GetItem(channel_trip, 1))) != channel_map.end()) && (channel_map.find(PyLong_AsLong(PyList_GetItem(channel_trip, 2))) != channel_map.end())){
			//If both channels exist add them to be calculated
			corr_channel_1_vec[i] = (channel_map.find(PyLong_AsLong(PyList_GetItem(channel_trip, 0)))->second);
			corr_channel_2_vec[i] = (channel_map.find(PyLong_AsLong(PyList_GetItem(channel_trip, 1)))->second);
			corr_channel_3_vec[i] = (channel_map.find(PyLong_AsLong(PyList_GetItem(channel_trip, 2)))->second);
		}
		else{
			printf("Couldn't find %i, %i & %i channel trip\n", PyLong_AsLong(PyList_GetItem(channel_trip, 0)), PyLong_AsLong(PyList_GetItem(channel_trip, 1)), PyLong_AsLong(PyList_GetItem(channel_trip, 2)));
		}
	}

	//Convert offset pyobject to an offset vector
	std::vector<int64> offset_vec(channel_vec.size(),0);
	for(int i = 0; i < PyObject_Length(offset_list); i++){
		PyObject *offset_pair = PyList_GetItem(offset_list,i);
		//Lookup channel and put it into offset_vec
		if(channel_map.find(PyLong_AsLong(PyList_GetItem(offset_pair, 0))) != channel_map.end()){
			offset_vec[channel_map.find(PyLong_AsLong(PyList_GetItem(offset_pair, 0)))->second] = PyLong_AsLong(PyList_GetItem(offset_pair, 1));
		}
	}

	//Vector containing the masked counts and total counts for each channel
	std::vector<std::vector<int32>> masked_counts(num_cpu_threads_files, std::vector<int32>(channel_vec.size(),0));
	std::vector<std::vector<int32>> tot_counts(num_cpu_threads_files, std::vector<int32>(channel_vec.size(),0));
	std::vector<std::vector<int32>> masked_block_counts(num_cpu_threads_files, std::vector<int32>(channel_vec.size(),0));
	//The masked and total time
	double tot_time = 0;
	double mask_block_time = 0;

	std::vector<std::vector<int32>> denom_counts(corr_channel_list_length,std::vector<int32>(num_cpu_threads_files,0));

	printf("Chunking %i files into %i blocks\n", file_list_length, blocks_req);
	printf("Bin Width\tPulse Spacing\tMax Pulse Distance\n");
	printf("%fns\t%fus\t%i\n", bin_width * 1e9, pulse_spacing * 1e6, max_pulse_distance);
	printf("Min time 2\tMax time 2\tMin time 3\tMax time 3\n");
	printf("%fus\t%fus\t%fus\t%fus\n", min_time_2*1e6, max_time_2*1e6, min_time_3*1e6, max_time_3*1e6);
	//Processes files in blocks
	for (int32 block_num = 0; block_num < blocks_req; block_num++) {
		//Allocate a vector to hold a block of shot_data
		std::vector<shotData> shot_block(num_cpu_threads_files);

		//Populate the shot_block with data from file
		populateBlock(&shot_block, &filelist, block_num, 1, num_cpu_threads_files);

		//Sort tags and convert them to bins
		sortAndBinBlock(&shot_block, bin_width, 1, num_cpu_threads_files, &offset_vec);

		if(disp_counts){
			//Get the start and stop clock bin
			for (int32 shot_file_num = 0; shot_file_num < num_cpu_threads_files; shot_file_num++) {
				if ((shot_block)[shot_file_num].file_load_completed) {
					int64 start_tag = ((shot_block[shot_file_num].start_tags[0] >> 1) << 27) + ((shot_block[shot_file_num].start_tags[1] >> 1) & 0x7FFFFFF);
					int64 end_tag = ((shot_block[shot_file_num].end_tags[0] >> 1) << 27) + ((shot_block[shot_file_num].end_tags[1] >> 1) & 0x7FFFFFF);
					tot_time += (double)(end_tag-start_tag) * tagger_resolution;
					mask_block_time += (double)(shot_block[shot_file_num].sorted_clock_tags[0][shot_block[shot_file_num].sorted_clock_tag_pointers[0] - 1] - shot_block[shot_file_num].sorted_clock_tags[1][0]) * tagger_resolution;
				}
			}
		}

		//Processes files
		#pragma omp parallel for num_threads(num_cpu_threads_files)
		for (int32 shot_file_num = 0; shot_file_num < num_cpu_threads_files; shot_file_num++) {
			if ((shot_block)[shot_file_num].file_load_completed) {
				for(int32 channel_list_id = 0; channel_list_id < corr_channel_list_length; channel_list_id++){
					if((corr_channel_1_vec[channel_list_id] != -1) && (corr_channel_2_vec[channel_list_id] != -1) && (corr_channel_3_vec[channel_list_id] != -1)){
						calculateNumer_g3_for_channel_trip_with_taus(&(shot_block[shot_file_num]), &min_bin_2, &max_bin_2, &min_bin_3, &max_bin_3, &bin_pulse_spacing, &max_pulse_distance, &coinc[channel_list_id][0], shot_file_num,num_cpu_threads_proc, corr_channel_1_vec[channel_list_id], corr_channel_2_vec[channel_list_id], corr_channel_3_vec[channel_list_id]);
						if (calc_norm) {
							calculateDenom_g3_for_channel_trip(&(shot_block[shot_file_num]), &max_bin, &bin_pulse_spacing, &max_pulse_distance, &(denom_counts[channel_list_id][shot_file_num]), shot_file_num, corr_channel_1_vec[channel_list_id], corr_channel_2_vec[channel_list_id], corr_channel_3_vec[channel_list_id]);
						}
					}
				}
				if(disp_counts){
					countTags(&(shot_block[shot_file_num]), &(tot_counts[shot_file_num]), &(masked_counts[shot_file_num]), &(masked_block_counts[shot_file_num]), &channel_vec);
				}
			}
		}
		printf("Finished block %i of %i\n", block_num + 1, blocks_req);

	}

	//Collapse streamed coincidence counts down to regular numerator and denominator
	for(int32 channel_list_id = 0; channel_list_id < corr_channel_list_length; channel_list_id++){
		for (int32 i = 0; i < num_cpu_threads_files; i++) {
			for (int32 thread = 0; thread < num_cpu_threads_proc; thread++) {
				for (int32 j = 0; j < ((max_bin_2 - min_bin_2 + 1) * (max_bin_3 - min_bin_3 + 1)); j++) {
					PyList_SetItem(numer, j + channel_list_id * ((max_bin_2 - min_bin_2 + 1) * (max_bin_3 - min_bin_3 + 1)), PyLong_FromLong(PyLong_AsLong(PyList_GetItem(numer, j)) + coinc[channel_list_id][j + thread * ((max_bin_2 - min_bin_2 + 1) * (max_bin_3 - min_bin_3 + 1)) + i * num_cpu_threads_proc * ((max_bin_2 - min_bin_2 + 1) * (max_bin_3 - min_bin_3 + 1))]));
				}
			}
			PyList_SetItem(denom, channel_list_id, PyLong_FromLong(PyLong_AsLong(PyList_GetItem(denom, channel_list_id)) + denom_counts[channel_list_id][i]));
		}
	}
	for(int32 channel_list_id = 0; channel_list_id < corr_channel_list_length; channel_list_id++){
		free(coinc[channel_list_id]);
	}

	if(disp_counts){
		//Collapse the counts down
		std::vector<int32> tot_counts_collapse(channel_vec.size(),0);
		std::vector<int32> masked_counts_collapse(channel_vec.size(),0);
		std::vector<int32> masked_block_counts_collapse(channel_vec.size(),0);
		for(int channel = 0; channel < channel_vec.size(); channel++){
			for (int32 shot_file_num = 0; shot_file_num < (num_cpu_threads_files); shot_file_num++){
				tot_counts_collapse[channel] += tot_counts[shot_file_num][channel];
				masked_counts_collapse[channel] += masked_counts[shot_file_num][channel];
				masked_block_counts_collapse[channel] += masked_block_counts[shot_file_num][channel];
			}
		}

		printf("Tot time\tMasked block time\n");
		printf("%f\t%f\n",tot_time, mask_block_time);
		printf("Count info:\n");
		for(int i = 0; i < channel_vec.size(); i++){
			printf("Channel %i:\n", channel_vec[i]);
			printf("Singles Counts:\n");
			printf("Tot\t\tMasked\t\tUnmasked block\tUnmasked non-block\n");
			printf("%i\t\t%i\t\t%i\t\t%i\n", tot_counts_collapse[i], masked_counts_collapse[i], masked_block_counts_collapse[i] - masked_counts_collapse[i], tot_counts_collapse[i] - masked_block_counts_collapse[i]);
			printf("Rates (s^-1):\n");
			printf("Tot\t\tMasked\t\tUnmasked block\tUnmasked non-block\n");
			printf("%.1f\t\t%.1f\t\t%.1f\t\t%.1f\n", ((double)tot_counts_collapse[i]) / tot_time, ((double)masked_counts_collapse[i]) / mask_block_time, (double)(masked_block_counts_collapse[i] - masked_counts_collapse[i]) / (mask_block_time), (double)(tot_counts_collapse[i] - masked_block_counts_collapse[i]) / (tot_time - mask_block_time));
		}
	}
	printf("\n\n");
}

extern "C" void EXPORT getG2Correlations_pulse(char **file_list, int32 file_list_length, double min_tau_1, double max_tau_1, double min_tau_2, double max_tau_2, double bin_width, PyObject *numer, int32 *denom, int32 num_cpu_threads_files) {

	std::vector<char *> filelist(file_list_length);
	//Grab filename and stick it into filelist vector
	for (int32 i = 0; i < file_list_length; i++) {
		filelist[i] = file_list[i];
	}

	int64 min_bin_tau_1 = (int64)round(min_tau_1 / bin_width);
	int64 max_bin_tau_1 = (int64)round(max_tau_1 / bin_width);
	int64 min_bin_tau_2 = (int64)round(min_tau_2 / bin_width);
	int64 max_bin_tau_2 = (int64)round(max_tau_2 / bin_width);

	int32 blocks_req;
	if (file_list_length < (num_cpu_threads_files)) {
		blocks_req = 1;
	}
	else if ((file_list_length % (num_cpu_threads_files)) == 0) {
		blocks_req = file_list_length / (num_cpu_threads_files);
	}
	else {
		blocks_req = file_list_length / (num_cpu_threads_files)+1;
	}

	printf("Chunking %i files into %i blocks\n", file_list_length, blocks_req);
	printf("Tau_1\tTau_2\tBin Width\n");
	printf("%f to %fus\t%f to %fus\t%fns\n", min_tau_1 * 1e6, max_tau_1 * 1e6, min_tau_2 * 1e6, max_tau_2 * 1e6, bin_width * 1e9);

	int32 *coinc;
	coinc = (int32*)malloc((max_bin_tau_1-min_bin_tau_1 + 1) * (max_bin_tau_2 - min_bin_tau_2 + 1) * num_cpu_threads_files * sizeof(int32));
	for (int32 id = 0; id < ((max_bin_tau_1 - min_bin_tau_1 + 1) * (max_bin_tau_2 - min_bin_tau_2 + 1) * num_cpu_threads_files); id++) {
		coinc[id] = 0;
	}

	//Get the channel list for the first file
	std::vector<int16> channel_vec;
	std::map<int16, int16> channel_map;
	if(file_list_length > 0){
		getChannelList(filelist[0], &channel_vec, &channel_map);
	}
	std::vector<int64> offset_vec(channel_vec.size(),0);

	std::vector<int32> denom_counts(num_cpu_threads_files, 0);

	//Processes files in blocks
	for (int32 block_num = 0; block_num < blocks_req; block_num++) {
		//Allocate a vector to hold a block of shot_data
		std::vector<shotData> shot_block(num_cpu_threads_files);

		//Populate the shot_block with data from file
		populateBlock(&shot_block, &filelist, block_num, 1, num_cpu_threads_files);

		//Sort tags and convert them to bins
		sortAndBinBlock(&shot_block, bin_width, 1, num_cpu_threads_files, &offset_vec);

		//Processes files
		#pragma omp parallel for num_threads(num_cpu_threads_files)
		for (int32 shot_file_num = 0; shot_file_num < num_cpu_threads_files; shot_file_num++) {
			if ((shot_block)[shot_file_num].file_load_completed) {
				calculateNumer_g2_pulse(&(shot_block[shot_file_num]), &min_bin_tau_1, &max_bin_tau_1, &min_bin_tau_2, &max_bin_tau_2, coinc, shot_file_num);
			}
		}
		printf("Finished block %i of %i\n", block_num + 1, blocks_req);

	}

	//Collapse streamed coincidence counts down to regular numerator and denominator
	for (int32 i = 0; i < num_cpu_threads_files; i++) {
		for (int32 j = 0; j < (max_bin_tau_1 - min_bin_tau_1 + 1) * (max_bin_tau_2 - min_bin_tau_2 + 1); j++) {
			PyList_SetItem(numer, j, PyLong_FromLong(PyLong_AsLong(PyList_GetItem(numer, j)) + coinc[j + i * (max_bin_tau_1 - min_bin_tau_1 + 1) * (max_bin_tau_2 - min_bin_tau_2 + 1)]));
		}
	}
	free(coinc);
	printf("\n\n");
}

extern "C" void EXPORT getG2Correlations_pairwise(char **file_list, int32 file_list_length, double max_time, double bin_width, double pulse_spacing, int64 max_pulse_distance, PyObject *numer, PyObject *denom, bool calc_norm, int32 num_cpu_threads_files, int32 num_cpu_threads_proc, PyObject *corr_channel_list, bool disp_counts, PyObject *offset_list) {
	std::vector<char *> filelist(file_list_length);
	//Grab filename and stick it into filelist vector
	for (int32 i = 0; i < file_list_length; i++) {
		filelist[i] = file_list[i];
	}

	//Figure out max bin and the pulse spacing in multiples of the bin width
	int64 max_bin = (int64)round(max_time / bin_width);
	int64 bin_pulse_spacing = (int64)round(pulse_spacing / bin_width);

	//Get length of list of channel pairs
	int corr_channel_list_length = PyObject_Length(corr_channel_list);

	//Make a coincidence vector the appropriate size
	std::vector<int32 *> coinc(corr_channel_list_length);
	for(int channel_list_id  = 0; channel_list_id  < corr_channel_list_length; channel_list_id ++){
		coinc[channel_list_id] = (int32*)malloc(((2 * (max_bin)+1)) * num_cpu_threads_files * num_cpu_threads_proc * sizeof(int32));
		for (int32 id = 0; id < ((2 * (max_bin)+1)) * num_cpu_threads_files * num_cpu_threads_proc; id++) {
			coinc[channel_list_id][id] = 0;
		}
	}
	//Figure out how many blocks we need to chunk the files into
	int32 blocks_req;
	if (file_list_length < (num_cpu_threads_files)) {
		blocks_req = 1;
	}
	else if ((file_list_length % (num_cpu_threads_files)) == 0) {
		blocks_req = file_list_length / (num_cpu_threads_files);
	}
	else {
		blocks_req = file_list_length / (num_cpu_threads_files)+1;
	}

	//Get the channel list for the first file
	std::vector<int16> channel_vec;
	std::map<int16, int16> channel_map;
	if(file_list_length > 0){
		getChannelList(filelist[0], &channel_vec, &channel_map);
	}

	//Convert python channel lists into channel lists the correlations calculations understand
	std::vector<int16> corr_channel_1_vec(corr_channel_list_length,-1);
	std::vector<int16> corr_channel_2_vec(corr_channel_list_length,-1);
	for(int i = 0; i < corr_channel_list_length; i++){
		PyObject* channel_pair = PyList_GetItem(corr_channel_list,i);
		//Check if the channels exist in our map
		if((channel_map.find(PyLong_AsLong(PyList_GetItem(channel_pair, 0))) != channel_map.end()) && (channel_map.find(PyLong_AsLong(PyList_GetItem(channel_pair, 1))) != channel_map.end())){
			//If both channels exist add them to be calculated
			corr_channel_1_vec[i] = (channel_map.find(PyLong_AsLong(PyList_GetItem(channel_pair, 0)))->second);
			corr_channel_2_vec[i] = (channel_map.find(PyLong_AsLong(PyList_GetItem(channel_pair, 1)))->second);
		}
		else{
			printf("Couldn't find %i and %i channel pair\n", PyLong_AsLong(PyList_GetItem(channel_pair, 0)), PyLong_AsLong(PyList_GetItem(channel_pair, 1)));
		}
	}
	
	//Convert offset pyobject to an offset vector
	std::vector<int64> offset_vec(channel_vec.size(),0);
	for(int i = 0; i < PyObject_Length(offset_list); i++){
		PyObject *offset_pair = PyList_GetItem(offset_list,i);
		//Lookup channel and put it into offset_vec
		if(channel_map.find(PyLong_AsLong(PyList_GetItem(offset_pair, 0))) != channel_map.end()){
			offset_vec[channel_map.find(PyLong_AsLong(PyList_GetItem(offset_pair, 0)))->second] = PyLong_AsLong(PyList_GetItem(offset_pair, 1));
		}
	}

	//Vector containing the masked counts and total counts for each channel
	std::vector<std::vector<int32>> masked_counts(num_cpu_threads_files, std::vector<int32>(channel_vec.size(),0));
	std::vector<std::vector<int32>> tot_counts(num_cpu_threads_files, std::vector<int32>(channel_vec.size(),0));
	std::vector<std::vector<int32>> masked_block_counts(num_cpu_threads_files, std::vector<int32>(channel_vec.size(),0));
	//The masked and total time
	double tot_time = 0;
	double mask_block_time = 0;

	printf("Chunking %i files into %i blocks\n", file_list_length, blocks_req);
	printf("Max Time\tBin Width\tPulse Spacing\tMax Pulse Distance\n");
	printf("%fus\t%fns\t%fus\t%i\n", max_time * 1e6, bin_width * 1e9, pulse_spacing * 1e6, max_pulse_distance);

	std::vector<std::vector<int32>> denom_counts(corr_channel_list_length,std::vector<int32>(num_cpu_threads_files,0));

	//Processes files in blocks
	for (int32 block_num = 0; block_num < blocks_req; block_num++) {
		//Allocate a vector to hold a block of shot_data
		std::vector<shotData> shot_block(num_cpu_threads_files);

		//Populate the shot_block with data from file
		populateBlock(&shot_block, &filelist, block_num, 1, num_cpu_threads_files);

		//Sort tags and convert them to bins
		sortAndBinBlock(&shot_block, bin_width, 1, num_cpu_threads_files, &offset_vec);

		if(disp_counts){
			//Get the start and stop clock bin
			for (int32 shot_file_num = 0; shot_file_num < num_cpu_threads_files; shot_file_num++) {
				if ((shot_block)[shot_file_num].file_load_completed) {
					int64 start_tag = ((shot_block[shot_file_num].start_tags[0] >> 1) << 27) + ((shot_block[shot_file_num].start_tags[1] >> 1) & 0x7FFFFFF);
					int64 end_tag = ((shot_block[shot_file_num].end_tags[0] >> 1) << 27) + ((shot_block[shot_file_num].end_tags[1] >> 1) & 0x7FFFFFF);
					tot_time += (double)(end_tag-start_tag) * tagger_resolution;
					mask_block_time += (double)(shot_block[shot_file_num].sorted_clock_tags[0][shot_block[shot_file_num].sorted_clock_tag_pointers[0] - 1] - shot_block[shot_file_num].sorted_clock_tags[1][0]) * tagger_resolution;
				}
			}
		}

		//Processes files
		#pragma omp parallel for num_threads(num_cpu_threads_files)
		for (int32 shot_file_num = 0; shot_file_num < num_cpu_threads_files; shot_file_num++) {
			if ((shot_block)[shot_file_num].file_load_completed) {
				for(int32 channel_list_id = 0; channel_list_id < corr_channel_list_length; channel_list_id++){
					if((corr_channel_1_vec[channel_list_id] != -1) && (corr_channel_2_vec[channel_list_id] != -1)){
						calculateNumer_g2_for_channel_pair(&(shot_block[shot_file_num]), &max_bin, &bin_pulse_spacing, &max_pulse_distance, &coinc[channel_list_id][0], shot_file_num, num_cpu_threads_proc, corr_channel_1_vec[channel_list_id], corr_channel_2_vec[channel_list_id]);
						if (calc_norm) {
							calculateDenom_g2_for_channel_pair(&(shot_block[shot_file_num]), &max_bin, &bin_pulse_spacing, &max_pulse_distance, &(denom_counts[channel_list_id][shot_file_num]), shot_file_num, corr_channel_1_vec[channel_list_id], corr_channel_2_vec[channel_list_id]);
						}
					}
				}
				if(disp_counts){
					countTags(&(shot_block[shot_file_num]), &(tot_counts[shot_file_num]), &(masked_counts[shot_file_num]), &(masked_block_counts[shot_file_num]), &channel_vec);
				}
			}
		}
		printf("Finished block %i of %i\n", block_num + 1, blocks_req);

	}

	//Collapse streamed coincidence counts down to regular numerator and denominator
	for(int32 channel_list_id = 0; channel_list_id < corr_channel_list_length; channel_list_id++){
		for (int32 i = 0; i < num_cpu_threads_files; i++) {
			for (int32 thread = 0; thread < num_cpu_threads_proc; thread++) {
				for (int32 j = 0; j < (2 * (max_bin)+1); j++) {
					if (j < (2 * (max_bin)+1)) {
						PyList_SetItem(numer, j + channel_list_id * (2 * (max_bin)+1), PyLong_FromLong(PyLong_AsLong(PyList_GetItem(numer, j + channel_list_id * (2 * (max_bin)+1))) + coinc[channel_list_id][j + thread * ((2 * (max_bin)+1)) + i * num_cpu_threads_proc * ((2 * (max_bin)+1))]));
					}
				}
			}
			PyList_SetItem(denom, channel_list_id, PyLong_FromLong(PyLong_AsLong(PyList_GetItem(denom, channel_list_id)) + denom_counts[channel_list_id][i]));
		}
	}
	for(int32 channel_list_id = 0; channel_list_id < corr_channel_list_length; channel_list_id++){
		free(coinc[channel_list_id]);
	}
	if(disp_counts){
		//Collapse the counts down
		std::vector<int32> tot_counts_collapse(channel_vec.size(),0);
		std::vector<int32> masked_counts_collapse(channel_vec.size(),0);
		std::vector<int32> masked_block_counts_collapse(channel_vec.size(),0);
		for(int channel = 0; channel < channel_vec.size(); channel++){
			for (int32 shot_file_num = 0; shot_file_num < (num_cpu_threads_files); shot_file_num++){
				tot_counts_collapse[channel] += tot_counts[shot_file_num][channel];
				masked_counts_collapse[channel] += masked_counts[shot_file_num][channel];
				masked_block_counts_collapse[channel] += masked_block_counts[shot_file_num][channel];
			}
		}

		printf("Tot time\tMasked block time\n");
		printf("%f\t%f\n",tot_time, mask_block_time);
		printf("Count info:\n");
		for(int i = 0; i < channel_vec.size(); i++){
			printf("Channel %i:\n", channel_vec[i]);
			printf("Singles Counts:\n");
			printf("Tot\t\tMasked\t\tUnmasked block\tUnmasked non-block\n");
			printf("%i\t\t%i\t\t%i\t\t%i\n", tot_counts_collapse[i], masked_counts_collapse[i], masked_block_counts_collapse[i] - masked_counts_collapse[i], tot_counts_collapse[i] - masked_block_counts_collapse[i]);
			printf("Rates (s^-1):\n");
			printf("Tot\t\tMasked\t\tUnmasked block\tUnmasked non-block\n");
			printf("%.1f\t\t%.1f\t\t%.1f\t\t%.1f\n", ((double)tot_counts_collapse[i]) / tot_time, ((double)masked_counts_collapse[i]) / mask_block_time, (double)(masked_block_counts_collapse[i] - masked_counts_collapse[i]) / (mask_block_time), (double)(tot_counts_collapse[i] - masked_block_counts_collapse[i]) / (tot_time - mask_block_time));
		}
	}
	printf("\n\n");

}

extern "C" void EXPORT getCounts(char **file_list, int32 file_list_length, int32 num_cpu_threads_files){
	std::vector<char *> filelist(file_list_length);
	//Grab filename and stick it into filelist vector
	for (int32 i = 0; i < file_list_length; i++) {
		filelist[i] = file_list[i];
	}
	//Figure out how many blocks are required
	int32 blocks_req;
	if (file_list_length < (num_cpu_threads_files)) {
		blocks_req = 1;
	}
	else if ((file_list_length % (num_cpu_threads_files)) == 0) {
		blocks_req = file_list_length / (num_cpu_threads_files);
	}
	else {
		blocks_req = file_list_length / (num_cpu_threads_files)+1;
	}

	//Get the channel list for the first file
	std::vector<int16> channel_vec;
	std::map<int16, int16> channel_map;
	if(file_list_length > 0){
		getChannelList(filelist[0], &channel_vec, &channel_map);
	}
	printf("Chunking %i files into %i blocks\n", file_list_length, blocks_req);

	std::vector<int64> offset_vec(channel_vec.size(),0);

	//Vector containing the masked counts and total counts for each channel
	std::vector<std::vector<int32>> masked_counts(num_cpu_threads_files, std::vector<int32>(channel_vec.size(),0));
	std::vector<std::vector<int32>> tot_counts(num_cpu_threads_files, std::vector<int32>(channel_vec.size(),0));
	std::vector<std::vector<int32>> masked_block_counts(num_cpu_threads_files, std::vector<int32>(channel_vec.size(),0));
	//The masked and total time
	double tot_time = 0;
	double mask_block_time = 0;
	//Processes files in blocks
	for (int32 block_num = 0; block_num < blocks_req; block_num++) {
		//Allocate a vector to hold a block of shot_data
		std::vector<shotData> shot_block(num_cpu_threads_files);

		//Populate the shot_block with data from file
		populateBlock(&shot_block, &filelist, block_num, 1, num_cpu_threads_files);
		//Sort tags and convert them to bins
		sortAndBinBlock(&shot_block, tagger_resolution * 2, 1, num_cpu_threads_files, &offset_vec);
		//For each file figure out the masked and total counts
		#pragma omp parallel for
		for (int32 shot_file_num = 0; shot_file_num < (num_cpu_threads_files); shot_file_num++) {
			if ((shot_block)[shot_file_num].file_load_completed) {
				countTags(&(shot_block[shot_file_num]), &(tot_counts[shot_file_num]), &(masked_counts[shot_file_num]), &(masked_block_counts[shot_file_num]), &channel_vec);
			}
		}
		//Get the start and stop clock bin
		for (int32 shot_file_num = 0; shot_file_num < num_cpu_threads_files; shot_file_num++) {
			if ((shot_block)[shot_file_num].file_load_completed) {
				int64 start_tag = ((shot_block[shot_file_num].start_tags[0] >> 1) << 27) + ((shot_block[shot_file_num].start_tags[1] >> 1) & 0x7FFFFFF);
				int64 end_tag = ((shot_block[shot_file_num].end_tags[0] >> 1) << 27) + ((shot_block[shot_file_num].end_tags[1] >> 1) & 0x7FFFFFF);
				tot_time += (double)(end_tag-start_tag) * tagger_resolution;
				mask_block_time += (double)(shot_block[shot_file_num].sorted_clock_tags[0][shot_block[shot_file_num].sorted_clock_tag_pointers[0] - 1] - shot_block[shot_file_num].sorted_clock_tags[1][0]) * tagger_resolution;
			}
		}
		printf("Finished block %i of %i\n", block_num + 1, blocks_req);
	}
	//Collapse the counts down
	std::vector<int32> tot_counts_collapse(channel_vec.size(),0);
	std::vector<int32> masked_counts_collapse(channel_vec.size(),0);
	std::vector<int32> masked_block_counts_collapse(channel_vec.size(),0);
	for(int channel = 0; channel < channel_vec.size(); channel++){
		for (int32 shot_file_num = 0; shot_file_num < (num_cpu_threads_files); shot_file_num++){
			tot_counts_collapse[channel] += tot_counts[shot_file_num][channel];
			masked_counts_collapse[channel] += masked_counts[shot_file_num][channel];
			masked_block_counts_collapse[channel] += masked_block_counts[shot_file_num][channel];
		}
	}

	printf("Tot time\tMasked block time\n");
	printf("%f\t%f\n",tot_time, mask_block_time);
	printf("Count info:\n");
	for(int i = 0; i < channel_vec.size(); i++){
		printf("Channel %i:\n", channel_vec[i]);
		printf("Singles Counts:\n");
		printf("Tot\t\tMasked\t\tUnmasked block\tUnmasked non-block\n");
		printf("%i\t\t%i\t\t%i\t\t%i\n", tot_counts_collapse[i], masked_counts_collapse[i], masked_block_counts_collapse[i] - masked_counts_collapse[i], tot_counts_collapse[i] - masked_block_counts_collapse[i]);
		printf("Rates (s^-1):\n");
		printf("Tot\t\tMasked\t\tUnmasked block\tUnmasked non-block\n");
		printf("%.1f\t\t%.1f\t\t%.1f\t\t%.1f\n", ((double)tot_counts_collapse[i]) / tot_time, ((double)masked_counts_collapse[i]) / mask_block_time, (double)(masked_block_counts_collapse[i] - masked_counts_collapse[i]) / (mask_block_time), (double)(tot_counts_collapse[i] - masked_block_counts_collapse[i]) / (tot_time - mask_block_time));
	}

	printf("\n\n");

}

extern "C" void EXPORT getCounts_and_return(char **file_list, int32 file_list_length, int32 num_cpu_threads_files, PyObject *channel_list, PyObject *masked_counts_back, PyObject *tot_counts_back, PyObject *masked_block_counts_back, double *tot_time_back, double *masked_block_time_back){
	std::vector<char *> filelist(file_list_length);
	//Grab filename and stick it into filelist vector
	for (int32 i = 0; i < file_list_length; i++) {
		filelist[i] = file_list[i];
	}
	//Figure out how many blocks are required
	int32 blocks_req;
	if (file_list_length < (num_cpu_threads_files)) {
		blocks_req = 1;
	}
	else if ((file_list_length % (num_cpu_threads_files)) == 0) {
		blocks_req = file_list_length / (num_cpu_threads_files);
	}
	else {
		blocks_req = file_list_length / (num_cpu_threads_files)+1;
	}


	//Get the channel list for the first file
	std::vector<int16> channel_vec;
	//std::map<int16, int16> channel_map;
	//#if(file_list_length > 0){
	//	getChannelList(filelist[0], &channel_vec, &channel_map);
	//}
	channel_vec.resize(PyObject_Length(channel_list));
	for(int i = 0; i < PyObject_Length(channel_list); i++){
		channel_vec[i] = PyLong_AsLong(PyList_GetItem(channel_list, i));
	}
	printf("Chunking %i files into %i blocks\n", file_list_length, blocks_req);

	std::vector<int64> offset_vec(channel_vec.size(),0);

	//Vector containing the masked counts and total counts for each channel
	std::vector<std::vector<int32>> masked_counts(num_cpu_threads_files, std::vector<int32>(channel_vec.size(),0));
	std::vector<std::vector<int32>> tot_counts(num_cpu_threads_files, std::vector<int32>(channel_vec.size(),0));
	std::vector<std::vector<int32>> masked_block_counts(num_cpu_threads_files, std::vector<int32>(channel_vec.size(),0));
	//The masked and total time
	double tot_time = 0;
	double mask_block_time = 0;
	//Processes files in blocks
	for (int32 block_num = 0; block_num < blocks_req; block_num++) {
		//Allocate a vector to hold a block of shot_data
		std::vector<shotData> shot_block(num_cpu_threads_files);

		//Populate the shot_block with data from file
		populateBlock(&shot_block, &filelist, block_num, 1, num_cpu_threads_files);
		//Sort tags and convert them to bins
		sortAndBinBlock(&shot_block, tagger_resolution * 2, 1, num_cpu_threads_files, &offset_vec);
		//For each file figure out the masked and total counts
		#pragma omp parallel for
		for (int32 shot_file_num = 0; shot_file_num < (num_cpu_threads_files); shot_file_num++) {
			if ((shot_block)[shot_file_num].file_load_completed) {
				countTags(&(shot_block[shot_file_num]), &(tot_counts[shot_file_num]), &(masked_counts[shot_file_num]), &(masked_block_counts[shot_file_num]), &channel_vec);
			}
		}
		//Get the start and stop clock bin
		for (int32 shot_file_num = 0; shot_file_num < num_cpu_threads_files; shot_file_num++) {
			if ((shot_block)[shot_file_num].file_load_completed) {
				int64 start_tag = ((shot_block[shot_file_num].start_tags[0] >> 1) << 27) + ((shot_block[shot_file_num].start_tags[1] >> 1) & 0x7FFFFFF);
				int64 end_tag = ((shot_block[shot_file_num].end_tags[0] >> 1) << 27) + ((shot_block[shot_file_num].end_tags[1] >> 1) & 0x7FFFFFF);
				tot_time += (double)(end_tag-start_tag) * tagger_resolution;
				mask_block_time += (double)(shot_block[shot_file_num].sorted_clock_tags[0][shot_block[shot_file_num].sorted_clock_tag_pointers[0] - 1] - shot_block[shot_file_num].sorted_clock_tags[1][0]) * tagger_resolution;
			}
		}
		printf("Finished block %i of %i\n", block_num + 1, blocks_req);
	}
	//Collapse the counts down
	std::vector<int32> tot_counts_collapse(channel_vec.size(),0);
	std::vector<int32> masked_counts_collapse(channel_vec.size(),0);
	std::vector<int32> masked_block_counts_collapse(channel_vec.size(),0);
	for(int channel = 0; channel < channel_vec.size(); channel++){
		for (int32 shot_file_num = 0; shot_file_num < (num_cpu_threads_files); shot_file_num++){
			tot_counts_collapse[channel] += tot_counts[shot_file_num][channel];
			masked_counts_collapse[channel] += masked_counts[shot_file_num][channel];
			masked_block_counts_collapse[channel] += masked_block_counts[shot_file_num][channel];
		}
	}

	/*printf("Tot time\tMasked block time\n");
	printf("%f\t%f\n",tot_time, mask_block_time);
	printf("Count info:\n");
	for(int i = 0; i < channel_vec.size(); i++){
		printf("Channel %i:\n", channel_vec[i]);
		printf("Singles Counts:\n");
		printf("Tot\t\tMasked\t\tUnmasked block\tUnmasked non-block\n");
		printf("%i\t\t%i\t\t%i\t\t%i\n", tot_counts_collapse[i], masked_counts_collapse[i], masked_block_counts_collapse[i] - masked_counts_collapse[i], tot_counts_collapse[i] - masked_block_counts_collapse[i]);
		printf("Rates (s^-1):\n");
		printf("Tot\t\tMasked\t\tUnmasked block\tUnmasked non-block\n");
		printf("%.1f\t\t%.1f\t\t%.1f\t\t%.1f\n", ((double)tot_counts_collapse[i]) / tot_time, ((double)masked_counts_collapse[i]) / mask_block_time, (double)(masked_block_counts_collapse[i] - masked_counts_collapse[i]) / (mask_block_time), (double)(tot_counts_collapse[i] - masked_block_counts_collapse[i]) / (tot_time - mask_block_time));
	}*/

	*tot_time_back = tot_time;
	*masked_block_time_back = mask_block_time;
	for(int i = 0; i < channel_vec.size(); i++){
		PyList_SetItem(tot_counts_back, i, PyLong_FromLong(tot_counts_collapse[i]));
		PyList_SetItem(masked_counts_back, i, PyLong_FromLong(masked_counts_collapse[i]));
		PyList_SetItem(masked_block_counts_back, i, PyLong_FromLong(masked_block_counts_collapse[i]));
	}

	printf("\n\n");

}