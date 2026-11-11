from pynwb import read_nwb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from PoissonNaiveBayesModel import PoissonNB

#------------------------------------------------------------------------------------
"""sets file path, configures session selection and feature selection, and prints settings"""

filepath = r''

#Assign to 'pre' for pre-injection decoding and 'post' for post-injection decoding
session_selection = ''

#Assign to 'neuron sum' for spike counts per neuron per trial, 'bin sum' for spike counts per bin per trial, or 'neuron bin combined' for spike counts per bin per neuron per trial
feature_selection = ''

nwb_file = read_nwb(filepath)

print(f"Session selection: {session_selection}-injection")
print(f"Feature selection: {feature_selection}")


#------------------------------------------------------------------------------------
"""extracts unit spikes"""
def get_unit_location(unit_row):
    mean_waveforms = unit_row['waveform_mean']
    waveform_mins = np.min(mean_waveforms, axis=0)
    peak_channel_idx = np.argmin(waveform_mins)
    detected_electrodes = unit_row['electrodes']
    unit_location = detected_electrodes.iloc[peak_channel_idx].location
    return unit_location

"""selects quality units in the specified brain regions"""

def select_units(units, brain_regions):
    units_spike_times = []
    location_units_spike_times = []
    for location in brain_regions:
        for _, row in units.query('(firing_range > 8) & (decoder_label== "sua") & (presence_ratio >.95)').iterrows():
            if get_unit_location(row) == location:
                location_units_spike_times.append(row.spike_times)
    units_spike_times += location_units_spike_times
    return units_spike_times

"""creates the spike data matrix that will be given to the decoder"""
def get_spike_matrix(stim_times, units_spike_times, bin_edges):
    time_resolution = np.mean(np.diff(bin_edges))
    # 3D spike matrix to be populated with spike counts
    spike_matrix = np.zeros((len(units_spike_times), len(stim_times), len(bin_edges)-1))

    # populate 3D spike matrix for each unit for each stimulus trial by counting spikes into bins
    for unit_idx in range(len(units_spike_times)):
        spike_times = units_spike_times[unit_idx]

        for stim_idx, stim_time in enumerate(stim_times):
            # get spike times that fall within the bin's time range relative to the stim time        
            first_bin_time = stim_time + bin_edges[0]
            last_bin_time = stim_time + bin_edges[-1]
            first_spike_in_range, last_spike_in_range = np.searchsorted(spike_times, [first_bin_time, last_bin_time])
            spike_times_in_range = spike_times[first_spike_in_range:last_spike_in_range]

            # convert spike times into relative time bin indices
            bin_indices = ((spike_times_in_range - (first_bin_time)) / time_resolution).astype(int)
            
            # mark that there is a spike at these bin times for this unit on this stim trial
            for bin_idx in bin_indices:
                spike_matrix[unit_idx, stim_idx, bin_idx] += 1

    return spike_matrix


#------------------------------------------------------------------------------------
"""creates spike data arrays"""

#selects brain region in the visual cortex of the mice
brain_regions = [
    'VISp1', 'VISp2/3', 'VISp4', 'VISp5', 'VISp6a', 'VISp6b',
    'VISam2/3', 'VISam4', 'VISam5', 'VISam6a', 'VISam6b',
]

spike_data = nwb_file.units.to_dataframe()

#extracts spike data times using the select_units function
spike_data_times = select_units(spike_data, brain_regions)

#prints number of neurons
print(f"Selected {len(spike_data_times)} units")


#------------------------------------------------------------------------------------
"""Converts presentation times and stimuli details from the nwb file to dataframe, and prints injection type"""

#Creates a new dataframe including image presentation data
behavior_presentations = nwb_file.intervals['behavior_presentations'].to_dataframe()

y = nwb_file.intervals['injection_times'].to_dataframe()
x = nwb_file.intervals['behavior_presentations'].to_dataframe()

#Assigns start_time to when the injection was administered
start_time = y.loc[0, 'start_time']

#Prints injection type
injection = y['injection_type'].iloc[0]
print(f"Injection type: {injection}")

#filters for image presentation before or after injection depending on user settings
if session_selection == 'pre':
    behavior_presentations_filtered = x[x['start_time'] < start_time]
elif session_selection == 'post':
    behavior_presentations_filtered = x[x['start_time'] >= start_time]
else:
    print("Invalid session selection setting. Assign to either 'pre' or 'post'")

#Creates a new stimulus dataframe with the start time, stop time, and image identity columns only
stimulus_df = behavior_presentations_filtered[['start_time', 'stop_time', 'image_name']].copy()


#------------------------------------------------------------------------------------
"""Sets bin size in ms, number of bins, and spike data observation window start and end times in ms"""

#bin size (5ms)
time_resolution = 0.005

#observation window starts 50 ms after stimulus is shown and ends 250 ms after presentation
window_start_time = 0.05
window_end_time = 0.250

#calculates number of time bins
n_bins = int(((window_end_time - window_start_time) / time_resolution)+1)
bin_edges = np.linspace(window_start_time, window_end_time, n_bins, endpoint=True)


#------------------------------------------------------------------------------------
"""Creates the spike data matrix that will be given to the decoder, manipulates matrices, and prints matrix shape"""

#creates the spike matrix using the get_spike_matrix function
spike_matrix = get_spike_matrix(stimulus_df['start_time'], spike_data_times, bin_edges)

#Manipulates matrices to represent neuron sums per trial, bin sums per trial, or bin sums per neuron per trial depending on user settings
if feature_selection == 'neuron sum':
    spike_matrix_final = np.sum(spike_matrix, axis=2).transpose(1,0)
elif feature_selection == 'bin sum':
    spike_matrix_final = np.sum(spike_matrix, axis=0)
elif feature_selection == 'neuron bin combined':
    n_neurons, n_trials, n_bins = spike_matrix.shape
    spike_matrix_final = spike_matrix.transpose(1, 0, 2).reshape(n_trials, n_neurons * n_bins)

print(f"Spike matrix shape: {spike_matrix_final.shape}")


#------------------------------------------------------------------------------------
"""Assigns the generated spike matrix and image identity column from stimulus_df to their respective variables, and performs a 80/20 stratified split of the data"""

X = spike_matrix_final
Y = stimulus_df['image_name']

#reserves 80% of the data for training, and 20% of the data for testing, and ensures that each of the 8 images occupy the same proportion of trials for each set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=41, stratify=Y)


#------------------------------------------------------------------------------------
"""Creates object (model) of class PoissonNB, trains the model, and prints results"""

model = PoissonNB()
model.fit(X_train, Y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
accuracy = accuracy_score(Y_test, predictions) * 100
highest_probabilities = np.max(probabilities, axis=1)
mean_confidence = np.mean(highest_probabilities) * 100
cm = confusion_matrix(Y_test, predictions)
ll = log_loss(Y_test, probabilities)
normalized_entropy = model.normalized_entropy(X_test)
mean_normalized_entropy = np.mean(normalized_entropy)

print(f"\nAccuracy: {accuracy}%")
print(f"Mean confidence: {mean_confidence}")
print(f"\n{cm}")
print(f"\nLog Loss: {ll} nats")
print(f"Mean normalized entropy: {mean_normalized_entropy}")
