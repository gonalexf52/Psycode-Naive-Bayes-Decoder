Psycode Naive Bayes Decoder

A Naive Bayes Decoder developed for OpenScope's Psycode Dataset—a project dedicated to investigating the effect of psilocybin administration on visual processing and behavior in the visual cortex of mice. This decoder leverages a Poisson distribution to model the probability of discrete spike counts over a time interval and attempts to decode natural-image identity from spike-train data.


Pipeline

-Loads neural spike-train data in Neurodata Without Borders (NWB) format.

-Creates spike data arrays using quality and region filters.

-Selects natural image presentations from the pre-injection or post-injection periods and manipulates spike matrices, both according to user settings.

-Counts spikes 50-250 ms after stimulus onset, with 5 ms time bins.

-Partitions the data using a stratified random split, reserving 20% for testing.

-Trains the model and prints all testing results, including: accuracy, mean confidence, log loss, mean normalized entropy, and a confusion matrix.

The decoder applies the Bayesian neural decoding framework in Wei et al. (2024), coauthored by Ian H. Stevenson.

$$
P(x^* \mid y^*, x, Y)
\propto
\left[
\prod_{i=1}^{N}
P(y_i^* \mid x^*, x, Y)
\right]
p(x^*)
$$


Set Up

Install required dependencies:
pip install numpy pynwb scikit-learn scipy

Download an individual NWB recording from the DANDI dataset and update these settings in naive-bayes-image-identity-decoder.py:
session_selection  = ‘pre’ or ‘post’
feature_selection = ‘neuron sum’ or ‘bin sum’ or ‘neuron bin combined’
filepath = r’path to individual recording file’

Run:
python decoder.py


Citations

Amaya, A., Bawany, A., Buice, M., Carlson, M., De Filippo, R., Durand, S., Gillis, R., Grasso, C., Groblewski, P., Han, W., Johnson, T., Kenney, J., Kiselycznyk, C., Koch, C., Lecoq, J. A., Loeffler, H., Marks, L., Naidoo, R., Ott, T., Ouellette, B., Peene, R. C., Rembado, I., Schmitz, D., Suarez, L., Swapp, J., Weber, J., Wilkes, J., Williford, A., & Wyrick, D. (2026). Allen Institute OpenScope - Psycode (Version 0.260626.2154) [Data set]. DANDI Archive. https://doi.org/10.48324/dandi.001417/0.260626.2154

Wei 魏赣超, G., Tajik Mansouri زینب تاجیک منصوری, Z., Wang 王晓婧, X., & Stevenson, I. H. (2024). Calibrating Bayesian Decoders of Neural Spiking Activity. The Journal of neuroscience : the official journal of the Society for Neuroscience, 44(18), e2158232024. https://doi.org/10.1523/JNEUROSCI.2158-23.2024