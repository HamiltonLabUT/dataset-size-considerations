# Authors
Maansi Desai, Alyssa Field, Liberty S. Hamilton

# Abstract
In many experiments that investigate auditory and speech processing in the brain using electroencephalography (EEG), the experimental paradigm is often lengthy and tedious. Typically, the experimenter errs on the side of including more data, more trials, and therefore conducting a longer task to ensure that the data are robust and effects are measurable. Recent studies used naturalistic stimuli to investigate the brain’s response to individual or a combination of multiple speech features using system identification techniques, such as multivariate temporal receptive field (mTRF) analyses. The neural data collected from such experiments must be divided into a training set and a test set to fit and validate the mTRF weights. While a good strategy is clearly to collect as much data as is feasible, it is unclear how much data are needed to achieve stable results. Furthermore, it is unclear whether the specific stimulus used for mTRF fitting and the choice of feature representation affects how much data would be required for robust and generalizable results. Here, we used previously collected EEG data from our lab using sentence stimuli and movie stimuli as well as EEG data from an open-source dataset using audiobook stimuli to better understand how much data needs to be collected for naturalistic speech experiments measuring acoustic and phonetic tuning. We found that the EEG receptive field structure tested here stabilizes after collecting a training dataset of approximately 200 seconds of TIMIT sentences, around 600 seconds of movie trailers training set data, and approximately 460 seconds of audiobook training set data. Thus, we provide suggestions on the minimum amount of data that would be necessary for fitting mTRFs from naturalistic listening data. Our findings are motivated by highly practical concerns when working with children, patient populations, or others who may not tolerate long study sessions. These findings will aid future researchers who wish to study naturalistic speech processing in healthy and clinical populations while minimizing participant fatigue and retaining signal quality.


## Data Availability:
The .h5 files which contain the model correlation values and weights are too large to upload onto OSF. Please contact the authors for the data if needed:

**Maansi Desai**: maansi.desai@utexas.edu

OR

**Liberty Hamilton**: Liberty.Hamilton@austin.utexas.edu

# Getting Started:
The purpose of this repo is to be able to use the code to assess how much training set data would be needed for future natural speech stimuli experiment. 
In order for the code to work, please ensure that you have an .h5 file which has the contents stored in this format: 



	naturalspeech.hf5
	│
    └── stimulus_type_1 (e.g. MovieTrailers)
	│	└── wav_file_name_1
	│	│	└── resp
	│	│	|	└── subject_ID
	│	│	│		└── epochs
	│	│	└── stim
	│	│		└── pitch
	│	│		└── envelope
	│	│		└── phonological-features
	│	└── wav_file_name_2
	│		└── resp
	│			└── ...
	│		└── stim
	│			└── ...
	└── stimulus_type_2 (e.g. TIMIT)
    │    └── wav_file_name_1 
	...
	...
	└── 

### Code to run (it's only one notebook! Well...technically two notebooks. One notebook runs r-code)
1)  `main-runScripts.ipynb` : change path, subject, stimulus class, model type (feature representation)
2)  `kneepoint_lmem.ipynb` : change path, stimulus class. This script will run the linear mixed effects model (r-code)


There are multiple python scripts which get called from the `main-runScripts.ipynb` notebook. The first half of this notebook uses two different python scripts which are used to output the results from the mTRF (multivariate temporal receptive field). These python files perform the analysis and output an .h5 file for a specific feature representation (acoustic envelope, phonological features, pitch) or a combination of all three of these speech features. Because an mTRF is fit on increasing amounts of training data, the processing of conducting these analysis and outputting a set of weights and correlation value for each iteration of training data can take an extremely long time! The duration of fitting models for each stimulus set based on feature representation can range between 24 hours to 3 weeks. The reason being, fitting a single feature (such as the acoustic envelope or pitch) for the TIMIT stimulus set only consisting of a single feature as a model input. TIMIT also did not have as much data as the movie trailers. In contrast, fitting a TRF using the phonological features (which includes 14 features in the model) for the movie trailers will take considerabely longer (about 2-3 weeks) because of the number of features as well as the amount of training set data which gets concatenated with each iteration. The bottom line is, if you want to output the .h5 file for each stimulus set using these individual audtiroy features or a combination of all three features, be prepared that you will be running these analyses for several months. As an additional note, it is best to fit these models using a computer which has a lot of RAM (40 - 64GB is an acceptable range). This will not speed up the process of fitting these models, but instead having more RAM will help with being able to run multiple subjects at the same time on a single device. There's additional information in the `main-runScripts.ipynb` notebook. 

### *Breakdown of repo and contents*

| Filename | Description |
| --- | --- |
| `audio_tools` | folder with custom-written functions for auditory feature extractions |
| `ridge` | folder with ridge regression code used to fit linear regression model |
| `run_STRF_analysis.py` | The main python file which contains functions for fitting mTRFs and for also plotting receptive fields and the correlations generated from the encoding model analysis. |
| `textgrids.py` | Previously used for textgrids and gets called in the `run_STRF_analysis.py` from model fitting in a previously published work (Desai et al. (2021), JNeuro. |
| `training_plotting_funcs.py` | Analysis functions used for plotting model correlation values and weights. |
| `kneepoint_lmem.ipynb` | Notebook which runs R code for linear mixed effects models. Loads in .csv file and performs statistical analysis. |
| `main-runScripts.ipynb` | Jupyter notebook imports `run_STRF_analysis.py` to plot the results (correlation values and weights from the mTRFs) for each stimulus type and for each figure from the manuscript. Run the cells in the order stated in the notebook to visualize the data and generate figures. There are more detailed instructions are the top of this notebook along with some FAQs.
| `audiobooks_how_much_training_data.py` or `TIMIT_how_much_training_data.py` or `trailers_how_much_training_data.py`| Call these .py files to fit encoding models for each stimulus type for increasing sentences/chunks of training set data plus bootstrapping.
