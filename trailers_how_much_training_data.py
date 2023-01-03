from run_STRF_analysis import *

#import h5py
#
user = input('Input your computer ID: ')
data_dir='/Users/%s/Box/MovieTrailersTask/Data/EEG/Participants' %(user)
# data_dir='/Users/%s/Desktop/data/EEG/MovieTrailers/Participants/' #load data
#data_dir ='/work/07267/mdesai/stampede2/data/trailers'
subject = input('Input subject ID: ')
stimulus_class = 'MovieTrailers'
rep_num = 10
resp_dict, stim_dict = loadEEGh5(subject, 'MovieTrailers', data_dir,
	eeg_epochs=True, resp_mean = False, binarymat=False, binaryfeatmat = True, envelope=True, pitch=True, gabor_pc10=True, 
	spectrogram=False, binned_pitches=False, spectrogram_scaled=False, scene_cut=True)
# resp_dict_MT0020, stim_dict = loadEEGh5('MT0020', 'MovieTrailers', data_dir, eeg_epochs=True, 
# 						 resp_mean = False, binaryfeatmat = True, binarymat=False, envelope=True,
# 						 pitch=True, spectrogram=False)

trailers_list = ['angrybirds-tlr1_a720p.wav', 'bighero6-tlr1_a720p.wav', 'bighero6-tlr2_a720p.wav', 
'bighero6-tlr3_a720p.wav', 'cars-3-trailer-4_a720p.wav', 'coco-trailer-1_a720p.wav', 
'ferdinand-trailer-2_a720p.wav', 'ferdinand-trailer-3_a720p.wav', 'ice-dragon-trailer-1_a720p.wav', 
'incredibles-2-trailer-1_a720p.wav', 'incredibles-2-trailer-2_a720p.wav', 'insideout-tlr2zzyy32_a720p.wav',
'insideout-usca-tlr2_a720p.wav', 'moana-clip-youre-welcome_a720p.wav', 'paddington-2-trailer-1_a720p.wav', 
'pandas-trailer-2_a720p.wav', 'pele-tlr1_a720p.wav', 'the-breadwinner-trailer-1_a720p.wav', 
'the-lego-ninjago-movie-trailer-1_a720p.wav', 
'thelittleprince-tlr_a720p.wav', 'trolls-tlr1_a720p.wav']

#For MT0001 below:
# trailers_list =['coco-trailer-1_a720p.wav','trolls-tlr1_a720p.wav','moana-clip-youre-welcome_a720p.wav','bighero6-tlr3_a720p.wav',	
# 'paddington-2-trailer-1_a720p.wav','incredibles-2-trailer-2_a720p.wav','cars-3-trailer-4_a720p.wav',	
# 'thelittleprince-tlr_a720p.wav','ferdinand-trailer-3_a720p.wav','deep-trailer-1_a720p.wav', 'the-breadwinner-trailer-1_a720p.wav', 
# 'the-lego-ninjago-movie-trailer-2_a720p.wav',	
# 'angrybirds-tlr1_a720p.wav',	
# 'incredibles-2-trailer-1_a720p.wav',	
# 'insideout-usca-tlr2_a720p.wav']

#MT0013 list below:
# trailers_list = ['angrybirds-tlr1_a720p.wav', 'bighero6-tlr1_a720p.wav', 'bighero6-tlr2_a720p.wav', 
# 'bighero6-tlr3_a720p.wav', 'coco-trailer-1_a720p.wav', 'deep-trailer-1_a720p.wav', 
# 'incredibles-2-trailer-1_a720p.wav', 'incredibles-2-trailer-2_a720p.wav', 'paddington-2-trailer-1_a720p.wav', 
# 'insideout-tlr2zzyy32_a720p.wav', 'ferdinand-trailer-3_a720p.wav', 'pele-tlr1_a720p.wav', 'pandas-trailer-2_a720p.wav',
#  'the-breadwinner-trailer-1_a720p.wav', 'moana-clip-youre-welcome_a720p.wav', 'trolls-tlr1_a720p.wav']
# resp_dict = {}

# for k in trailers_list:
# 	resp_dict[k] = [np.concatenate((resp_dict_MT0002[k][0], resp_dict_MT0020[k][0]), axis=0)]
test_set = ['paddington-2-trailer-1_a720p.wav', 'insideout-tlr2zzyy32_a720p.wav'] #the test set for the remaining MTs

all_stimuli = trailers_list
training_set = np.setdiff1d(all_stimuli, test_set)
print(training_set)

val_inds = np.zeros((len(all_stimuli),), dtype=np.bool) 
train_inds = np.zeros((len(all_stimuli),), dtype=np.bool)
for i in np.arange(len(all_stimuli)):
	if all_stimuli[i] in test_set:
		print(all_stimuli[i])
		val_inds[i] = True
	else:
		train_inds[i] = True

print("Total number of training sentences:")
print(sum(train_inds))
print("Total number of validation sentences:")
print(sum(val_inds))

train_inds = np.where(train_inds==True)[0]
val_inds = np.where(val_inds==True)[0]

print("Training indices:")
print(train_inds)
print("Validation indices:")
print(val_inds)

logging.basicConfig(level=logging.DEBUG) 

#time delays used in STRF
delay_min = 0.0
delay_max = 0.6
wt_pad = 0.1 # Amount of padding for delays, since edge artifacts can make weights look weird

fs = 128.0
delays = np.arange(np.floor((delay_min-wt_pad)*fs), np.ceil((delay_max+wt_pad)*fs), dtype=np.int) #create array to pass time delays in

print("Delays:", delays)

# Regularization parameters (alphas - also sometimes called lambda)
#load alphas from full audio STRF MT file (constant) as opposed to assess alpha through each iteration -- will expedite process
f = h5py.File('%s/%s/%s_STRF_by_pitchenvsphnfeatgabor10pc_sc_MT.hf5' %(data_dir, subject, subject), 'r')
alphas = [f['valphas_mt'][0]]
# alphas = np.hstack((0, np.logspace(2,8,20))) # Gives you 15 values between 10^2 and 10^8

nalphas = len(alphas)
use_corr = True # Use correlation between predicted and validation set as metric for goodness of fit
single_alpha = True # Use the same alpha value for all electrodes (helps with comparing across sensors)
nboots = 20 # How many bootstraps to do. (This is number of times you take a subset of the training data to find the best ridge parameter)

all_wts = [] # STRF weights (this will be ndelays x channels)
all_corrs = [] # correlation performance of length [nchans]
all_corrs_shuff = [] # List of lists, how good is a random model

# train_inds and val_inds are defined in the cell above, and is based on getting 80% of the trials
# for each unique stimulus to be in the training set, and the remaining 20% to be in 
# the validation set
current_stim_list_train = np.array([all_stimuli[r][0] for r in train_inds])
current_stim_list_val = np.array([all_stimuli[r][0] for r in val_inds])

# Create training and validation response matrices
print(resp_dict[training_set[0]][0].shape)
print(test_set)

print(len(training_set))
for r in training_set:
	print(r)


vResp_numtrials = [resp_dict[r][0].shape[0] for r in training_set if resp_dict[r][0].shape[0] ]

print(vResp_numtrials[0])
print(np.shape(vResp_numtrials))

#find number of training set stimuli 
ntrials = np.shape(vResp_numtrials)[0]
print(ntrials)
vreps = np.arange(1,ntrials+1)
print(vreps)
nboots = 10

corrs_reps = dict()
wts_reps = dict()


vResp_temp = [resp_dict[r][0].mean(0) for r in test_set]
vResp = np.hstack((vResp_temp)).T
print(vResp.shape)


vStim_temp = np.atleast_2d(np.vstack([np.vstack(stim_dict[r]).T for r in test_set]))

vStim_temp = vStim_temp/vStim_temp.max(0)

print(vStim_temp.max(0).shape)
print('**********************************')

vStim = make_delayed(vStim_temp, delays)


training_inds = train_inds #already have this above 
ntraining_sentences = len(training_inds)

tResp_orig = np.hstack(([resp_dict[r][0].mean(0) for r in training_set])).T #words correspond to numbers for each stimulus type

tStim_temp = np.atleast_2d(np.vstack([np.vstack(stim_dict[r]).T for r in training_set]))

tStim_orig = make_delayed(tStim_temp, delays)

nsamples = 10
chunking_mt = 2*128 #time for MT for duration of chunking 
training_mt_chunks = np.int(tResp_orig.shape[0]/chunking_mt)
for v in np.arange(rep_num, training_mt_chunks+1):
	corrs_reps[v] = []
#     trial_combos = [k for k in itools.combinations(np.arange(ntraining_sentences),  v)]
#     print(trial_combos)
#     for t in trial_combos:
	for s in np.arange(nsamples):
		t=random.sample(np.arange(training_mt_chunks).tolist(),v)
		print('**********************************')
		print('**********************************')
		print('The length of the number of training set stimuli is: %s' %(len(t)))
#         new_training_inds = [training_inds[n] for n in t]


		allinds = range(tResp_orig.shape[0]) #list of every time sample 
		indchunks = list(zip(*[iter(allinds)]*chunking_mt)) #gives different chunks of size defined by chunking_mt
		random.shuffle(indchunks) #shuffles all chunks so it will shuffle order in time and randomly select
		heldinds = list(itools.chain(*indchunks[:v])) #randomly selecting v from current number of chunks in loop
		

		tStim = tStim_orig[heldinds,:]
		tResp = tResp_orig[heldinds,:]

		chunklen = np.int(len(delays)*3) # We will randomize the data in chunks 
		nchunks = np.floor(0.2*tStim.shape[0]/chunklen).astype('int')

		nchans = tResp.shape[1]
		print(tStim.shape)
		print((tResp).shape)
		print(vStim.shape)
		print(vResp.shape)
		print('**************************')
		print(chunklen)
#         Fit the STRFs - RUNNING THE MODEL HERE!
		wt, corrs, valphas, allRcorrs, valinds, pred, Pstim = bootstrap_ridge(tStim, tResp, vStim, vResp, 
																		  alphas, nboots, chunklen, nchunks, 
																		  use_corr=use_corr,  single_alpha = single_alpha, 
																		  use_svd=False, corrmin = 0.05,
																		  joined=[np.array(np.arange(nchans))])
	   
		corrs_reps[v].append(corrs)

	#add h5py here to append each corr iteration here
	with h5py.File('%s/envelope/%s_%s_corrs_reps_trainingData.hf5' %(data_dir, subject, stimulus_class), 'a') as g:
		g.create_dataset('/rep%d' %(v), data=np.array(corrs))
		g.create_dataset('/weight%d' %(v), data=np.array(wt))