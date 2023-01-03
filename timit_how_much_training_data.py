from run_STRF_analysis import *

# stimulus_class = input('Enter the stimulus class type: ') #TIMIT or MovieTrailers

# data_dir='/Users/md42732/Desktop/data/EEG/MovieTrailers/Participants/'

# user = input('Enter user for computer:   ')

# # data_dir='/Users/%s/Box/MovieTrailersTask/Data/EEG/Participants' %(user)
# #data_dir = '/Users/%s/Desktop/UT_Austin/Research/Data_Analysis/EEG/MovieTrailers/Participants/' %(user)
# data_dir='/Users/%s/Desktop/data/EEG/MovieTrailers/Participants/' %(user) #load data

# # data_dir='/Users/%s/Box/MovieTrailersTask/Data/EEG/Participants' %(user)
# #data_dir = '/Users/%s/Desktop/UT_Austin/Research/Data_Analysis/EEG/MovieTrailers/Participants/' %(user)


# subject = input('Enter subject ID: ') #e.g. MT0003
# stimulus_class = 'TIMIT'

# def training_stimuli_avg_corr(subject, stimulus_class, data_dir, save_dir, save_fig=False):
#if stimulus_class == 'TIMIT':

def run_timit(subject, data_dir, model_type, stimulus_class = 'TIMIT'):

	if model_type == 'full_model':
		resp_dict, stim_dict = loadEEGh5(subject,stimulus_class, data_dir, eeg_epochs=True, resp_mean = False,
									 binaryfeatmat = True, binarymat=False, envelope=True, pitch=True, binned_pitch=False, gabor_pc10=False, spectrogram=False, nat_sound=False)
		model_output = 'auditory_training_data'
		alpha_h5 = 'pitchenvsphnfeat'
	if model_type == 'envelope':
		resp_dict, stim_dict = loadEEGh5(subject, stimulus_class, data_dir,eeg_epochs=True, resp_mean = False, binarymat=False, 
				binaryfeatmat = False, envelope=True, pitch=False, gabor_pc10=False, spectrogram=False, 
				binned_pitches=False, spectrogram_scaled=False, scene_cut=False) 
		model_output = 'envelope'
		alpha_h5 = 'envs'

	if model_type == 'phnfeat':
		resp_dict, stim_dict = loadEEGh5(subject, stimulus_class, data_dir,eeg_epochs=True, resp_mean = False, binarymat=False, 
				binaryfeatmat = True, envelope=False, pitch=False, gabor_pc10=False, spectrogram=False, 
				binned_pitches=False, spectrogram_scaled=False, scene_cut=False) 
		model_output = 'phnfeat'
		alpha_h5 = 'phnfeat'

	if model_type == 'pitch':
		resp_dict, stim_dict = loadEEGh5(subject, stimulus_class, data_dir,eeg_epochs=True, resp_mean = False, binarymat=False, 
				binaryfeatmat = False, envelope=False, pitch=True, gabor_pc10=False, spectrogram=False, 
				binned_pitches=False, spectrogram_scaled=False, scene_cut=False) 
		model_output = 'pitch'
		alpha_h5 = 'pitch'


	# Check whether the output path to save strfs exists or not
	output_dir = f'/{data_dir}/{stimulus_class}/{model_output}'
	isExist = os.path.exists(output_dir)

	if not isExist:
		os.makedirs(output_dir)
		print("The new directory is created!")

	test_set = ['fcaj0_si1479.wav', 'fcaj0_si1804.wav', 'fdfb0_si1948.wav', 
	'fdxw0_si2141.wav', 'fisb0_si2209.wav', 'mbbr0_si2315.wav', 
	'mdlc2_si2244.wav', 'mdls0_si998.wav', 'mjdh0_si1984.wav', 
	'mjmm0_si625.wav']

	stim_list = []
	for key in resp_dict.keys():
		print(key)
		stim_list.append(key)
	all_stimuli = [k for k in stim_list if len(resp_dict[k]) > 0]
	training_set = np.setdiff1d(all_stimuli, test_set)
	print(training_set)


	# if stimulus_class == 'MovieTrailers':
	# 	resp_dict, stim_dict = loadEEGh5(subject, 'MovieTrailers', data_dir, eeg_epochs=True, 
	# 							 resp_mean = False, binaryfeatmat = True, binarymat=False, envelope=True,
	# 							 pitch=True, spectrogram=False)
	# 	# resp_dict_MT0020, stim_dict = loadEEGh5('MT0020', 'MovieTrailers', data_dir, eeg_epochs=True, 
	# 	# 						 resp_mean = False, binaryfeatmat = True, binarymat=False, envelope=True,
	# 	# 						 pitch=True, spectrogram=False)

	# 	trailers_list = ['angrybirds-tlr1_a720p.wav', 'bighero6-tlr1_a720p.wav', 'bighero6-tlr2_a720p.wav', 
	# 	'bighero6-tlr3_a720p.wav', 'cars-3-trailer-4_a720p.wav', 'coco-trailer-1_a720p.wav', 
	# 	'ferdinand-trailer-2_a720p.wav', 'ferdinand-trailer-3_a720p.wav', 'ice-dragon-trailer-1_a720p.wav', 
	# 	'incredibles-2-trailer-1_a720p.wav', 'incredibles-2-trailer-2_a720p.wav', 'insideout-tlr2zzyy32_a720p.wav',
	# 	'insideout-usca-tlr2_a720p.wav', 'moana-clip-youre-welcome_a720p.wav', 'paddington-2-trailer-1_a720p.wav', 
	# 	'pandas-trailer-2_a720p.wav', 'pele-tlr1_a720p.wav', 'the-breadwinner-trailer-1_a720p.wav', 
	# 	'the-lego-ninjago-movie-trailer-1_a720p.wav', 'the-lego-ninjago-movie-trailer-2_a720p.wav', 
	# 	'thelittleprince-tlr_a720p.wav', 'trolls-tlr1_a720p.wav']


	# 	# resp_dict = {}

	# 	# for k in trailers_list:
	# 	# 	resp_dict[k] = [np.concatenate((resp_dict_MT0002[k][0], resp_dict_MT0020[k][0]), axis=0)]
	# 	test_set = ['paddington-2-trailer-1_a720p.wav', 'insideout-tlr2zzyy32_a720p.wav'] #the test set for the remaining MTs

	# 	all_stimuli = trailers_list
	# 	training_set = np.setdiff1d(all_stimuli, test_set)
	# 	print(training_set)

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
	f = h5py.File('%s/participants/%s/%s_STRF_by_%s_TIMIT.hf5' %(data_dir, subject, subject, alpha_h5), 'r')
	alphas = [f['valphas_timit'][0]]

	#alphas = np.hstack((0, np.logspace(2,8,20))) # Gives you 15 values between 10^2 and 10^8

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
	wts_rep = dict()


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
	#print(ntraining_sentences)
	nsamples = 10
	for v in np.arange(10, ntraining_sentences+1): # Loop through all sentences, starting with one
	#for v in np.arange(10,15): # Loop through all sentences, starting with one
		corrs_reps[v] = []
		wts_rep[v] = []
	#     trial_combos = [k for k in itools.combinations(np.arange(ntraining_sentences),  v)]
	#     print(trial_combos)
	#     for t in trial_combos:
		for s in np.arange(nsamples):
			t=random.sample(np.arange(ntraining_sentences).tolist(),v)
			print('**********************************')
			print('**********************************')
			print('The length of the number of training set stimuli is: %s' %(len(t)))
	#         new_training_inds = [training_inds[n] for n in t]
	#         print(new_training_inds)
			new_training_set = [training_set[n] for n in t]

			print(new_training_set)


			tStim_temp = np.atleast_2d(np.vstack([np.vstack(stim_dict[r]).T for r in new_training_set]))

			tStim = make_delayed(tStim_temp, delays)

			chunklen = np.int(len(delays)*3) # We will randomize the data in chunks 
			tResp = np.hstack(([resp_dict[r][0].mean(0) for r in new_training_set])).T #words correspond to numbers for each stimulus type

			nchans = tResp.shape[1]
			nchunks = np.floor(0.2*tStim.shape[0]/chunklen).astype('int')

			# if stimulus_class == 'MovieTrailers':
			# 	chunking_mt = 2.0 #is this how you specify?? 
			# 	allinds = range(tResp)
			# 	indchunks = list(zip(*[iter(allinds)]*chunking_mt))
			# 	random.shuffle(indchunks)
			# 	heldinds = list(itools.chain(*indchunks[:nchunks]))
			# 	notheldinds = list(set(allinds)-set(heldinds))
			# 	valinds.append(heldinds)

			# 	tStim = Rstim[heldinds,:]
			# 	tResp = Rresp[heldinds,:]
			# else:
				
			
			# tStim_temp = np.atleast_2d(np.vstack([np.vstack(stim_dict[r]).T for r in new_training_set]))

			# tStim = make_delayed(tStim_temp, delays)

			
			#adding vResp:
	#         vResp_temp = [resp_dict[r][0].mean(0) for r in test_set]
	#         vResp = np.hstack((vResp_temp)).T
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
			wts_rep[v].append(wt)

		#save corrs in .h5 file 
		with h5py.File('%s/%s_%s_corrs_reps_trainingData.hf5' %(output_dir, subject, stimulus_class), 'a') as g:
			g.create_dataset('/rep%d' %(v), data=np.array(corrs))
			g.create_dataset('/weight%d' %(v), data=np.array(wt))


	# #plot correlation vals with STD
	# reps_list =  np.arange(10,371,1).tolist()
	# corrs_reps_avg = []
	# corrs_reps_std = []
	# for i in reps_list:
	# 	corrs_reps_avg.append(np.mean(corrs_reps[i]))
	# 	print(np.mean(corrs_reps[i]))
		
	# 	#print(np.array(corrs_reps[i])[:,23])
	# 	corrs_reps_std.append(np.std(corrs_reps[i])/np.sqrt(len(reps_list)))

	# plt.fill_between(reps_list, np.array(corrs_reps_avg)+np.array(corrs_reps_std), np.array(corrs_reps_avg)-np.array(corrs_reps_std), alpha=0.5)

	# plt.plot(reps_list, corrs_reps_avg)

	# plt.xlabel('Increasing number of training set stimuli')
	# plt.ylabel('Average correlation for training set')

	# if save_fig:
	# 	plt.savefig('%s/%s_%s_corrs_reps_trainingData.pdf' %(save_dir, subject, stimulus_class))

	return corrs_reps, wts_rep
