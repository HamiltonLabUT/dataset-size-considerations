import h5py
import numpy as np
import matplotlib 
from matplotlib import pyplot as plt 
from matplotlib import cm, rcParams
# import ffmpeg
from kneed import DataGenerator, KneeLocator
import os
import pandas as pd
import scipy
from scipy import stats
import mne
import seaborn as sns
#plot weights at t=0, t=kneepoint, t=final rep



def individual_subj_corrs(datapath, model_type, stimulus_class, subject_list=['MT0008']):
	'''
	Parameters:
	-----------
	subject_list : list
		- list of all subjects defined at the top of the script. Just individual subject for now. 
	datapath : string 
		- path to where the .h5 files live in Box
	model_type: string 
		The acceptable inputs for both MovieTrailers and TIMIT are:
			- auditory_training_data: full auditory model with envs, pitch, phnfeat
			- envelope: envelope only (1 feature)
			- pitch: pitch only (1 feature)
			- phnfeat : phonological features (14 features)
	stimulus_class : string 
		- TIMIT or MovieTrailers (spelled exactly as written)

	Returns
	-------
	x : the number of reps of training data 
	y : average correlation value for each repetition of training data added


	'''
	if stimulus_class == 'TIMIT':
		nreps=361
		chunk_len = 1
	elif stimulus_class == 'MovieTrailers':
		nreps=1277
		chunk_len = 2
	else:
		print('Stimulus class not recognized')
	all_corrs = np.zeros((nreps, len(subject_list)))
	all_corrs[:] = np.NaN


	x=[]
	y = []
	for idx, subj in enumerate(subject_list):
		corrs_avg = []
		#corrs_std = []
		with h5py.File('%s/%s/%s/%s_%s_corrs_reps_trainingData.hf5'%(datapath, stimulus_class, model_type,subj,stimulus_class), 'r') as fh:
			# print([k for k in fh.keys() if k[:2]=='re'])
			repkeys = [np.int(k[3:]) for k in fh.keys() if k[:2]=='re']
			repkeys.sort()
			# reps = np.arange(len(fh))
			# reps_list=reps.tolist()
			for i in repkeys:
				corrs=[k for k in fh['/rep%d' %(i)]]
				y.append(np.mean(corrs)) #append avg from corrs 
				x.append(i)
	return x, y 

#####
def plot_corrs(corrs, subj, rep_1=10, rep_last=1014, chunk_len=2):
	'''
	Plot correlation values based on increasing number of repetitions for a single subject
	This function gets called in the `training_plot_corrs` below.
	'''
	corrmean=np.array(corrs).mean(1)
	corrstd=np.array(corrs).std(1)/np.sqrt(64)
	x=np.linspace(chunk_len*rep_1, chunk_len*rep_last, len(corrs))
	#plt.fill_between(x, corrmean+corrstd, corrmean-corrstd, alpha=0.5)
	plt.plot(x, corrmean, color='#808080', alpha=0.5)
	plt.xlabel('Amt. Training Data (s)')
	plt.ylabel('r')

def training_plot_corrs(subject_list, datapath, model_type, stimulus_class, chans_64=True, save_fig=True):
	'''
	Parameters:
	-----------
	subject_list : list
		- list of all subjects defined at the top of the script.
	datapath : string 
		- path to where the .h5 files live in Box
	model_type: string 
		The acceptable inputs for both MovieTrailers and TIMIT are:
			- auditory_training_data: full auditory model with envs, pitch, phnfeat
			- envelope: envelope only (1 feature)
			- pitch: pitch only (1 feature)
			- phnfeat : phonological features (14 features)
	stimulus_class : string 
		- TIMIT or MovieTrailers (spelled exactly as written)

	Returns
	-------
	kneepoint_avg : kneepoint calculation
	reps_list : the number of reps of training data 
	grand_corr_avg : average correlation value across all subjects 
	grand_corr_std : avg. standard deviation for corr values across all subjects 

	'''
	if stimulus_class == 'TIMIT':
		nreps=361
		chunk_len = 1
	elif stimulus_class == 'MovieTrailers':
		nreps=1277
		chunk_len = 2
	else:
		print('Stimulus class not recognized')
	all_corrs = np.zeros((nreps, len(subject_list)))
	all_corrs[:] = np.NaN

	kneepoint = []
	for idx, subj in enumerate(subject_list):
		corrs_avg = []
		corrs_std = []
		with h5py.File('%s/%s/%s/%s_%s_corrs_reps_trainingData.hf5'%(datapath, stimulus_class, model_type,subj,stimulus_class), 'r') as fh:
		
			# print([k for k in fh.keys() if k[:2]=='re'])
			repkeys = [np.int(k[3:]) for k in fh.keys() if k[:2]=='re']
			repkeys.sort()
			# reps = np.arange(len(fh))
			# reps_list=reps.tolist()
			for i in repkeys:
				if chans_64:
					corrs=[k for k in fh['/rep%d' %(i)]]
					title_name = '64 channels'
					print(len(corrs))
				else:
					print('Only providing corrs for the first 32 channels if you were recording from a 32 channel system')
					corrs=[k for k in fh['/rep%d' %(i)]][:32] #for first 32 channels only
					title_name = '32 channels'
					print(len(corrs))
				corrs_avg.append(np.mean(corrs)) #append avg from corrs 
				corrs_std.append(np.std(corrs)/np.sqrt(len(corrs))) #standard error 
			#all_corrs[:,idx]=np.array(corrs_avg)
			corrs_array = np.array(corrs_avg)
			print('Now processing: %s' %(subj))
			all_corrs[:corrs_array.shape[0], idx] = corrs_array
			kneedle = KneeLocator(repkeys, corrs_avg, S=1.0, curve="concave", direction="increasing", interp_method='polynomial')
			#kneepoint.append(np.argmin(np.abs(np.diff(np.diff(corrs_array)))))
			kneepoint.append(kneedle.knee)


			plt.plot(repkeys, corrs_avg, color='#808080', alpha=0.5)
			#plt.plot(repkeys, corrs_avg, color='red', alpha=0.5)
			
			plt.annotate(f'{subj}', xy=(repkeys[-1],corrs_avg[-1]), xycoords='data')
			plt.xlabel('Increasing number of training set stimuli')
			plt.ylabel('Average correlation for training set')
			#print(len(corrs))
			# if save_fig:
			# 	plt.savefig('%s/figures/%s_%s_corrs/%s_%s_frame%04d.png' %(datapath, subject, stimulus_class, stimulus_class, subject, i))


	#grand_corr_avg = np.array(all_corrs).mean(1)
	grand_corr_avg = np.nanmean(all_corrs, axis=1)
	corrs_array = np.array(all_corrs)
	reps_list = list(np.arange(10, nreps+10))
	grand_corr_std = np.nanstd(corrs_array, axis=1)/np.sqrt(corrs_array.shape[1])

	if model_type == 'phnfeat':
		color='red'
	if model_type == 'pitch':
		color ='blue'
	if model_type == 'envelope':
		color = 'gray'
	if model_type == 'auditory_training_data':
		color = 'blue'
	plt.fill_between(reps_list, grand_corr_avg+grand_corr_std, grand_corr_avg-grand_corr_std, alpha=0.6, color=color)

	
	rep_1=10
	# rep_last=1014
	#chunk_len=1
	rep_last=nreps
	y=np.linspace(chunk_len*rep_1, chunk_len*rep_last, len(reps_list))
	# kneepoint = np.argmin(np.abs(np.diff(np.diff(grand_corr_avg))))
	print(np.mean(kneepoint))
	kneepoint_avg = np.mean(kneepoint)

	#print(np.mean(kneepoint))
	#kneepoint_avg = np.mean(kneepoint)
	kneepoint_se = np.std(kneepoint)/np.sqrt(len(kneepoint)) #plot this 

	plt.vlines(np.mean(kneepoint), plt.ylim()[0], plt.ylim()[1], linestyles='dashed', color='black')
	plt.axvspan(kneepoint_avg-kneepoint_se, kneepoint_avg + kneepoint_se, alpha=0.5, color='gray')
	# plt.fill_betweenx(corrs_avg, np.mean(kneepoint) + np.std(kneepoint)/np.sqrt(len(kneepoint)), np.mean(kneepoint) - np.std(kneepoint)/np.sqrt(len(kneepoint)), alpha=0.3)

	#plt.legend(subject_list)
	plt.title(f'{stimulus_class} : {model_type}. {title_name}')
	plt.savefig('%s/figures/allSubjects_%s_%s_Training_corrs_%s.pdf' %(datapath, model_type,stimulus_class,title_name))

	return kneepoint_avg, kneepoint_se, reps_list, grand_corr_avg, grand_corr_std, kneepoint

def percent_change(kneepoint, grand_corr_avg, stimulus_class):
	print(stimulus_class)
	print('The percentage increase between rep1 and kneepoint is: ')
	rep_knee_rep0 = ((grand_corr_avg[int(np.mean(kneepoint))]-grand_corr_avg[0])/grand_corr_avg[0])*100
	print(rep_knee_rep0)
	print('The percentage increase between final rep and kneepoint is: ')
	rep_final_rep_knee = ((grand_corr_avg[-1]-grand_corr_avg[int(np.mean(kneepoint))])/grand_corr_avg[int(np.mean(kneepoint))])*100
	print(rep_final_rep_knee)
	
####### WEIGHTS PLOTTING #########
	
def plot_weight_times(datapath, stimulus_class, subject='MT0008', save_fig=True):
	'''
	Generates image of receptive field for each subject for each repetition and outputs the image as a .PDF

	'''
	if stimulus_class == 'TIMIT':
		nreps=361
	elif stimulus_class == 'MovieTrailers':
		nreps=1277
	else:
		print('Undefined stimulus class')
	all_wts = np.empty((nreps, len(subject)))
	all_wts[:] = np.NaN

	wts = dict()


	print('Now processing subject: %s' %(subject))
	fname = '%s/%s/auditory_training_data/%s_%s_corrs_reps_trainingData.hf5'%(datapath, stimulus_class, subject, stimulus_class)
	wts[subject]=[]
	with h5py.File(fname,'r') as f:
		repkeys = [np.int(k[3:]) for k in f.keys() if k[:2]=='re']
		repkeys.sort()
		for r in repkeys:
			wts[subject].append(f['weight%d'%(r)][:])

	delay_min = 0.0
	delay_max = 0.6
	wt_pad = 0.1 # Amount of padding for delays, since edge artifacts can make weights look weird

	fs = 128.0
	delays = np.arange(np.floor((delay_min-wt_pad)*fs), np.ceil((delay_max+wt_pad)*fs), dtype=np.int) #create array to pass time delays in
	ndelays = len(delays)
	print("Delays:", delays)

	# wts_avg = []

	subj_wts = np.array(wts[subject])
	x = subj_wts.reshape(subj_wts.shape[0], ndelays, np.int(subj_wts.shape[1]/ndelays),  subj_wts.shape[2])
	#smax = x[0].max() 
	smax = x.mean(3).max()


	feat_labels = ['sonorant','obstruent','voiced','back','front','low','high','dorsal',
				   'coronal','labial','syllabic','plosive','fricative','nasal', 'env', 'F0']

	nfeats = len(feat_labels)
	t = np.linspace(delay_min, delay_max, len(delays))
	#weights_averaging = np.vstack(x)
                    

	#create folder to save each weight if it does not exist:
	# if not os.path.exists('%s/figures/%s_%s_wts/' %(datapath, subject, stimulus_class)
	os.makedirs('%s/figures/%s_%s_wts/' %(datapath, subject, stimulus_class), exist_ok=True)


	for i in np.arange(x.shape[0]):
		if i%1==0:
			plt.clf()
			plt.imshow(x[i,:,:].mean(2).T, vmin=-smax, vmax=smax, cmap=cm.RdBu_r, aspect='auto', interpolation='nearest')
			plt.gca().set_xticks([0, (len(delays)-1)/2, len(delays)-1])
			plt.gca().set_xticklabels([t[0], t[np.int((len(delays)-1)/2)], t[len(delays)-1]])
			plt.gca().set_yticks(np.arange(nfeats))
			plt.gca().set_yticklabels(feat_labels)
			plt.xlabel('Delays (s)')
			plt.ylabel('Correlation values (r)')

			plt.title(i)
			plt.colorbar()

			#plt.show()
			#plt.pause(0.01)
			if save_fig:
				plt.savefig('%s/figures/%s_%s_wts/%s_%s_frame%04d.jpg' %(datapath, subject, stimulus_class, stimulus_class, subject, i), dpi=600)


def wts_stabilization(datapath, stimulus_class,  model_type, subject, save_fig=True):
	if stimulus_class == 'TIMIT':
		nreps=361
	elif stimulus_class == 'MovieTrailers':
		nreps=1277
	else:
		print('Undefined stimulus class')
	all_wts = np.empty((nreps, len(subject)))
	all_wts[:] = np.NaN

	wts = dict()


	print('Now processing subject: %s' %(subject))
	fname = '%s/%s/%s/%s_%s_corrs_reps_trainingData.hf5'%(datapath, stimulus_class, model_type, subject, stimulus_class)
	wts[subject]=[]
	with h5py.File(fname,'r') as f:
		repkeys = [np.int(k[3:]) for k in f.keys() if k[:2]=='re']
		repkeys.sort()
		for r in repkeys:
			wts[subject].append(f['weight%d'%(r)][:])


	delay_min = 0.0
	delay_max = 0.6
	wt_pad = 0.1 # Amount of padding for delays, since edge artifacts can make weights look weird

	fs = 128.0
	delays = np.arange(np.floor((delay_min-wt_pad)*fs), np.ceil((delay_max+wt_pad)*fs), dtype=np.int) #create array to pass time delays in
	ndelays = len(delays)
	print("Delays:", delays)

	# wts_avg = []

	subj_wts = np.array(wts[subject])
	x = subj_wts.reshape(subj_wts.shape[0], ndelays, np.int(subj_wts.shape[1]/ndelays),  subj_wts.shape[2])


	# feat_labels = ['sonorant','obstruent','voiced','back','front','low','high','dorsal',
	# 			   'coronal','labial','syllabic','plosive','fricative','nasal', 'env', 'F0']

	# nfeats = len(feat_labels)
	# t = np.linspace(delay_min, delay_max, len(delays))
	# weights_averaging = np.vstack(x)

	wts_vector = []
	y=x.mean(axis=(1,2)) #average across delays and number of features
	for i in np.arange(y.shape[0]):
		if i == y.shape[0]-1:
			break 
		else:
			wts_vector.append(np.corrcoef(y[i], y[i+1])[0,1])

	if save_fig:
		plt.plot(wts_vector, '.', color='red')
		plt.plot(wts_vector, color='black')
		plt.xlabel('Rep num')
		plt.ylabel('correlation value "r"')
		plt.title(f'{subject} {model_type}')

		plt.ylim([-0.75, 1.00])

		plt.savefig('%s/figures/%s_%s_wts/%s_%s_wts_stabilizationPlot.pdf' %(datapath, subject, stimulus_class, stimulus_class, subject))

def lmem_csv(user, datapath, subject_list):
	stimulus_class = ['TIMIT', 'MovieTrailers']

	model_types = ['auditory_training_data', 'envelope', 'pitch', 'phnfeat']

	all_subjs = []
	all_models = []


	kneepoint = []
	#kneepoint = dict()
	stim_type = []


	for s in stimulus_class:
		if s == 'TIMIT':
			nreps=371
		else:
			nreps=1277
		all_corrs = np.zeros((nreps, len(subject_list)))
		all_corrs[:] = np.NaN
		for m in model_types:
			for idx, subj in enumerate(subject_list):
				corrs_avg = []
				corrs_std = []
				with h5py.File('%s/%s/%s/%s_%s_corrs_reps_trainingData.hf5'%(datapath, s, m,subj,s), 'r') as fh:
					# print([k for k in fh.keys() if k[:2]=='re'])
					repkeys = [np.int(k[3:]) for k in fh.keys() if k[:2]=='re']
					repkeys.sort()
					# reps = np.arange(len(fh))
					# reps_list=reps.tolist()
					for i in repkeys:
						corrs=[k for k in fh['/rep%d' %(i)]]
						corrs_avg.append(np.mean(corrs)) #append avg from corrs 
						corrs_std.append(np.std(corrs)/np.sqrt(len(corrs))) #standard error 
					#all_corrs[:,idx]=np.array(corrs_avg)
					corrs_array = np.array(corrs_avg)
					print('Now processing: %s' %(subj))
					all_corrs[:corrs_array.shape[0], idx] = corrs_array
					kneedle = KneeLocator(repkeys, corrs_avg, S=1.0, curve="concave", direction="increasing", interp_method='polynomial')
					#kneepoint.append(np.argmin(np.abs(np.diff(np.diff(corrs_array)))))
					kneepoint.append(kneedle.knee)
					all_subjs.append(subj)
					all_models.append(m)
					stim_type.append(s)


	data = {'subject': all_subjs, 'model': all_models, 'kneepoint': np.array(kneepoint).ravel(), 'stimulus_set': stim_type}
	df = pd.DataFrame.from_dict(data)
	df.to_csv(f'{datapath}/kneepoint.csv')

def subj_kneepoint(subject_list, datapath, model_type, stimulus_class):
	'''
	Parameters:
	-----------
	subject_list : list
		- list of all subjects defined at the top of the script.
	datapath : string 
		- path to where the .h5 files live in Box
	model_type: string 
		The acceptable inputs for both MovieTrailers and TIMIT are:
			- auditory_training_data: full auditory model with envs, pitch, phnfeat
			- envelope: envelope only (1 feature)
			- pitch: pitch only (1 feature)
			- phnfeat : phonological features (14 features)
	stimulus_class : string 
		- TIMIT or MovieTrailers (spelled exactly as written)

	Returns
	-------
	kneepoint : kneepoint calculation

	'''
	if stimulus_class == 'TIMIT':
		nreps=361
		chunk_len = 1
	elif stimulus_class == 'MovieTrailers':
		nreps=1277
		chunk_len = 2
	else:
		print('Stimulus class not recognized')
	all_corrs = np.zeros((nreps, len(subject_list)))
	all_corrs[:] = np.NaN

	kneepoint = []
	for idx, subj in enumerate(subject_list):
		corrs_avg = []
		corrs_std = []
		with h5py.File('%s/%s/%s/%s_%s_corrs_reps_trainingData.hf5'%(datapath, stimulus_class, model_type,subj,stimulus_class), 'r') as fh:
			# print([k for k in fh.keys() if k[:2]=='re'])
			repkeys = [np.int(k[3:]) for k in fh.keys() if k[:2]=='re']
			repkeys.sort()
			# reps = np.arange(len(fh))
			# reps_list=reps.tolist()
			for i in repkeys:
				corrs=[k for k in fh['/rep%d' %(i)]]
				corrs_avg.append(np.mean(corrs)) #append avg from corrs 
				corrs_std.append(np.std(corrs)/np.sqrt(len(corrs))) #standard error 
			#all_corrs[:,idx]=np.array(corrs_avg)
			corrs_array = np.array(corrs_avg)
			print('Now processing: %s' %(subj))
			all_corrs[:corrs_array.shape[0], idx] = corrs_array
			kneedle = KneeLocator(repkeys, corrs_avg, S=1.0, curve="concave", direction="increasing", interp_method='polynomial')
			#kneepoint.append(np.argmin(np.abs(np.diff(np.diff(corrs_array)))))
			kneepoint.append(kneedle.knee)
	return kneepoint 


#rank test comparison
def rank_test(datapath, stimulus_class, subject_list, model_type, kneepoint, user='maansidesai'):
	dims = []
	for idx, i in enumerate(subject_list):
		# raw=mne.io.read_raw_fif(f'{i}_postICA_rejected.fif', preload=True)
		raw=mne.io.read_raw_fif(f'/Users/{user}/Box/generalizable_EEG_manuscript/data/participants/{i}/downsampled_128/{i}_postICA_rejected.fif', preload=True)
		#rank=np.linalg.matrix_rank(raw.get_data()[:64,:])
		data=raw.get_data()[:64,:]
		covmat = np.dot(data, data.T)
		eigenval, eigenvec = np.linalg.eig(covmat)
		dimensionality = np.sum(eigenval)**2/np.sum(eigenval**2)
		dims.append(dimensionality)

		plt.plot(kneepoint[idx], dimensionality, '.', label=i, color='red')
		plt.annotate(f'{i}', xy=(kneepoint[idx],dimensionality), xycoords='data')
	# plt.gca().set_xticks(np.arange(len(subject_list)))
	# plt.gca().set_xticklabels(subject_list)
	
	sns.regplot(kneepoint, dims)
	if stimulus_class == 'TIMIT':
		label = '2-second long sentences of training data'
	else:
		label = '2-second chunks of training data'
	plt.xlabel(f'{label}')
	plt.ylabel('rank')
	plt.title(f'{stimulus_class}')
	
	plt.savefig('%s/figures/ranktest_kneepoint_%s.pdf' %(datapath, stimulus_class))

	#plt.legend()

def snr_kneepoint(datapath, eegpath, stimulus_class, subject_list, model_type, kneepoint, user='maansidesai'):
	if stimulus_class == 'MovieTrailers':
		type = 'MT'
	else:
		type = 'TIMIT'
	corrs = []
	if model_type == 'auditory_training_data':
		h5_strf = 'pitchenvsphnfeat'
	for idx, i in enumerate(subject_list):
		#corrs, reps = individual_subj_corrs(datapath, model_type, stimulus_class, subject_list=[i])
		with h5py.File(f'{eegpath}/{i}/{i}_STRF_by_{h5_strf}_{type}.hf5', 'r') as f:
			norm_corr_avg = np.nanmean(f['/corrs_%s_norm' %(type.lower())][:])
			corrs.append(norm_corr_avg)


		plt.plot(norm_corr_avg, kneepoint[idx], '.', color='red')

	plt.xlabel('maximum correlation (noise ceiling)')
	plt.ylabel('kneepoint')

	plt.title(f'SNR vs. kneepoint for {stimulus_class}')


	#For movie trailers
	if stimulus_class == 'MovieTrailers':
		[slope, intercept, r_value, p_value, std_err] = scipy.stats.linregress(corrs, kneepoint)
		plt.plot([0.01,0.17], [0.01*slope + intercept, 0.17*slope+intercept], color='black')
		plt.annotate(f'p-value: {p_value}', xy=(0.12, 160),
				xytext=(0.12, 160), size=10, color = 'black')
		plt.annotate(f'r-value: {r_value}', xy=(0.12, 150),
				xytext=(0.12, 140), size=10, color = 'black')

	else:
		corrs[15]=0
		[slope, intercept, r_value, p_value, std_err] = scipy.stats.linregress(corrs, kneepoint)
		plt.plot([0.15,0.7], [0.15*slope + intercept, 0.7*slope+intercept], color='black')
		plt.annotate(f'p-value: {p_value}', xy=(0.6, 160),
		xytext=(0.55, 60), size=10, color = 'black')

		plt.annotate(f'r-value: {r_value}', xy=(0.6, 150),
				xytext=(0.55, 50), size=10, color = 'black')
	print(r_value)
	print(p_value)

def broderick_data_corrs(datapath, subject_list, nreps=1448, chunk_len=2, chans_128 = True, save_fig=True):
	all_corrs = np.zeros((nreps, len(subject_list)))
	all_corrs[:] = np.NaN
	kneepoint = []
	for idx, subj in enumerate(subject_list):
		corrs_avg = []
		corrs_std = []
		with h5py.File(f'{datapath}/{subj}_corrs_reps_trainingData.hf5', 'r') as fh:
			# print([k for k in fh.keys() if k[:2]=='re'])
			repkeys = [np.int(k[3:]) for k in fh.keys() if k[:2]=='re']
			repkeys.sort()
			# reps = np.arange(len(fh))
			# reps_list=reps.tolist()
			for i in repkeys:
				if chans_128:
					corrs=[k for k in fh['/rep%d' %(i)]]
					title_name = '128 channels'
					print(len(corrs))
					color = '#808080'
					alpha = 0.5
				else:
					print('Only providing corrs for the first 64 channels if you were recording from a 128 channel system')
					corrs=[k for k in fh['/rep%d' %(i)]][:64] #for first 64 channels only
					title_name = '64 channels'
					print(len(corrs))
					color = 'red'
					alpha = 0.3
				corrs_avg.append(np.mean(corrs)) #append avg from corrs 
				corrs_std.append(np.std(corrs)/np.sqrt(len(corrs))) #standard error 
			#all_corrs[:,idx]=np.array(corrs_avg)
			corrs_array = np.array(corrs_avg)
			print('Now processing: %s' %(subj))
			all_corrs[:corrs_array.shape[0], idx] = corrs_array
			kneedle = KneeLocator(repkeys, corrs_avg, S=1.0, curve="concave", direction="increasing", interp_method='polynomial')
			#kneepoint.append(np.argmin(np.abs(np.diff(np.diff(corrs_array)))))
			kneepoint.append(kneedle.knee)

			plt.plot(repkeys, corrs_avg, color=color, alpha=alpha)
			#plt.plot(repkeys, corrs_avg, color='#808080', alpha=0.5)
			#plt.annotate(f'{subj}', xy=(repkeys[-1],corrs_avg[-1]), xycoords='data')
			plt.xlabel('# of 2-second chunks of training set stimuli')
			plt.ylabel('Average correlation for training set')
	#grand_corr_avg = np.array(all_corrs).mean(1)
	grand_corr_avg = np.nanmean(all_corrs, axis=1)
	corrs_array = np.array(all_corrs)
	reps_list = list(np.arange(10, nreps+10))
	grand_corr_std = np.nanstd(corrs_array, axis=1)/np.sqrt(corrs_array.shape[1])


	plt.fill_between(reps_list, grand_corr_avg+grand_corr_std, grand_corr_avg-grand_corr_std, alpha=0.6, color='green')

	
	rep_1=10
	# rep_last=1014
	#chunk_len=1
	rep_last=nreps
	y=np.linspace(chunk_len*rep_1, chunk_len*rep_last, len(reps_list))
	# kneepoint = np.argmin(np.abs(np.diff(np.diff(grand_corr_avg))))
	print(np.mean(kneepoint))
	kneepoint_avg = np.mean(kneepoint)

	#print(np.mean(kneepoint))
	#kneepoint_avg = np.mean(kneepoint)
	kneepoint_se = np.std(kneepoint)/np.sqrt(len(kneepoint)) #plot this 

	plt.vlines(np.mean(kneepoint), plt.ylim()[0], plt.ylim()[1], linestyles='dashed', color='black')
	plt.axvspan(kneepoint_avg-kneepoint_se, kneepoint_avg + kneepoint_se, alpha=0.5, color='gray')
	#plt.axvspan(kneepoint_avg-kneepoint_se, kneepoint_avg + kneepoint_se, alpha=0.5, color='red')
	# plt.fill_betweenx(corrs_avg, np.mean(kneepoint) + np.std(kneepoint)/np.sqrt(len(kneepoint)), np.mean(kneepoint) - np.std(kneepoint)/np.sqrt(len(kneepoint)), alpha=0.3)

	#plt.legend(subject_list)
	if save_fig:
		plt.title(f'Audiobook data')
		plt.savefig('%s/figures/allSubjects_audiobook_Training_corrs.pdf' %(datapath))

	return kneepoint, kneepoint_avg, grand_corr_avg