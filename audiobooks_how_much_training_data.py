import scipy.io 
import h5py
import numpy as np
import glob
from run_STRF_analysis import *

def drop_channels(user, subject_list, files):
    ch_names = []
    for idx, subject in enumerate(subject_list):
        eeg = glob.glob(f'{files}/EEG/{subject}/*.fif')
        eeg.sort()
        
        for t in eeg:
            idx=idx+1
            #eeg_mat = scipy.io.loadmat(t)
            eeg_mat = mne.io.read_raw_fif(t)
            ch_names.append(eeg_mat.info['ch_names'])
    picks = set.intersection(*map(set,ch_names))
    print(len(picks))
    return picks

def create_h5_audiobooks(user, files, subject_list, picks, h5_name='audiobook_training'):
    with h5py.File(f'{files}/{h5_name}.hf5', 'a') as g:
        for idx, subject in enumerate(subject_list):

            #add EEG data
            eeg = glob.glob(f'{files}/EEG/{subject}/*.fif')
            eeg.sort()
            for t in eeg:
                idx=idx+1
                #eeg_mat = scipy.io.loadmat(t)
                eeg_mat = mne.io.read_raw_fif(t)
                bad_chs = [ch for ch in eeg_mat.info['ch_names'] if ch not in picks]
                print('The bad channels are: ')
                print(eeg_mat.info['bads'])
                eeg_mat.drop_channels(bad_chs)
                #data_eeg = eeg_mat['eegData'] 
                data_eeg = eeg_mat.get_data()
                print(data_eeg.shape)
                i = t.split('/')[-1].split(f'{subject}_')[1].split('_ICA.fif')[0]

                try:
                    g.create_dataset(f'/resp/{subject}/%s' %(i), data=np.array(data_eeg))
                except:
                    'Already exists'

        #now add envelope stimuli
        stim = glob.glob(f'{files}/Stimuli/Envelopes/*.mat')
        stim.sort()
        for id, i in enumerate(stim):
            id = id+1
            stim_mat = scipy.io.loadmat(i)
            data_envs = stim_mat['env']
            run_num = int(i.split('/')[-1].split('_128Hz.mat')[0].split('audio')[1])
            try:
                g.create_dataset('/stim/env%d' %(run_num), data=np.array(data_envs))
            except:
                print('Envelope stim already exists')


#open h5 file and create stim and resp dictionaries

def audiobook_boostrap(subj, files, rep_num=10, h5_name='audiobook_training.hf5'):
    stim_dict = dict()
    resp_dict = dict()
    with h5py.File(f'{files}/{h5_name}','r') as fh:

        all_stim = [k for k in fh['/stim'].keys()]
        print(all_stim)

        for idx, wav_name in enumerate(all_stim): 
            idx = idx+1
            print(wav_name)
            stim_dict[wav_name] = []
            resp_dict[wav_name] = []

            #f'/resp/{subject}/%s' %(i)
            epochs_data = fh[f'resp/{subj}/Run{idx}'][:]
            print('Original shape of epochs is: ')
            print(epochs_data.shape)
            envs = fh[f'/stim/env{idx}'][:]
            ntimes = envs.shape[0] #always resample to the size of envelope (0th dimension) 

            envs = fh[f'/stim/env{idx}'][:]
            #envs = scipy.signal.resample(envs, ntimes) #resampling to size of phnfeat
            
            print('envs shape is:')
            print(envs.shape)

            epochs_data = np.expand_dims(epochs_data.T, axis=0)
            if envs.shape[0] < epochs_data.shape[1]:
                epochs_data = epochs_data[:,:envs.shape[0],:]
                print(epochs_data.shape)
                
            else:
                envs = envs[:epochs_data.shape[1],:]
                print(envs.shape)
            
            stim_dict[wav_name].append(envs.T)
            resp_dict[wav_name].append(epochs_data)

    #rep_num = 10 
    stim_list = []
    for key in resp_dict.keys():
        print(key)
        stim_list.append(key)
    all_stimuli = [k for k in stim_list if len(resp_dict[k]) > 0]
    #test_set = random.sample(all_stimuli, 4) #take ~20% of data for test_set. This is actually 21% 
    test_set = ['env10', 'env8', 'env7', 'env18'] 
    print('Printing test_set stimuli')
    print(test_set)
    training_set = np.setdiff1d(all_stimuli, test_set)
    print('Printing training_set stimuli')
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
    alphas = np.hstack((0, np.logspace(2,8,20))) # Gives you 15 values between 10^2 and 10^8

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
    current_stim_list_train = np.array([all_stimuli[r] for r in train_inds])
    current_stim_list_val = np.array([all_stimuli[r] for r in val_inds])


    # current_stim_list_train = np.array([all_stimuli[r][0] for r in train_inds])
    # current_stim_list_val = np.array([all_stimuli[r][0] for r in val_inds])
    # Create training and validation response matrices
    print(resp_dict[training_set[0]][0].shape)
    print(test_set)

    print(len(training_set))
    for r in training_set:
        print(r)


    vResp_numtrials = [resp_dict[r][0].shape[0] for r in training_set if resp_dict[r][0].shape[0]]

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
    #vResp = np.vstack((vResp_temp[0]))
    vResp = np.vstack((vResp_temp))
    vResp = vResp.astype(np.single)
    print(vResp.shape)


    vStim_temp = np.atleast_2d(np.vstack([np.vstack(stim_dict[r]).T for r in test_set]))

    vStim_temp = vStim_temp/vStim_temp.max(0)

    print(vStim_temp.max(0).shape)
    print('**********************************')
    vStim_temp = vStim_temp.astype(np.single)
    vStim = make_delayed(vStim_temp, delays)

    training_inds = train_inds #already have this above 
    ntraining_sentences = len(training_inds)

    tResp_orig = np.vstack(([resp_dict[r][0].mean(0) for r in training_set])) #words correspond to numbers for each stimulus type
    tResp_orig = tResp_orig.astype(np.single)

    tStim_temp = np.atleast_2d(np.vstack([np.vstack(stim_dict[r]).T for r in training_set]))
    tStim_temp = tStim_temp.astype(np.single)

    tStim_orig = make_delayed(tStim_temp, delays)

    #print(ntraining_sentences)
    # nsamples = 10

    # chunking_mt = 2*128 #time for MT for duration of chunking 
    # #training_mt_chunks = np.int(tResp_orig.shape[0]/chunking_mt)
    # training_mt_chunks = np.int(tResp_orig.shape[1]/chunking_mt)
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

        # with h5py.File('%s/%s_corrs_reps_trainingData.hf5' %(files, subj), 'a') as g:
        #     g.create_dataset('/rep%d' %(v), data=np.array(corrs))
        #     g.create_dataset('/weight%d' %(v), data=np.array(wt))

        with h5py.File('%s/%s_corrs_reps_trainingData.hf5' %(files, subj ), 'a') as g:
            g.create_dataset('/rep%d' %(v), data=np.array(corrs))
            g.create_dataset('/weight%d' %(v), data=np.array(wt))

    return corrs_reps, wts_rep

def returnNotMatches(a, b):
    list_not = [[x for x in a if x not in b], [x for x in b if x not in a]]
    return list_not


#######################
if __name__ == "__main__":
    user = 'maansidesai'
    files = f'/Users/{user}/Desktop/NaturalSpeech/'

    # Create .h5 file to save all matlab files (non-preprocessed EEG data + stim)
    subject_list = os.listdir(f'{files}/EEG')
    subject_list = [sub for sub in subject_list if 'Subject' in sub and int(re.search(r'\d+', sub).group())]
    subject_list.sort()
    print(subject_list)
    subj = input('Input subject number: ')


    for i in rep_list:
        corrs_reps, wts_rep = audiobook_boostrap(subj, files, rep_num=10, h5_name='audiobook_training.hf5')