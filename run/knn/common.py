# -*- coding: utf-8 -*-

"""Common methods for training and testing fixed-efficiency kNN regressor."""

# Basic import(s)
import itertools

# Scientific import(s)
import ROOT
import numpy as np
import pandas as pd
from scipy.special import erf

# Project import(s)
from adversarial.utils import wpercentile, loadclf, garbage_collect
from adversarial.profile import profile

# Common definition(s)
VAR  = 'jet_ungrtrk500'   # 'D2' 'NN' | Substructure variable to decorrelate
EFF  = 1.0    # Fixed backround single-jet efficiency at which to perform decorrelation
MODEL = 'mcCR' #'All signal models'
INPUT = 'mcCR'
FIT = 'poly3'
MIN_STAT = 50
FIT_RANGE = (1000, 6000)
VARX = 'mjj'  # X-axis variable from which to decorrelate
VARY = 'mjj'   # Y-axis variable from which to decorrelate
VARS = [VARX, VARY]
AXIS = {      # Dict holding (num_bins, axis_min, axis_max) for axis variables
    'mjj': (50, 1000., 6000.),
    'mjj':  (50, 1000., 6000.),
}

#### ________________________________________________________________________
####
#### @NOTE: It is assumed that, for the chosen `VAR`, signal is towards small
####        values; and background towards large values.
#### ________________________________________________________________________


@garbage_collect

def func(x, a, b, c):
    """ error function"""
    y = []
    for x in range(len(x)):
        y.append(a * erf(b*(x+c))) 
    return y 

def standardise (array, y=None, rank=None):
    """
    Standardise axis-variables for kNN regression.

    Arguments:
        array: (N,2) numpy array or Pandas DataFrame containing axis variables.

    Returns:
        (N,2) numpy array containing standardised axis variables.
    """

    # If DataFrame, extract relevant columns and call method again.
    if isinstance(array, pd.DataFrame):
        if rank is None:
            X = array[[VARX, VARY]].values.astype(np.float)
        elif 'lead' in rank:
            X = array[[VARX, VARY]].values.astype(np.float)
            #X = array[[VARX, 'lead_'+VARY]].values.astype(np.float)
        elif 'sub' in rank:
            X = array[[VARX, VARY]].values.astype(np.float)
            #X = array[[VARX, 'sub_'+VARY]].values.astype(np.float)
        else:
            X = array[[VARX, VARY]].values.astype(np.float)
            print "Bad rank given to run.knn.commen.standardise?"
        return standardise(X)

    # If receiving separate arrays
    if y is not None:
        x = array
        assert x.shape == y.shape
        shape = x.shape
        X = np.vstack((x.flatten(), y.flatten())).T
        X = standardise(X)
        x,y = list(X.T)
        x = x.reshape(shape)
        y = y.reshape(shape)
        return x,y

    # Check(s)
    assert array.shape[1] == 2

    # Standardise
    X = np.array(array, dtype=np.float)
    for dim, var in zip([0,1], [VARX, VARY]):
        X[:,dim] -= float(AXIS[var][1])
        X[:,dim] /= float(AXIS[var][2] - AXIS[var][1])
        pass

    return X


@profile
def add_knn (data, feat=VAR, newfeat=None, path=None):
    """
    Add kNN-transformed `feat` to `data`. Modifies `data` in-place.

    Arguments:
        data: Pandas DataFrame to which to add the kNN-transformed variable.
        feat: Substructure variable to be decorrelated.
        newfeat: Name of output feature. By default, `{feat}kNN`.
        path: Path to trained kNN transform model.
    """

    # Check(s)
    assert path is not None, "add_knn: Please specify a model path."

    if newfeat is None:
        newfeat = '{}kNN'.format(feat)
        pass

    # Prepare data array
    if ('1D' in FIT) or ('lin' in FIT) or ('poly' in FIT) or ('erf' in FIT):
        X = data[VARX].values.astype(np.float)
        X = X.reshape(-1,1)
        
    else:
        X = standardise(data, rank=newfeat)

    # Load model
    knn = loadclf(path)
    
    # Add new classifier to data array

    if 'erf' in FIT: 
        print "HEJ ",  knn[0], knn[1], knn[2]
        if 'lead' in newfeat:
            data[newfeat] =  pd.Series(data['lead_'+feat] - func(X, knn[0], knn[1], knn[2]), index=data.index) #predict(X).flatten()
        if 'sub' in newfeat:
            data[newfeat] =  pd.Series(data['sub_'+feat] - func(X, knn[0], knn[1], knn[2]), index=data.index) #predict(X).flatten()
        else:
            data[newfeat] =  pd.Series(data[feat] - func(X, knn[0], knn[1], knn[2]), index=data.index) #predict(X).flatten()

    else:
        if 'lead' in newfeat:
            data[newfeat] =  pd.Series(data['lead_'+feat] - knn.predict(X), index=data.index) #predict(X).flatten()
        elif 'sub' in newfeat:
            data[newfeat] =  pd.Series(data['sub_'+feat] - knn.predict(X), index=data.index)  #predict(X).flatten()
        else:
            print "Something wrong with the newfeat name?"
            #print "Check shapes !!: ", data[feat].shape, knn.predict(X).flatten().shape

            data[newfeat] = pd.Series(data[feat] - knn.predict(X).flatten(), index=data.index) #

    return


@profile
def fill_profile (data):
    """Fill ROOT.TH2F with the measured, weighted values of the `EFF`-percentile
    of the background `VAR`. """

    # Define arrays
    shape   = (AXIS[VARX][0], AXIS[VARY][0])
    bins    = [np.linspace(AXIS[var][1], AXIS[var][2], AXIS[var][0] + 1, endpoint=True) for var in VARS]
    x, y, z = (np.zeros(shape) for _ in range(3))

    # Create `profile` histogram
    profile = ROOT.TH2F('profile', "", len(bins[0]) - 1, bins[0].flatten('C'), len(bins[1]) - 1, bins[1].flatten('C'))
    #data['weight1'] =  data['sample_weight']*data['MC_weight']

    # Fill profile
    for i,j in itertools.product(*map(range, shape)):

        # Bin edges in x and y
        edges = [bin[idx:idx+2] for idx, bin in zip([i,j],bins)]

        # Masks
        msks = [(data[var] > edges[dim][0]) & (data[var] <= edges[dim][1]) for dim, var in enumerate(VARS)]
        msk = reduce(lambda x,y: x & y, msks)

        # Percentile
        perc = np.nan
        if np.sum(msk) > 20:  # Ensure sufficient statistics for meaningful percentile. Was 20
            perc = wpercentile(data=data.loc[msk, VAR].values, percents=100-EFF, weights=data.loc[msk, 'TotalEventWeight'].values) #wpercentile
            pass

        x[i,j] = np.mean(edges[0])
        y[i,j] = np.mean(edges[1])
        z[i,j] = perc
        
        # Set non-zero bin content
        if perc != np.nan:
            profile.SetBinContent(i + 1, j + 1, perc)
            pass
        pass

    # Normalise arrays
    x,y = standardise(x,y, rank=None)

    # Filter out NaNs
    msk = ~np.isnan(z)
    x, y, z = x[msk], y[msk], z[msk]


    return profile, (x,y,z)

@profile
def fill_profile_1D (data):
    """Fill ROOT.TH2F with the measured, weighted values of the `EFF`-percentile
    of the background `VAR`. """

    # Define arrays
    #bins    = np.linspace(AXIS[VARX][1], AXIS[VARX][2], AXIS[VARX][0] + 1, endpoint=True)
    # Make variable sized bins
    #bins = np.linspace(AXIS[VARX][1], 4000, 40, endpoint=True)
    #bins = np.append(bins, [4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000])

    # Build bin structure with at least ?50 event in each bin 
    # and bin widths of at least AXIS[VARX][0]
    
    minBinSize = 100 #AXIS[VARX][0]
    binEdge = AXIS[VARX][2]
    binList = []
    binList.append(binEdge)
    k=1
    while binEdge-k*minBinSize > AXIS[VARX][1]:
        msk = (data[VARX] > binEdge-k*minBinSize) & (data[VARX] <= binEdge)
        if (np.sum(msk)*EFF/100. > MIN_STAT):
            binEdge -= k*minBinSize
            binList.append(binEdge)
            k=1
        else: 
            k+=1

    binList.append(AXIS[VARX][1])
    binList.reverse()
    bins = np.array(binList)
    print "Bins: ", len(bins), bins

    shape = len(bins) -1 #AXIS[VARX][0] # 
    x, y, e = (np.zeros(shape) for _ in range(3))

    # Create `profile` histogram
    profile = ROOT.TH1F('profile', "", len(bins)-1, bins)

    #if INPUT == "mc":
    #    data.loc[:,'TotalEventWeight'] /=  139000000.


    # Fill profile
    for i in (range(shape)):

        # Masks
        msk = (data[VARX] > bins[i]) & (data[VARX] <= bins[i+1])

        # Percentile
        #perc = np.nan
        #if np.sum(msk) > 20:  # Ensure sufficient statistics for meaningful percentile. Was 20
        perc = wpercentile(data=data.loc[msk, VAR].values, percents=100-EFF, weights=data.loc[msk, 'TotalEventWeight'].values) #wpercentile
        #   pass

        x[i] = np.mean([bins[i], bins[i+1]])
        y[i] = perc
        if np.sum(msk) > 0:
            e[i] = np.sqrt(np.sum(msk))/np.sum(msk)
        else:
            print "Bin ", i, " has np.sum(msk) < 20. Weird."
            e[i] = 0

        # Set non-zero bin content
        if perc != np.nan:
            profile.SetBinContent(i+1, perc)
            pass
        pass

    # Normalise array
    # x = standardise(x, rank=None)

    # Filter out NaNs
    msk = ~np.isnan(y)
    x, y, e = x[msk], y[msk], y[msk]


    return profile, (x,y,e)
