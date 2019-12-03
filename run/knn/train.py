#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for training fixed-efficiency kNN regressor."""

# Basic import(s)
import gzip
import pickle

# Scientific import(s)
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import roc_curve

# Project import(s)
from adversarial.utils import parse_args, initialise, load_data, mkdir, saveclf  #, initialise_backend
from adversarial.profile import profile, Profile
from adversarial.constants import *
#from run.adversarial.common import initialise_config

# Local import(s)
from .common import *


# Main function definition
@profile
def main (args):

    # Initialise
    args, cfg = initialise(args)

    # Load data
    data, _, _ = load_data('data/' + args.input) #Train=True removed since we use the data file

    # -------------------------------------------------------------------------
    ####
    #### # Initialise Keras backend
    #### initialise_backend(args)
    ####
    #### # Neural network-specific initialisation of the configuration dict
    #### initialise_config(args, cfg)
    ####
    #### # Keras import(s)
    #### from keras.models import load_model
    ####
    #### # NN
    #### from run.adversarial.common import add_nn
    #### with Profile("NN"):
    ####     classifier = load_model('models/adversarial/classifier/full/classifier.h5')
    ####     add_nn(data, classifier, 'NN')
    ####     pass
    # -------------------------------------------------------------------------

    # Compute background efficiency at sig. eff. = 50%
    eff_sig = 0.10
    fpr, tpr, thresholds = roc_curve(data['signal'], data[VAR], sample_weight=data['weight'])
    idx = np.argmin(np.abs(tpr - eff_sig))
    print "Background acceptance @ {:.2f}% sig. eff.: {:.2f}% ({} > {:.2f})".format(eff_sig * 100., (fpr[idx]) * 100., VAR, thresholds[idx]) #changed from 1-fpr[idx]
    #print "Signal efficiency @ {:.2f}% bkg. acc.: {:.2f}% ({} > {:.2f})".format(eff_sig * 100., (fpr[idx]) * 100., VAR, thresholds[idx]) #changed from 1-fpr[idx]
    print "Chosen target efficiency: {:.2f}%".format(EFF)

    # Filling profile
    data = data[data['signal'] == 0]
    profile_meas, (x,y,z) = fill_profile(data)

    # Format arrays
    X = np.vstack((x.flatten(), y.flatten()))
    X = X.T
    Y = z.flatten()

    # Fit KNN regressor
    print "debugging more: x.shape = ", X.shape,", y.ndim = ", Y.ndim

    knn = KNeighborsRegressor(weights='distance')
    knn.fit(X, Y)

    # Save KNN classifier
    saveclf(knn, 'models/knn/knn_{:s}_{}_{}.pkl.gz'.format(VAR, EFF, MODEL))

    return 0


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()  # (adversarial=True)

    # Call main function
    main(args)
    pass
