#!/bin/env python
# -*- coding: utf-8 -*-

# Basic import(s)
# ...

# Get ROOT to stop hogging the command-line options
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import root_numpy

# Scientific import(s)
import math
import numpy as np
from numpy.lib.recfunctions import append_fields
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsRegressor

# Project import(s)
from adversarial.utils import wpercentile, parse_args, initialise, load_data, latex, mkdir
from adversarial.profile import profile, Profile
from adversarial.constants import *
from .studies.common import *
from run.knn.common import add_knn, MODEL, VAR, EFF as knn_eff

# Custom import(s)
import rootplotting as rp


# Main function definition
@profile
def main (args):

    # Definitions
    histstyle = dict(**HISTSTYLE)

    # Initialise
    args, cfg = initialise(args)

    # Load data
    data, features, _ = load_data('data/' + args.input) #, test=True) # 

    outFile = ROOT.TFile.Open("figures/knn_jet_ungrtrk500_eff{}_data.root".format(knn_eff),"RECREATE")


    EFF = 0.5
    VAR = 'jet_ungrtrk500'
    VARX = 'dijetmass'
    FIT_RANGE = (0, 6000) # Necessary?

    #eff_sig = 0.50
    #fpr, tpr, thresholds = roc_curve(data['signal'], data[kNN_basevar], sample_weight=data['weight'])
    #idx = np.argmin(np.abs(tpr - eff_sig))
    #print "Background acceptance @ {:.2f}% sig. eff.: {:.2f}% ({} > {:.2f})".format(eff_sig * 100., (fpr[idx]) * 100., kNN_basevar, thresholds[idx]) #changed from 1-fpr[idx]
    #print "Chosen target efficiency: {:.2f}%".format(kNN_eff)


    weight = 'weight'  # 'weight_test' / 'weight'
    bins_mjj = np.linspace(100, 8000, 20)
    fineBins = np.linspace(100, 8000, 7900)
    fineBinsRe = fineBins.reshape(-1,1)

    percs = []
    for i in range(1, len(bins_mjj)):
        
        msk = (data[VARX] > bins_mjj[i-1]) & (data[VARX] <= bins_mjj[i]) & (data['signal']==0) 

        if np.sum(msk) > 20:  # Ensure sufficient statistics for meaningful percentile. Was 20
            percs.append( wpercentile(data=data.loc[msk, VAR].values, percents=100-EFF, weights=data.loc[msk, weight].values) )#wpercentile
            
        else:
            percs.append(0)

    print "Length of percs: ", len(percs), percs

    percs = percs[0:-1]
    bins_mjj = bins_mjj[0:-1]
    
    X = bins_mjj.reshape(-1,1)
    X = X[1:len(bins_mjj)]


    print len(X), len(percs)

    # Fit parameters
    knn_neighbors = 2
    knn_weights = 'uniform'
    fit_deg = 1

    knn = KNeighborsRegressor(n_neighbors=5, weights='distance') 
    y_knn = knn.fit(X, percs).predict(fineBinsRe)
    
    c = rp.canvas(batch=True)
    knnFit = c.plot(y_knn, bins=fineBins, linecolor=ROOT.kRed+2, linewidth=2, linestyle=1, label="knn fit, uniform", option='L')

    c.save('figures/distributions/percentile_test.pdf'.format(EFF, args.input))           

    outFile.cd()
    knnFit.SetName("kNNfit")
    knnFit.Write()
    outFile.Close()

    """
    coeff = np.polyfit(bins_mjj[1:], percs, deg=2)
    y_fit = []

    print coeff

    for i in range(len(bins_mjj)):
        #y_fit.append(coeff[0]*bins_mjj[i] + coeff[1])
        y_fit.append(coeff[0]*(bins_mjj[i]**2) + coeff[1]*bins_mjj[i] + coeff[2])

    # Plot kNN fit
    c = rp.canvas(batch=True)

    percPlot = c.plot(percs, bins=bins_mjj, markercolor=ROOT.kGreen+2, linecolor=ROOT.kGreen+2, markerstyle=20, label="data")

    knnFit = c.plot(y_knn, bins=bins_mjj, linecolor=ROOT.kRed+2, linewidth=2, linestyle=1, label="knn fit, uniform", option='L')

    c.ylabel("{} percentile".format(EFF))
    c.xlabel("m_{jj} [GeV]")

    c.save('figures/distributions/percentile_knnD2_{}_{}.pdf'.format(EFF, args.input))
    c.save('figures/distributions/percentile_knnD2_{}_{}.eps'.format(EFF, args.input))

    del c

    # Plot polynomial fit
    c = rp.canvas(batch=True)

    percPlot = c.plot(percs, bins=bins_mjj, markercolor=ROOT.kGreen+2, linecolor=ROOT.kGreen+2, markerstyle=20, label="data")
    knnFit = c.plot(y_fit, bins=bins_mjj, linecolor=ROOT.kRed+2, linewidth=2, linestyle=1, label="pol2 fit", option='L')

    c.ylabel("{} percentile".format(EFF))
    c.xlabel("m_{jj} [GeV]")

    c.save('figures/distributions/percentile_pol2_{}_{}.pdf'.format(EFF, args.input))
    c.save('figures/distributions/percentile_pol2_{}_{}.eps'.format(EFF, args.input))

    del c
    """

# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args(backend=True, plots=True)

    # Call main function
    main(args)
    pass
