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

# Project import(s)
from adversarial.utils import parse_args, initialise, load_data, latex, mkdir
from adversarial.profile import profile, Profile
from adversarial.constants import *
from .studies.common import *
from run.knn.common import add_knn, MODEL as sigModel, VAR as kNN_basevar, EFF as kNN_eff

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
    mc, features, _ = load_data('data/djr_LCTopo_2.h5') #, test=True) # 
    data, features, _ = load_data('data/djr_LCTopo_data.h5') #, test=True) # 

    histstyle[True] ['label'] = 'Multijets'
    histstyle[False]['label'] = 'Dark jets, Model A, m = 2 TeV'

    # Add knn variables

    #base_var = ['lead_jet_ungrtrk500', 'sub_jet_ungrtrk500']                         
    base_var = 'jet_ungrtrk500'
    kNN_var = base_var.replace('jet', 'knn')
    #base_vars = ['lead_'+base_var, 'sub_'+base_var]
    #kNN_vars = ['lead_'+kNN_var, 'sub_'+kNN_var]

    """
    with Profile("Add variables"):
        #for i in range(len(base_var)):                                               
        print "k-NN base variable: {} (cp. {})".format(base_var, kNN_var)
        add_knn(data, newfeat='lead_'+kNN_var, path='models/knn/knn_{}_{}_{}.pkl.gz'.format(base_var, kNN_eff, sigModel))
        add_knn(data, newfeat='sub_'+kNN_var, path='models/knn/knn_{}_{}_{}.pkl.gz'.format(base_var, kNN_eff, sigModel))
        add_knn(mc, newfeat='lead_'+kNN_var, path='models/knn/knn_{}_{}_{}.pkl.gz'.format(base_var, kNN_eff, sigModel))
        add_knn(mc, newfeat='sub_'+kNN_var, path='models/knn/knn_{}_{}_{}.pkl.gz'.format(base_var, kNN_eff, sigModel))
    """
        #add_knn(data, newfeat=kNN_var, path='models/knn/knn_{}_{}_{}.pkl.gz'.format(base_var, kNN_eff, sigModel))

    bins_pt = np.linspace(450, 5000, 50)

    # Useful masks
    msk_bkg_data  = data['signal'] == 0
    msk_bkg_mc  = (mc['signal'] == 0) #& (mc['weight']<0.0002)
    msk_sig_mc  = (mc['signal'] == 1) #& (mc['weight']<0.0002)

    msk_CR = (mc['lead_jet_ungrtrk500']<20) | (mc['sub_jet_ungrtrk500']<20)

    scale = 139*1000000 # (inverse nanobarn)

    # pT dist
    c = rp.canvas(batch=True)
    hist_incl_data = c.hist(data.loc[msk_bkg_data, 'jet_pt'].values, bins=bins_pt, weights=data.loc[msk_bkg_data, 'weight'].values, label="Data, control region", normalise=False, linecolor=ROOT.kGreen+2)

    hist_incl_mc = c.hist(mc.loc[msk_bkg_mc, 'sub_jet_pt'].values, bins=bins_pt, weights=scale*mc.loc[msk_bkg_mc, 'weight'].values, label="MC, scaled with lumi", normalise=False, linecolor=ROOT.kViolet+2)

    hist_incl_sig = c.hist(mc.loc[msk_sig_mc, 'sub_jet_pt'].values, bins=bins_pt, weights=mc.loc[msk_sig_mc, 'weight'].values, label="Combined Signal", normalise=False, linecolor=ROOT.kOrange+2)

    c.legend(width=0.4, xmin=0.5,  ymax=0.9)
    c.ylabel("Number of events")
    c.xlabel("Sub-leading jet pT [GeV]")
    c.logy()
    #c.ylim(0.00005, 5)
    #c.save('figures/distributions/mjj_Bkg_CR20.pdf'.format(knnCut))
    #c.save('figures/distributions/mjj_Bkg_CR20.eps'.format(knnCut))
    c.save('figures/distributions/sub_pt_bkg_data_mc.pdf')
    c.save('figures/distributions/sub_pt_bkg_data_mc.eps')

    print "Data bkg effective entries: ", hist_incl_data.GetEffectiveEntries()
    print "MC bkg effective entries: ", hist_incl_mc.GetEffectiveEntries()

    print "Data bkg integral: ", hist_incl_data.Integral()
    print "MC bkg integral: ", hist_incl_mc.Integral()

    del c

    c = rp.canvas(batch=True)
    hist_bkg_CR = c.hist(mc.loc[(msk_bkg_mc & msk_CR), 'lead_jet_pt'].values, bins=bins_pt, weights=scale*mc.loc[(msk_bkg_mc & msk_CR), 'weight'].values, label="MC, control region", normalise=False, linecolor=ROOT.kGreen+2)

    hist_sig_CR = c.hist(mc.loc[(msk_sig_mc & msk_CR), 'lead_jet_pt'].values, bins=bins_pt, weights=mc.loc[(msk_sig_mc & msk_CR), 'weight'].values, label="MC, control region", normalise=False, linecolor=ROOT.kGreen+2)

    
    print "CR sig contamination (eff. entries): ", hist_sig_CR.GetEffectiveEntries()/(hist_bkg_CR.GetEffectiveEntries()+hist_sig_CR.GetEffectiveEntries())
    print "CR sig contamination (integral): ", hist_sig_CR.Integral()/(hist_bkg_CR.Integral()+hist_sig_CR.Integral())

    print "CR sig efficiency (eff. entries): ", hist_sig_CR.GetEffectiveEntries()/hist_incl_sig.GetEffectiveEntries()
    print "CR sig efficiency (integral): ", hist_sig_CR.Integral()/hist_incl_sig.Integral()


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args(backend=True, plots=True)

    # Call main function
    main(args)
    pass
