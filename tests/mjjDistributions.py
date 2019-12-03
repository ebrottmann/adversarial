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
import h5py

# Project import(s)
from adversarial.utils import parse_args, initialise, load_data, latex, mkdir
from adversarial.profile import profile, Profile
from adversarial.constants import *
from .studies.common import *
from run.knn.common import add_knn, FIT, MODEL as sigModel, VAR as kNN_basevar, EFF as kNN_eff

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
    #data = np.zeros(1, 95213009, 10)
    data, features, _ = load_data('data/djr_LCTopo_2.h5') # + args.input) #, test=True) # 
    #data2, features, _ = load_data('data/djr_LCTopo_2.h5') # + args.input) #, test=True) # 
    #data = np.concatenate((data1, data2))

    #f1 = h5py.File('data/djr_LCTopo_1.h5', 'r')
    #f2 = h5py.File('data/djr_LCTopo_2.h5', 'r')

    knnCut = 0
    ntrkCut = 50
    emfracCut = 0.65
    scale = 139*1000000 # (inverse nanobarn)
    signal_to_plot = 7

    sigDict = {
        0: 'All Models',
        1: 'Model A, m = 2 TeV',
        2: 'Model A, m = 1 TeV',
        3: 'Model A, m = 1.5 TeV',
        4: 'Model A, m = 2.5 TeV',
        5: 'Model B, m = 1 TeV',
        6: 'Model B, m = 1.5 TeV',
        7: 'Model B, m = 2 TeV',
        8: 'Model B, m = 2.5 TeV',
        9: 'Model C, m = 1 TeV',
        10: 'Model C, m = 1.5 TeV',
        11: 'Model C, m = 2 TeV',
        12: 'Model C, m = 2.5 TeV',
        13: 'Model D, m = 1 TeV',
        14: 'Model D, m = 1.5 TeV',
        15: 'Model D, m = 2 TeV',
        16: 'Model D, m = 2.5 TeV',
        }


    outHistFile = ROOT.TFile.Open("figures/mjjHistograms_kNN{}_eff{}.root".format(knnCut, kNN_eff),"RECREATE")

    histstyle[True] ['label'] = 'Multijets'
    histstyle[False]['label'] = 'Dark jets, {}'.format(sigDict[signal_to_plot]) 

    # Add knn variables

    #base_var = ['lead_jet_ungrtrk500', 'sub_jet_ungrtrk500']                         
    base_var = 'jet_ungrtrk500'
    kNN_var = base_var.replace('jet', 'knn')
    #base_vars = ['lead_'+base_var, 'sub_'+base_var]
    #kNN_vars = ['lead_'+kNN_var, 'sub_'+kNN_var]

    print data.shape

    with Profile("Add variables"):
        #for i in range(len(base_var)):                                               
        print "k-NN base variable: {} (cp. {})".format(base_var, kNN_var)
        add_knn(data, newfeat='lead_'+kNN_var, path='models/knn/{}_{}_{}_{}.pkl.gz'.format(FIT, base_var, kNN_eff, sigModel))
        add_knn(data, newfeat='sub_'+kNN_var, path='models/knn/{}_{}_{}_{}.pkl.gz'.format(FIT, base_var, kNN_eff, sigModel))

        #add_knn(data, newfeat=kNN_var, path='models/knn/knn_{}_{}_{}.pkl.gz'.format(base_var, kNN_eff, sigModel))

        print 'models/knn/knn_{}_{}_{}.pkl.gz'.format(base_var, kNN_eff, sigModel)

        """
        base_var = ['lead_jet_ungrtrk500', 'sub_jet_ungrtrk500']
        kNN_var = [var.replace('jet', 'knn') for var in base_var]
        
        with Profile("Add variables"):
        from run.knn.common import add_knn, MODEL, VAR as kNN_basevar, EFF as kNN_eff
        print "k-NN base variable: {} (cp. {})".format(kNN_basevar, kNN_var)
        for i in range(len(base_var)):
        add_knn(data, newfeat=kNN_var[i], path='models/knn/knn_{}_{}_{}.pkl.gz'.format(base_var[i], kNN_eff, MODEL))
        print 'models/knn/knn_{}_{}_{}.pkl.gz'.format(base_var[i], kNN_eff, MODEL)
        """
        
    weight = 'weight'  # 'weight_test' / 'weight'
    bins_pt = np.linspace(450, 3500, 40)
    bins_mjj = np.linspace(0, 8000, 80)

    # Useful masks
    msk_bkg  = data['signal'] == 0
    if signal_to_plot == 0:
        msk_sig  = data['signal'] == 1
    else:
        msk_sig  = data['sigType'] == signal_to_plot

    #msk_weight = data['weight']<0.2

           
    msk_knn = (data['lead_knn_ungrtrk500']>knnCut) & (data['sub_knn_ungrtrk500']>knnCut) 
    msk_ungr = (data['lead_jet_ungrtrk500']>ntrkCut) & (data['sub_jet_ungrtrk500']>ntrkCut) 
    msk_emfrac = (data['lead_jet_EMFrac']<emfracCut) & (data['sub_jet_EMFrac']<emfracCut) 

    msk_knn_1 = (data['lead_knn_ungrtrk500']>knnCut)
    msk_ungr_1 = (data['lead_jet_ungrtrk500']>ntrkCut)

    #msk_knn = (data['knn_ungrtrk500']>knnCut)
    #msk_ungr = (data['jet_ungrtrk500']>90.0)

    msk_ntrkBkg = msk_ungr & msk_emfrac & msk_bkg #& msk_weight #& msk_pt & msk_m & msk_eta
    msk_ntrkSig = msk_ungr & msk_emfrac & msk_sig  #& msk_pt & msk_m & msk_eta

    msk_knnBkg = msk_knn & msk_bkg
    msk_knnSig = msk_knn & msk_sig

    msk_ntrkBkg1 = msk_ungr_1 & msk_bkg #& msk_weight #& msk_pt & msk_m & msk_eta
    msk_ntrkSig1 = msk_ungr_1 & msk_sig  #& msk_pt & msk_m & msk_eta
    msk_knnBkg1 = msk_knn_1 & msk_bkg #& msk_weight #& msk_pt & msk_m & msk_eta
    msk_knnSig1 = msk_knn_1 & msk_sig  #& msk_pt & msk_m & msk_eta

    msk_inclBkg = msk_bkg #& msk_weight #& msk_pt & msk_m & msk_eta 
    msk_inclSig = msk_sig #& msk_pt & msk_m & msk_eta 


    # Mjj dist with cut on ntrk, ungrtrk compared to inclusive selection
    c = rp.canvas(batch=True)
    hist_inclBkg = c.hist(data.loc[msk_inclBkg, 'dijetmass'].values, bins=bins_mjj, weights=scale*data.loc[msk_inclBkg, weight].values, label="Multijets, Inclusive", normalise=True, linecolor=ROOT.kGreen+2, linewidth=3)
    hist_knnBkg = c.hist(data.loc[msk_knnBkg, 'dijetmass'].values, bins=bins_mjj, weights=scale*data.loc[msk_knnBkg, weight].values, label="Multijets, n_{{trk}}^{{#epsilon}}>{}".format(knnCut), normalise=True, linecolor=ROOT.kMagenta+2, linestyle=2, linewidth=3)

    hist_ntrkBkg = c.hist(data.loc[msk_ntrkBkg, 'dijetmass'].values, bins=bins_mjj, weights=scale*data.loc[msk_ntrkBkg, weight].values, label="Multijets, n_{{trk}}>{}".format(ntrkCut), normalise=True, linecolor=ROOT.kOrange+2, linestyle=2, linewidth=3)
    #hist_CRBkg = c.hist(data.loc[msk_CR_bkg, 'dijetmass'].values, bins=bins_mjj, weights=scale*data.loc[msk_CR_bkg, weight].values, label="CR Bkg, C<20", normalise=True, linecolor=ROOT.kGray+2, linestyle=2)

    c.legend(width=0.4, xmin=0.5,  ymax=0.9)
    c.ylabel("Fraction of jets")
    c.xlabel("m_{jj} [GeV]")
    c.logy()
    #c.ylim(0.00005, 5)
    #c.save('figures/distributions/mjj_Bkg_CR20.pdf'.format(knnCut))
    #c.save('figures/distributions/mjj_Bkg_CR20.eps'.format(knnCut))
    c.save('figures/distributions/mjj_BkgDist_ntrk{}_knn{}_{}.pdf'.format(ntrkCut,knnCut, FIT))
    c.save('figures/distributions/mjj_BkgDist_ntrk{}_knn{}_{}.eps'.format(ntrkCut,knnCut, FIT))

    del c


    c = rp.canvas(batch=True)
    hist_Sig = c.hist(data.loc[msk_sig, 'dijetmass'].values, bins=bins_mjj, weights=data.loc[msk_sig, weight].values, label="Model A, m = 2 TeV, inclusive", normalise=True, linecolor=ROOT.kGreen+2)

    hist_knnSig = c.hist(data.loc[msk_knnSig, 'dijetmass'].values, bins=bins_mjj, weights=data.loc[msk_knnSig, weight].values, label="Model A, m = 2 TeV, #it{{n}}_{{trk}}^{{#epsilon}}>{}".format(knnCut), normalise=True, linecolor=ROOT.kMagenta+2, linestyle=2)

    hist_ntrkSig = c.hist(data.loc[msk_ntrkSig, 'dijetmass'].values, bins=bins_mjj, weights=data.loc[msk_ntrkSig, weight].values, label="Model A, m = 2 TeV, #it{{n}}_{{trk}}>{}".format(ntrkCut), normalise=True, linecolor=ROOT.kOrange+2, linestyle=2)

    #hist_CRSig = c.hist(data.loc[msk_CR_sig, 'dijetmass'].values, bins=bins_mjj, weights=data.loc[msk_CR_sig, weight].values, label="Sig, CR", normalise=True, linecolor=ROOT.kGray+2, linestyle=2)

    c.legend(width=0.4, xmin=0.5,  ymax=0.9)
    c.ylabel("Fraction of jets")
    c.xlabel("m_{jj} [GeV]")
    c.logy()
    #c.ylim(0.00005, 5)
    c.save('figures/distributions/mjj_SigDist_ntrk{}_knn{}_{}.pdf'.format(ntrkCut,knnCut, FIT))
    c.save('figures/distributions/mjj_SigDist_ntrk{}_knn{}_{}.eps'.format(ntrkCut,knnCut, FIT))


    del c

    c = rp.canvas(batch=True)

    hist_knnSig = c.hist(data.loc[msk_knnSig, 'dijetmass'].values, bins=bins_mjj, weights=data.loc[msk_knnSig, weight].values, label="Model A, m = 2 TeV, knn_ntrk>{}".format(knnCut), normalise=False, linecolor=ROOT.kBlue+1, linestyle=1)

    hist_knnBkg = c.hist(data.loc[msk_knnBkg, 'dijetmass'].values, bins=bins_mjj, weights=scale*data.loc[msk_knnBkg, weight].values, label="Multijets, knn_ntrk>{}".format(knnCut), normalise=False, linecolor=ROOT.kMagenta+2, linestyle=2)

    hist_ntrkBkg = c.hist(data.loc[msk_ntrkBkg, 'dijetmass'].values, bins=bins_mjj, weights=scale*data.loc[msk_ntrkBkg, weight].values, label="Multijets, ntrk>{}".format(ntrkCut), normalise=False, linecolor=ROOT.kOrange+2, linestyle=2)

    c.legend(width=0.4, xmin=0.3,  ymax=0.9)
    c.ylabel("Number of events")
    c.xlabel("m_{jj} [GeV]")
    c.logy()
    #c.ylim(0.00005, 5)
    c.save('figures/distributions/mjj_Dist_noNorm_knn{}_{}.pdf'.format(knnCut, FIT))
    c.save('figures/distributions/mjj_Dist_noNorm_knn{}_{}.eps'.format(knnCut, FIT))

    bins_mjj = np.linspace(0,10000, 50)

# Unscaled histograms for calculating efficiencies

    hist_inclBkg = c.hist(data.loc[msk_inclBkg, 'dijetmass'].values, bins=bins_mjj, weights=scale*data.loc[msk_inclBkg, weight].values, normalise=False)

    hist_inclSig = c.hist(data.loc[msk_inclSig, 'dijetmass'].values, bins=bins_mjj, weights=data.loc[msk_inclSig, weight].values, normalise=False)

    hist_ntrkSig = c.hist(data.loc[msk_ntrkSig, 'dijetmass'].values, bins=bins_mjj, weights=data.loc[msk_ntrkSig, weight].values, normalise=False)

    hist_knnSig = c.hist(data.loc[msk_knnSig, 'dijetmass'].values, bins=bins_mjj, weights=data.loc[msk_knnSig, weight].values, normalise=False)

    hist_ntrkSig1 = c.hist(data.loc[msk_ntrkSig1, 'dijetmass'].values, bins=bins_mjj, weights=data.loc[msk_ntrkSig1, weight].values, normalise=False)

    hist_ntrkBkg1 = c.hist(data.loc[msk_ntrkBkg1, 'dijetmass'].values, bins=bins_mjj, weights=data.loc[msk_ntrkBkg1, weight].values, normalise=False)

    hist_knnBkg1 = c.hist(data.loc[msk_knnBkg1, 'dijetmass'].values, bins=bins_mjj, weights=data.loc[msk_knnBkg1, weight].values, normalise=False)

    hist_knnSig1 = c.hist(data.loc[msk_knnSig1, 'dijetmass'].values, bins=bins_mjj, weights=data.loc[msk_knnSig1, weight].values, normalise=False)


    print "Bkg inclusive integral: ", hist_inclBkg.GetEffectiveEntries() 
    print "Sig inclusive integral: ", hist_inclSig.GetEffectiveEntries() 

    print "Bkg pass kNN eff entries / integral: ", hist_knnBkg.GetEffectiveEntries(), hist_knnBkg.Integral()
    print "Sig pass kNN eff entries / integral: ", hist_knnSig.GetEffectiveEntries(), hist_knnSig.Integral()

    print "Bkg pass ntrk eff entries / integral: ", hist_ntrkBkg.GetEffectiveEntries(), hist_ntrkBkg.Integral()
    print "Sig pass ntrk eff entries / integral: ", hist_ntrkSig.GetEffectiveEntries(), hist_ntrkSig.Integral()

    print "Bkg Eff. knn_ntrk> {}, eff. entries: ".format(knnCut), 100*hist_knnBkg.GetEffectiveEntries()/hist_inclBkg.GetEffectiveEntries()
    print "Sig Eff. knn_ntrk> {}, eff. entries: ".format(knnCut), 100*hist_knnSig.GetEffectiveEntries()/hist_inclSig.GetEffectiveEntries() 

    print "Bkg Eff. knn_ntrk> {}, integral: ".format(knnCut), 100*hist_knnBkg.Integral()/hist_inclBkg.Integral()
    print "Sig Eff. knn_ntrk> {}, integral: ".format(knnCut), 100*hist_knnSig.Integral()/hist_inclSig.Integral() 

    print "Bkg Eff. ntrk>{}, eff. entries: ".format(ntrkCut), 100*hist_ntrkBkg.GetEffectiveEntries()/hist_inclBkg.GetEffectiveEntries()
    print "Sig Eff. ntrk>{}, eff. entries: ".format(ntrkCut), 100*hist_ntrkSig.GetEffectiveEntries()/hist_inclSig.GetEffectiveEntries() #, hist_ntrkSig.GetEffectiveEntries()


    print "Bkg Eff. 1 jet knn_ntrk> {}, eff. entries: ".format(knnCut), 100*hist_knnBkg1.GetEffectiveEntries()/hist_inclBkg.GetEffectiveEntries()
    print "Sig Eff. 1 jet knn_ntrk> {}, eff. entries: ".format(knnCut), 100*hist_knnSig1.GetEffectiveEntries()/hist_inclSig.GetEffectiveEntries() 

    print "Bkg Eff. 1 jet knn_ntrk> {}, integral: ".format(knnCut), 100*hist_knnBkg1.GetEffectiveEntries()/hist_inclBkg.GetEffectiveEntries()
    print "Sig Eff. 1 jet knn_ntrk> {}, integral: ".format(knnCut), 100*hist_knnSig1.GetEffectiveEntries()/hist_inclSig.GetEffectiveEntries() 



    outHistFile.cd()
    hist_knnBkg.SetName("bkg_knn")
    hist_knnSig.SetName("sig_knn")
    hist_knnBkg.Write()
    hist_knnSig.Write()
    outHistFile.Close()
    # Mjj dist for CR compared to inclusive selection

    """
    c_CR = rp.canvas(batch=True)

    hist_incl = c_CR.hist(data.loc[msk_incl, 'dijetmass'].values, bins=bins_mjj, weight=data.loc[msk_incl, weight].values, label="Inclusive", normalise=True, linecolor=ROOT.kGreen+2)
    hist_CR = c_CR.hist(data.loc[msk_inclCR, 'dijetmass'].values, bins=bins_mjj, weight=data.loc[msk_inclCR, weight].values, label="CR", normalise=True, linecolor=ROOT.kBlue+2)
    hist_pass4 = c_CR.hist(data.loc[msk_pass4, 'dijetmass'].values, bins=bins_mjj, weight=data.loc[msk_pass4, weight].values, label="CR + ntrk_knn>-10", normalise=True, linecolor=ROOT.kMagenta+2)
    #hist_pass5 = c_CR.hist(data.loc[msk_pass5, 'dijetmass'].values, bins=bins_mjj, weight=data.loc[msk_pass5, weight].values, label="CR + ntrk>100", normalise=True, linecolor=ROOT.kOrange+2)

    c_CR.legend(width=0.4, xmin=0.3, ymax=0.9)
    c_CR.ylabel("Fraction of jets")
    c_CR.xlabel("m_{jj} [GeV]")
    c_CR.logy()
    c_CR.save('figures/distributions/mjj_dist_CR.pdf')
    c_CR.save('figures/distributions/mjj_dist_CR.eps')

    hist_CR = c_CR.hist(data.loc[msk_pass3, 'dijetmass'].values, bins=bins_mjj, weight=data.loc[msk_pass3, weight].values, label="CR", normalise=False, linecolor=ROOT.kBlue+2)
    hist_pass4 = c_CR.hist(data.loc[msk_pass4, 'dijetmass'].values, bins=bins_mjj, weight=data.loc[msk_pass4, weight].values, normalise=False)
    #hist_pass5 = c_CR.hist(data.loc[msk_pass5, 'dijetmass'].values, bins=bins_mjj, weight=data.loc[msk_pass5, weight].values, normalise=False)

    #print "Eff. ntrk > 100 (CR): ", hist_pass5.GetEffectiveEntries()/hist_CR.GetEffectiveEntries() 
    print "Eff. knn_ntrk > -10 (CR): ", hist_pass4.GetEffectiveEntries()/hist_CR.GetEffectiveEntries() 


    # Mjj dist for CR compared to inclusive selection
    c_CR2 = rp.canvas(batch=True)

    hist_incl = c_CR2.hist(data.loc[msk_incl, 'dijetmass'].values, bins=bins_mjj, weight=data.loc[msk_incl, weight].values, label="Inclusive", normalise=True, linecolor=ROOT.kGreen+2)
    #hist_CR2 = c_CR2.hist(data.loc[msk_inclCR, 'dijetmass'].values, bins=bins_mjj, weight=data.loc[msk_inclCR, weight].values, label="CR", normalise=True, linecolor=ROOT.kBlue+2)
    #hist_pass6 = c_CR2.hist(data.loc[msk_pass6, 'dijetmass'].values, bins=bins_mjj, weight=data.loc[msk_pass6, weight].values, label="CR + ntrk_knn>-10", normalise=True, linecolor=ROOT.kMagenta+2)
    #hist_pass7 = c_CR2.hist(data.loc[msk_pass7, 'dijetmass'].values, bins=bins_mjj, weight=data.loc[msk_pass7, weight].values, label="CR + ntrk>100", normalise=True, linecolor=ROOT.kOrange+2)

    c_CR2.legend(width=0.4, xmin=0.3, ymax=0.9)
    c_CR2.ylabel("Fraction of jets")
    c_CR2.xlabel("m_{jj} [GeV]")
    c_CR2.logy()
    c_CR2.save('figures/distributions/mjj_dist_CR2.pdf')
    c_CR2.save('figures/distributions/mjj_dist_CR2.eps')

    hist_CR2 = c_CR2.hist(data.loc[msk_pass3, 'dijetmass'].values, bins=bins_mjj, weight=data.loc[msk_pass3, weight].values, label="CR_C1", normalise=False, linecolor=ROOT.kBlue+2)
    #hist_pass6 = c_CR2.hist(data.loc[msk_pass6, 'dijetmass'].values, bins=bins_mjj, weight=data.loc[msk_pass6, weight].values, normalise=False)
    #hist_pass7 = c_CR2.hist(data.loc[msk_pass7, 'dijetmass'].values, bins=bins_mjj, weight=data.loc[msk_pass7, weight].values, normalise=False)
    """
    #print "Eff. ntrk > 100 (CR2): ", hist_pass7.GetEffectiveEntries()/hist_CR2.GetEffectiveEntries() 
    #print "Eff. knn_ntrk > -10 (CR2): ", hist_pass6.GetEffectiveEntries()/hist_CR2.GetEffectiveEntries() 

# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args(backend=True, plots=True)

    # Call main function
    main(args)
    pass



