#!/usr/bin/env python
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
    data, features, _ = load_data('data/' + args.input)

    histstyle[True] ['label'] = 'Multijets'
    histstyle[False]['label'] = 'Dark jets, Model A, m = 2 TeV'

    # Add knn variables

    #base_var = ['lead_jet_ungrtrk500', 'sub_jet_ungrtrk500']
    #kNN_var = [var.replace('jet', 'knn') for var in base_var]

    #base_var = ['ntrk_sum']
    #kNN_var = [var + '-knn' for var in base_var]

    """
    with Profile("Add variables"):
        from run.knn.common import add_knn, MODEL, VAR as kNN_basevar, EFF as kNN_eff
        print "k-NN base variable: {} (cp. {})".format(kNN_basevar, kNN_var)
        for i in range(len(base_var)):
            add_knn(data, newfeat=kNN_var[i], path='models/knn/knn_{}_{}_{}.pkl.gz'.format(base_var[i], kNN_eff, MODEL))
            print 'models/knn/knn_{}_{}_{}.pkl.gz'.format(base_var[i], kNN_eff, MODEL)
    """

    # Check variable distributions
    axes = {
        'jet_ungrtrk500': (50, 0, 100),
        #'lead_knn_ungrtrk500': (50, -100, 50),
        'jet_pt': (50, 0, 3000),
        'dijetmass': (50, 0, 7000),
        }

    scale = 139*1000000

    weight = 'weight'  # 'weight_test' / 'weight'
    msk_bkg  = data['signal'] == 0   # @TEMP signal
    msk_sig  = data['sigType'] == 1   # @TEMP signal
    #msk_weight = data['weight']<0.002
    #msk_bkg = msk_bkg & msk_weight

    #msk_CR = (data['lead_jet_ungrtrk500']<20) | (data['sub_jet_ungrtrk500']<20)


    ###### 3D histograms ####### 

    vary = 'jet_pt'
    varx = 'dijetmass'
    varz = 'jet_ungrtrk500'


    #for i,varx in enumerate(axisvars):                                                                                                                                                                   
    #   for vary in axisvars[i+1:]:                                                                                                                                                                        
    # Canvas                   
    can4 = rp.canvas(batch=True)
    pad = can4.pads()[0]._bare()
    pad.cd()
    pad.SetRightMargin(0.20)

    #can4 = ROOT.TCanvas("canvas", "", 800, 600)
    #can4.SetRightMargin(0.20)
    # Create, fill histogram                                                                                                                                                                               
    h2_bkg = ROOT.TH2F('{}_{}'.format(varx, vary), "", *(axes[varx] + axes[vary]))

    root_numpy.fill_hist(h2_bkg, data.loc[msk_bkg, [varx, vary]].values) #, scale*data.loc[msk_bkg, weight].values)#*data.loc[msk_bkg, varz].values)

    #h2_bkg.Scale(1./h2_bkg.Integral())

    print h2_bkg.Integral()

    # Draw                                                                                                                                                                                                 
    h2_bkg.Draw("COLZ")

    # Decorations                                                                                                                                                                                          
    h2_bkg.GetXaxis().SetTitle(latex(varx, ROOT=True))
    h2_bkg.GetYaxis().SetTitle(latex(vary, ROOT=True))
    #h2_bkg.GetZaxis().SetTitle(latex(varz, ROOT=True))
    #pad.SetLogz()
    #can4.zlim(0.0, 0.04)
    h2_bkg.GetZaxis().SetRangeUser(0.0, 300000)


    # Save                                                                                                                                                                                                 
    can4.save('figures/distributions/3d_{}_{}_{}_bkg.pdf'.format(varx, vary, varz))
    can4.save('figures/distributions/3d_{}_{}_{}_bkg.eps'.format(varx, vary, varz))



    # ntrk distribution
    """ 
    can1 = rp.canvas(batch=True)
    bins1 = np.linspace(0, 150, 75)

    h_ungrB = can1.hist(data.loc[msk_bkg, 'lead_jet_ungrtrk500'].values, bins=bins1, weights=data.loc[msk_bkg, weight].values, label='ungrtrk, bkg', normalise=True, linecolor=ROOT.kGreen+2)

    h_ungeS = can1.hist(data.loc[msk_sig, 'lead_jet_ungrtrk500'].values, bins=bins1, weights=data.loc[msk_sig, weight].values, label='ungrtrk, sig', normalise=True, linecolor=ROOT.kGreen+2, linestyle=2)
    
    can1.legend(width=0.3, xmin=0.6, ymax=0.9)
    can1.save('figures/distributions/ungrtrk_dist.pdf')
    can1.save('figures/distributions/ungrtrk_dist.eps')


    # 2D histograms

    axisvars = sorted(list(axes))
    
    varx = 'lead_jet_ungrtrk500'
    vary = 'sub_jet_ungrtrk500'


    #for i,varx in enumerate(axisvars):
    #   for vary in axisvars[i+1:]:
    # Canvas
    can3 = ROOT.TCanvas()
    can3.SetRightMargin(0.20)
    
    # Create, fill histogram
    h2_bkg = ROOT.TH2F('{}_{}'.format(varx, vary), "", *(axes[varx] + axes[vary]))
    h2_sig = ROOT.TH2F('{}_{}'.format(varx, vary), "", *(axes[varx] + axes[vary]))

    root_numpy.fill_hist(h2_bkg, data.loc[msk_bkg, [varx, vary]].values, data.loc[msk_bkg, weight].values)
    root_numpy.fill_hist(h2_sig, data.loc[msk_sig, [varx, vary]].values, data.loc[msk_sig, weight].values)
    
    # Draw
    h2_bkg.Draw("COLZ")

    # Decorations
    h2_bkg.GetXaxis().SetTitle(latex(varx, ROOT=True))
    h2_bkg.GetYaxis().SetTitle(latex(vary, ROOT=True))
    can3.SetLogz()
    
    # Save
    can3.SaveAs('figures/distributions/2d_{}_{}_bkg.pdf'.format(varx, vary))
    can3.SaveAs('figures/distributions/2d_{}_{}_bkg.eps'.format(varx, vary))

    can6 = ROOT.TCanvas()
    can6.SetRightMargin(0.20)

    h2_sig.Draw("COLZ")

    # Decorations
    h2_sig.GetXaxis().SetTitle(latex(varx, ROOT=True))
    h2_sig.GetYaxis().SetTitle(latex(vary, ROOT=True))
    can6.SetLogz()
    
    # Save
    can6.SaveAs('figures/distributions/2d_{}_{}_sig.pdf'.format(varx, vary))
    can6.SaveAs('figures/distributions/2d_{}_{}_sig.eps'.format(varx, vary))

    ### Subleading vs. leading knn_ntrk

    varx = 'lead_knn_ungrtrk500'
    vary = 'sub_knn_ungrtrk500'


    # Canvas
    can4 = ROOT.TCanvas()
    can4.SetRightMargin(0.20)

    h2_C1_bkg = ROOT.TH2F('{}_{}'.format(varx, vary), "", *(axes[varx] + axes[vary]))
    root_numpy.fill_hist(h2_C1_bkg, data.loc[msk_bkg, [varx, vary]].values, 100. * data.loc[msk_bkg, weight].values)
    h2_C1_sig = ROOT.TH2F('{}_{}'.format(varx, vary), "", *(axes[varx] + axes[vary]))
    root_numpy.fill_hist(h2_C1_sig, data.loc[msk_sig, [varx, vary]].values, 100. * data.loc[msk_sig, weight].values)

    # Draw
    h2_C1_bkg.Draw("COLZ")

    # Decorations
    h2_C1_bkg.GetXaxis().SetTitle(latex(varx, ROOT=True))
    h2_C1_bkg.GetYaxis().SetTitle(latex(vary, ROOT=True))
    can4.SetLogz()

    can4.SaveAs('figures/distributions/2d_{}_{}_bkg.pdf'.format(varx, vary))
    can4.SaveAs('figures/distributions/2d_{}_{}_bkg.eps'.format(varx, vary))


    # Canvas
    can5 = ROOT.TCanvas()
    can5.SetRightMargin(0.20)

    # Draw
    h2_C1_sig.Draw("COLZ")

    # Decorations
    h2_C1_sig.GetXaxis().SetTitle(latex(varx, ROOT=True))
    h2_C1_sig.GetYaxis().SetTitle(latex(vary, ROOT=True))
    can5.SetLogz()

    can5.SaveAs('figures/distributions/2d_{}_{}_sig.pdf'.format(varx, vary))
    can5.SaveAs('figures/distributions/2d_{}_{}_sig.eps'.format(varx, vary))

    """


    return


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args(backend=True, plots=True)

    # Call main function
    main(args)
    pass
