#!/bin/env python
# -*- coding: utf-8 -*-

# Basic import(s)
# ...

#Produce plot of very basic sensitivity estimate


# Get ROOT to stop hogging the command-line options
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import root_numpy
from array import array
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
    data, features, _ = load_data('data/djr_LCTopo_1.h5') #, test=True)
    #data2, features, _ = load_data('data/djr_LCTopo_2.h5') #, test=True)

    #data = np.concatenate((data1, data2))

    sigNumber = 0

    sigDict = {
        0: 'All Models',
        1: 'Model A, m = 1 TeV',
        2: 'Model A, m = 1.5 TeV',
        3: 'Model A, m = 2 TeV',
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

    outFile = ROOT.TFile.Open("figures/sensitivity_targetEff{}.root".format(kNN_eff),"RECREATE")

    histstyle[True] ['label'] = 'Multijets'
    histstyle[False]['label'] = 'Dark jets, {}'.format(sigDict[sigNumber])

    # Add knn variables

    #base_var = ['lead_jet_ungrtrk500', 'sub_jet_ungrtrk500']
    base_var = 'jet_ungrtrk500'
    kNN_var = base_var.replace('jet', 'knn')
    #base_vars = [base_var]
    #kNN_vars = [kNN_var]
    base_vars = ['lead_'+base_var, 'sub_'+base_var]
    kNN_vars = ['lead_'+kNN_var, 'sub_'+kNN_var]

    
    with Profile("Add variables"):
        #for i in range(len(base_var)):
        print "k-NN base variable: {} (cp. {})".format(base_var, kNN_var)
        add_knn(data, newfeat='lead_'+kNN_var, path='models/knn/knn1D_{}_{}_{}.pkl.gz'.format(base_var, kNN_eff, sigModel))
        add_knn(data, newfeat='sub_'+kNN_var, path='models/knn/knn1D_{}_{}_{}.pkl.gz'.format(base_var, kNN_eff, sigModel))

        #add_knn(data, newfeat=kNN_var, path='models/knn/knn_{}_{}_{}.pkl.gz'.format(base_var, kNN_eff, sigModel))
        print 'models/knn/knn_{}_{}_{}.pkl.gz'.format(base_var, kNN_eff, sigModel)

    # Check variable distributions
        
    weight = 'weight'  # 'weight_test' / 'weight'
    scale = 139*1000000 # (inverse nanobarn)

    msk_bkg = data['signal'] == 0
    if sigNumber==0:
        msk_sig = data['signal'] == 1 
    else:
        msk_sig = data['sigType'] == sigNumber 


    knnBins = np.linspace(-100, 200, 75, endpoint=True)
    effBins = np.linspace(0,1,100, endpoint=True)

    for var in kNN_vars:
        ### Canvas ###
        c = rp.canvas(num_pads=2, batch=True)
        c_tmp = rp.canvas(num_pads=1, batch=True)

        ### Plot ###
        h2 = c.pads()[0].hist(data.loc[msk_sig, var].values, bins=knnBins, weights=data.loc[msk_sig, weight].values, normalise=True, **histstyle[False])
        h1 = c.pads()[0].hist(data.loc[msk_bkg, var].values, bins=knnBins, weights=scale*data.loc[msk_bkg, weight].values, normalise=True, **histstyle[True])

        h1_incl = c_tmp.hist(data.loc[msk_bkg, var].values, bins=knnBins, weights=scale*data.loc[msk_bkg, weight].values, normalise=False)
        h2_incl = c_tmp.hist(data.loc[msk_sig, var].values, bins=knnBins, weights=data.loc[msk_sig, weight].values, normalise=False)

        #h1_CR = c_tmp.hist(data.loc[msk_CR_bkg, var].values, bins=knnBins, weights=scale*data.loc[msk_CR_bkg, weight].values, normalise=False)
        #h2_CR = c_tmp.hist(data.loc[msk_CR_sig, var].values, bins=knnBins, weights=data.loc[msk_CR_sig, weight].values, normalise=False)

        print "bkg. incl integral: ", h1_incl.GetEffectiveEntries()
        print "sig. incl integral: ", h2_incl.GetEffectiveEntries()
        #print "bkg. CR efficiency: ", h1_CR.GetEffectiveEntries()/h1_incl.GetEffectiveEntries()
        #print "sig. CR efficiency: ", h2_CR.GetEffectiveEntries()/h2_incl.GetEffectiveEntries()

        normFactor = 1.0 / (3./2 + np.sqrt(h1_incl.GetEffectiveEntries()) )
        print "Sensitivity with no cut: ", normFactor

        ### sensitivity ###
        sensitivity, bkg_eff_1jet = array( 'd' ), array( 'd' )
        #sensitivity = []
        #bkg_eff_1jet = []
        i = 0
        for cut in knnBins:

            msk_pass = (data[kNN_vars[0]]>cut) & (data[kNN_vars[1]]>cut)
            msk_pass1 = data[var]>cut
            #msk_pass = (data[var]>cut)
            msk_bkg_pass = msk_bkg & msk_pass
            msk_sig_pass = msk_sig & msk_pass

            msk_bkg_pass1 = msk_bkg & msk_pass1
            msk_sig_pass1 = msk_sig & msk_pass1

            h1_pass = c_tmp.hist(data.loc[msk_bkg_pass, var].values, bins=knnBins, weights=scale*data.loc[msk_bkg_pass, weight].values, normalise=False)
            h2_pass = c_tmp.hist(data.loc[msk_sig_pass, var].values, bins=knnBins, weights=data.loc[msk_sig_pass, weight].values, normalise=False)

            h1_pass1 = c_tmp.hist(data.loc[msk_bkg_pass1, var].values, bins=knnBins, weights=data.loc[msk_bkg_pass1, weight].values, normalise=False)

            if ( h2_incl.GetEffectiveEntries()>0 ) : #and h1_pass.GetEffectiveEntries()>0) :
                sensitivity.append( ((h2_pass.GetEffectiveEntries()/h2_incl.GetEffectiveEntries()) / (3./2 + np.sqrt(h1_pass.GetEffectiveEntries()) )) / normFactor )

                #print "bkg. eff. @ " , cut, ": ", h1_pass.GetEffectiveEntries()/h1_incl.GetEffectiveEntries()  
                #print "signal eff. @ ", cut, ": ", h2_pass.GetEffectiveEntries()/h2_incl.GetEffectiveEntries()
                #print "Sensitivity gain@ ", cut, ": ", ((h2_pass.GetEffectiveEntries()/h2_incl.GetEffectiveEntries()) / (3./2 + np.sqrt(h1_pass.GetEffectiveEntries())) ) / normFactor

            else: 
                sensitivity.append(0)

            if (h1_incl.GetEffectiveEntries()>0 ) :
                bkg_eff_1jet.append(h1_pass1.GetEffectiveEntries()/h1_incl.GetEffectiveEntries())
            else:
                bkg_eff_1jet.append(0)
                
            i = i+1

        #c.pads()[0].ylim(0,0.25)
        c.pads()[0].logy()
        c.pads()[0].xlim(-100,200)
        c.pads()[1].ylim(0,30)
        c.pads()[1].xlim(-100,200)
        c.pads()[1].graph( sensitivity, bins=knnBins) #, oob=False )

        ### Decorations ###
        c.legend(width=0.4, xmin=0.3, ymax=0.9)
        #c.xlabel("n_{trk}^{#epsilon={}\%}".format(kNN_eff)) #latex(var, ROOT=True))
        c.xlabel("n_{trk}^{#epsilon}") #latex(var, ROOT=True))
        c.ylabel("Fraction of jets")
        c.pads()[1].ylabel("Sensitivity gain")#"#epsilon_{S}/(#frac{3}{2} + #sqrt{B})/")
        c.pads()[1].text(["Sensitivity = #varepsilon_{S}/(#frac{3}{2} + #sqrt{B})", 
                ], xmin=0.2, ymax=0.80, ATLAS=False)

        c.save('figures/distributions/sensitivity_{}_sig{}_eff{}.pdf'.format(var, sigNumber, kNN_eff))
        c.save('figures/distributions/sensitivity_{}_sig{}_eff{}.eps'.format(var, sigNumber, kNN_eff))

        del c

        gr_sen = ROOT.TGraph(len(sensitivity), knnBins, sensitivity)
        gr_eff = ROOT.TGraph(len(bkg_eff_1jet), knnBins, bkg_eff_1jet)

        gr_more = ROOT.TGraph(len(sensitivity), bkg_eff_1jet, sensitivity)

        gr_sen.GetXaxis().SetTitle("#it{n}_{trk}^{#epsilon}-cut")
        gr_sen.GetYaxis().SetTitle("Sensitivity gain")
        gr_eff.GetYaxis().SetTitle("Single jet #varepsilon_{B}")
        gr_sen.GetYaxis().SetAxisColor(ROOT.kOrange+2)
        gr_eff.GetYaxis().SetAxisColor(ROOT.kGreen+2)
        gr_sen.SetMarkerColor(ROOT.kOrange+2)
        gr_eff.SetMarkerColor(ROOT.kGreen+2)
        gr_eff.SetDrawOption("Y+")

        c2 = rp.canvas(batch=True)
        c2.pads()[0].logx()
        c2.pads()[0].cd()
        #c2.pads()[0].graph(sensitivity, bkg_eff_1jet)
        gr_more.GetXaxis().SetTitle("Single jet #varepsilon_{B}")
        gr_more.GetYaxis().SetTitle("Sensitivity gain")
        #gr_more.GetXaxis().SetRangeUser(0, 0.02)
        gr_more.Draw("AP")


        #c2 = ROOT.TCanvas("can2", "", 200,10,700,500) #(batch=True)
        #pad1 = ROOT.TPad("pad1", "", 0,0,1,1) #c2.pads()[0]._bare()
        #pad1.Draw()
        #pad1.cd()
        #gr_sen.Draw("AP")
        

        #c2.cd()
        #pad2 = ROOT.TPad("pad2", "", 0,0,1,1) #c2.pads()[0]._bare()
        #pad2.SetFillStyle(4000)
        #pad2.Draw()
        #pad2.cd()
        #gr_eff.Draw("PY+")

        #gr_eff.Draw("APY+")
        #gr_sen.Draw("SAME")

        #gr_sen = c2.graph(sensitivity, bins=knnBins, markercolor=ROOT.kOrange+2)
        #gr_eff = c2.graph(bkg_eff_1jet, bins=knnBins, markercolor=ROOT.kGreen+2, option='Y+' )
        #gr_eff.GetYaxis.SetRange(0,1)
        #gr_eff.Draw("SAME Y+")
        #c2.xlabel("Single jet #varepsilon_{B}")
        #c2.ylabel("Sensitivity gain")
        #c2.text(["#epsilon=0.5 %",], xmin=0.2, ymax=0.8, ATLAS=False)

        ### Save ###
        #mkdir('figures/distributions')

        c2.save('figures/distributions/sensitivity_{}_eff{}_1jet.pdf'.format(var,kNN_eff) )
        del c2

        outFile.cd()
        gr_more.SetName("sensitivity_eff{}".format(kNN_eff))
        gr_more.Write()
        outFile.Close()

        #print 'figures/distributions/sensitivity_{}_sig{}_eff{}.pdf'.format(var, sigNumber, kNN_eff)
        pass
    

    # Plot also the normal ntrk distribution for cross check with Roland's result
    """
    msk_bkg = data['signal'] == 0
    if sigNumber==0:
        msk_sig = data['signal'] == 1 # data['sigType'] == sigNumber #                             
    else:
        msk_sig = data['sigType'] == sigNumber # data['sigType'] == sigNumber #                    
    #msk_weight = data['weight']<0.0002
    #msk_bkg = msk_bkg & msk_pt & msk_m & msk_eta 
    #msk_sig = msk_sig & msk_pt & msk_m & msk_eta 


    baseBins = np.linspace(0, 200, 75, endpoint=True) #axes[var][1], axes[var][2], axes[var][0] + 1, endpoint=True)

    for var in base_vars:
        ### Canvas ###
        c = rp.canvas(num_pads=2, batch=True)
        c.pads()[0].logy()

        c_tmp = rp.canvas(batch=True)

        ### Plot ###
        h2 = c.pads()[0].hist(data.loc[msk_sig, var].values, bins=baseBins, weights=data.loc[msk_sig, weight].values, normalise=True, **histstyle[False])
        h1 = c.pads()[0].hist(data.loc[msk_bkg, var].values, bins=baseBins, weights=scale*data.loc[msk_bkg, weight].values, normalise=True, **histstyle[True])

        h1_incl = c_tmp.hist(data.loc[msk_bkg, var].values, bins=baseBins, weights=scale*data.loc[msk_bkg, weight].values, normalise=False)
        h2_incl = c_tmp.hist(data.loc[msk_sig, var].values, bins=baseBins, weights=data.loc[msk_sig, weight].values, normalise=False)


        print "bkg. incl integral: ", h1_incl.GetEffectiveEntries()
        print "sig. incl integral: ", h2_incl.GetEffectiveEntries()

        normFactor = 1.0 / (3./2 + np.sqrt(h1_incl.Integral()) )

        #print "Sensitivity with no cut: ", normFactor


        ### sensitivity ###
        sensitivity = []
        i = 0
        for cut in baseBins:
            #print cut

            msk_pass = (data[base_vars[0]]>cut) & (data[base_vars[1]]>cut) #
            #msk_pass = data[var]>cut

            msk_bkg_pass = msk_bkg & msk_pass
            msk_sig_pass = msk_sig & msk_pass
            
            h1_pass = c_tmp.hist(data.loc[msk_bkg_pass, var].values, bins=baseBins, weights=scale*data.loc[msk_bkg_pass, weight].values, normalise=False)
            h2_pass = c_tmp.hist(data.loc[msk_sig_pass, var].values, bins=baseBins, weights=data.loc[msk_sig_pass, weight].values, normalise=False)


            if ( h2_incl.Integral()>0 ): #and h1_pass.Integral()>0 ):
                sensitivity.append( (h2_pass.Integral()/h2_incl.Integral()) /  (3./2. + np.sqrt(h1_pass.Integral())) / normFactor )

                #print "signal eff.  at ", cut, ": ", (h2_pass.Integral()/h2_incl.Integral()) 
                #print "bkg eff.  at ", cut, ": ", (h1_pass.Integral()/h1_incl.Integral()) 
                #print "sensitivity gain at ", cut, ": ", (h2_pass.Integral()/h2_incl.Integral()) /  (3./2. + np.sqrt(h1_pass.Integral())) / normFactor

            else:
                sensitivity.append(0)

            i = i+1

        c.pads()[1].ylim(0,80)
        c.pads()[1].xlim(0,200)
        c.pads()[1].graph( sensitivity, bins=baseBins) #, oob=False )

        ### Decorations ###
        c.legend(width=0.4, xmin=0.3, ymax=0.9)
        #c.xlabel(latex(var, ROOT=True))
        c.ylabel("Fraction of jets")
        c.xlabel("n_{trk}") #latex(var, ROOT=True))                                             
        c.pads()[1].ylabel("sensitivity gain") #"#epsilon_{S}/(#frac{3}{2} + #sqrt{B})")
        c.pads()[1].text(["sensitivity = #epsilon_{S}/(#frac{3}{2} + #sqrt{B})",
                ], xmin=0.2, ymax=0.80, ATLAS=False)

        ### Save ###
        c.save('figures/distributions/sensitivity_{}_sig{}_eff{}.pdf'.format(var, sigNumber, kNN_eff))
        c.save('figures/distributions/sensitivity_{}_sig{}_eff{}.eps'.format(var, sigNumber, kNN_eff))
        pass

    """
    # 2D histograms
"""
    msk = data['signal'] == 1
    axisvars = sorted(list(knn_axes))
    
    varx = kNN_var[0];
    vary = kNN_var[1];

    #for i,varx in enumerate(axisvars):
    #   for vary in axisvars[i+1:]:
    # Canvas
    c = ROOT.TCanvas()
    c.SetRightMargin(0.20)
    
    # Create, fill histogram
    h2 = ROOT.TH2F('{}_{}'.format(varx, vary), "", *(knn_axes[vary] + knn_axes[vary]))
    root_numpy.fill_hist(h2, data.loc[msk, [varx, vary]].values, 100. * data.loc[msk, weight].values)
    
    # Draw
    h2.Draw("COLZ")

    # Decorations
    h2.GetXaxis().SetTitle(latex(varx, ROOT=True))
    h2.GetYaxis().SetTitle(latex(vary, ROOT=True))
    c.SetLogz()
    
    # Save
    c.SaveAs('figures/distributions/2d_{}_{}.pdf'.format(varx, vary))
    c.SaveAs('figures/distributions/2d_{}_{}.eps'.format(varx, vary))
#    pass
#pass

    return
"""

# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args(backend=True, plots=True)

    # Call main function
    main(args)
    pass
