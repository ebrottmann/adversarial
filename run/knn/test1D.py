#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for testing fixed-efficiency kNN regressor."""

# Basic import(s)
import gzip
import pickle

# Scientific import(s)
import ROOT
import numpy as np
import root_numpy

# Project import(s)
from adversarial.utils import latex, parse_args, initialise, load_data, mkdir, loadclf  #, initialise_backend
from adversarial.profile import profile, Profile
from adversarial.constants import *
from tests.studies import jetmasscomparison
#from run.adversarial.common import initialise_config

# Local import(s)
from .common import *

# Custom import(s)
import rootplotting as rp

# Global definitions
BOUND = ROOT.TF1('bounds_0', "TMath::Sqrt( TMath::Power( 50, 2) * TMath::Exp(-x) )", AXIS[VARX][1], AXIS[VARX][2])

BOUND.SetLineColor(ROOT.kGray + 3)
BOUND.SetLineWidth(1)
BOUND.SetLineStyle(2)

YRANGE = (20., 120.)
XRANGE = FIT_RANGE 

# Main function definition
@profile
def main (args):

    # Initialise
    args, cfg = initialise(args)

    # Load data
    data, _, _ = load_data('data/' + args.input, test=True)
    msk_sig = data['signal'] == 1
    msk_bkg = ~msk_sig

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

    # Fill measured profile
    profile_meas, (x,percs, err) = fill_profile_1D(data[msk_bkg])

    # Add k-NN variable
    knnfeat = 'knn'
    add_knn(data, newfeat=knnfeat, path='models/knn/{}_{}_{}_{}.pkl.gz'.format(FIT, VAR, EFF, MODEL)) 

    # Loading KNN classifier
    knn = loadclf('models/knn/{}_{:s}_{}_{}.pkl.gz'.format(FIT, VAR, EFF, MODEL))


    # Filling fitted profile
    with Profile("Filling fitted profile"):
        rebin = 8

        # Short-hands
        vbins, vmin, vmax = AXIS[VARX]

        # Re-binned bin edges  @TODO: Make standardised right away?
        # edges = np.interp(np.linspace(0, vbins, vbins * rebin + 1, endpoint=True), 
        #                  range(vbins + 1),
        #                  np.linspace(vmin, vmax,  vbins + 1,         endpoint=True))

        fineBins = np.linspace(vmin, vmax,  vbins*rebin + 1,         endpoint=True)
        orgBins = np.linspace(vmin, vmax,  vbins + 1,         endpoint=True)

        # Re-binned bin centres
        fineCentres = fineBins[:-1] + 0.5 * np.diff(fineBins)
        orgCentres = orgBins[:-1] + 0.5 * np.diff(orgBins)
        
        pass

        # Get predictions evaluated at re-binned bin centres
        fit = knn.predict(fineCentres.reshape(-1,1)) #centres.reshape(-1,1))

        # Fill ROOT "profile"
        profile_fit = ROOT.TH1F('profile_fit', "", len(fineBins) - 1, fineBins.flatten('C'))
        root_numpy.array2hist(fit, profile_fit)

        outFile = ROOT.TFile.Open("figures/knn_jet_ungrtrk500_eff{}_data.root".format(EFF),"RECREATE")
        outFile.cd()
        profile_fit.SetName("kNNfit")
        profile_fit.Write()
        outFile.Close()

        # profile_meas2 = ROOT.TH1F('profile_meas', "", len(x) - 1, x.flatten('C'))
        # root_numpy.array2hist(percs, profile_meas2)
        profile_meas2 = ROOT.TGraph(len(x), x, percs) 
        pass


    # Plotting
    with Profile("Plotting"):
        # Plot
        plot(profile_meas2, profile_fit)
        pass

    # Plotting local selection efficiencies for D2-kNN < 0
    # -- Compute signal efficiency
    for sig, msk in zip([True, False], [msk_sig, msk_bkg]):

        # Define arrays
        shape   = AXIS[VARX][0]
        bins    = np.linspace(AXIS[VARX][1], AXIS[VARX][2], AXIS[VARX][0] + 1, endpoint=True)
        x, y = (np.zeros(shape) for _ in range(2))

        # Create `profile` histogram
        profile = ROOT.TH1F('profile', "", len(bins) - 1, bins.flatten('C') )

        # Compute inclusive efficiency in bins of `VARX`
        effs = list()

        for i in range(shape):
            msk_bin  = (data[VARX] > bins[i]) & (data[VARX] <= bins[i+1])
            msk_pass =  data[knnfeat] > 0 # <?
            num = data.loc[msk & msk_bin & msk_pass, 'weight'].values.sum()
            den = data.loc[msk & msk_bin,            'weight'].values.sum()
            if den > 0:
                eff = num/den
                profile.SetBinContent(i + 1, eff)
                effs.append(eff)
            else:
                print i, "Density = 0"

            pass


        c = rp.canvas(batch=True)
        pad = c.pads()[0]._bare()
        pad.cd()
        pad.SetRightMargin(0.20)
        pad.SetLeftMargin(0.15)
        pad.SetTopMargin(0.10)

        # Styling
        profile.SetLineColor(rp.colours[1])
        profile.SetMarkerStyle(24)
        profile.GetXaxis().SetTitle( "#it{m}_{jj} [GeV]" ) #latex(VARX, ROOT=True) + "[GeV]") #+ " = log(m^{2}/p_{T}^{2})")
        #profile.GetXaxis().SetTitle("Large-#it{R} jet " + latex(VARX, ROOT=True))# + " = log(m^{2}/p_{T}^{2})")
        profile.GetYaxis().SetTitle("Selection efficiency for #it{n}_{trk}^{#varepsilon=%s%%}>0" % ( EFF))

        profile.GetYaxis().SetNdivisions(505)
        profile.GetXaxis().SetTitleOffset(1.4)
        profile.GetYaxis().SetTitleOffset(1.8)
        profile.GetXaxis().SetRangeUser(*XRANGE)

        yrange = (0., 0.07)
        if yrange:
            profile.GetYaxis().SetRangeUser(*yrange)
            pass

        # Draw
        profile.Draw()

        # Decorations
        #c.text(qualifier=QUALIFIER, ymax=0.92, xmin=0.15)
        #c.text(["#sqrt{s} = 13 TeV", "Model" + MODEL if sig else "Multijets"], ATLAS=False)

        # -- Efficiencies
        xaxis = profile.GetXaxis()
        yaxis = profile.GetYaxis()
        tlatex = ROOT.TLatex()
        tlatex.SetTextColor(ROOT.kGray + 2)
        tlatex.SetTextSize(0.023)
        tlatex.SetTextFont(42)
        tlatex.SetTextAlign(32)
        xt = xaxis.GetBinLowEdge(xaxis.GetNbins())

        for eff, ibin in zip(effs,range(1, yaxis.GetNbins() + 1)):
            yt = yaxis.GetBinCenter(ibin)
            tlatex.DrawLatex(xt, yt, "%s%.1f%%" % ("#bar{#varepsilon}^{rel}_{%s} = " % ('sig' if sig else 'bkg') if ibin == 1 else '', eff * 100.))
        pass

        # -- Bounds
        #BOUND.Draw("SAME")
        #c.latex("m > 50 GeV",  -4.5, BOUNDS[0].Eval(-4.5) + 30, align=21, angle=-37, textsize=13, textcolor=ROOT.kGray + 3)
        #c.latex("m < 300 GeV", -2.5, BOUNDS[1].Eval(-2.5) - 30, align=23, angle=-57, textsize=13, textcolor=ROOT.kGray + 3)

        # Save
        mkdir('figures/knn/')
        c.save('figures/knn/{}_eff_{}_{:s}_{}_{}.pdf'.format(FIT, 'sig' if sig else 'bkg', VAR, EFF, MODEL))
        c.save('figures/knn/{}_eff_{}_{:s}_{}_{}.eps'.format(FIT, 'sig' if sig else 'bkg', VAR, EFF, MODEL))
        del c
        
        pass

    return


def plot (profile, fit):
    """
    Method for delegating plotting.
    """

    # rootplotting
    c = rp.canvas(batch=True)
    pad = c.pads()[0]._bare()
    pad.cd()
    pad.SetRightMargin(0.20)
    pad.SetLeftMargin(0.15)
    pad.SetTopMargin(0.10)

    # Styling
    #profile.SetLineColor(4)
    profile.SetMarkerColor(4)
    profile.SetMarkerStyle(20)
    fit.SetLineColor(2)
    fit.SetMarkerColor(4)
    fit.SetMarkerStyle(20)
    profile.GetXaxis().SetTitle( "#it{m}_{jj} [GeV]" ) #latex(VARX, ROOT=True) + " [GeV]") #+ " = log(m^{2}/p_{T}^{2})")
    profile.GetYaxis().SetTitle( "#it{P}^{#varepsilon=%s%%}" % (EFF) )
#"%s %s^{(%s%%)}" % ("#it{k}-NN fitted" if fit else "Measured", latex(VAR, ROOT=True), EFF))

    profile.GetYaxis().SetNdivisions(505)
    profile.GetXaxis().SetTitleOffset(1.4)
    profile.GetYaxis().SetTitleOffset(1.4)
    profile.GetXaxis().SetRangeUser(*XRANGE)

    if YRANGE:
        profile.GetYaxis().SetRangeUser(*YRANGE)
        pass

    # Draw Goddamn it

    #    print profile.GetBinContent(10), profile.GetNbinsX(), profile.GetEntries()

    profile.Draw("AP")
    fit.Draw("SAME") #("SAME")
    
    leg = ROOT.TLegend(0.2, 0.75, 0.5, 0.85)
    leg.AddEntry(profile, "Control Region Data", "p")
    leg.AddEntry(fit, "k-NN fit", "l")
    leg.Draw() 

    #BOUND.DrawCopy("SAME")
    #c.latex("m > 50 GeV",  -4.5, BOUNDS[0].Eval(-4.5) + 30, align=21, angle=-37, textsize=13, textcolor=ROOT.kGray + 3)
    #c.latex("m < 300 GeV", -2.5, BOUNDS[1].Eval(-2.5) - 30, align=23, angle=-57, textsize=13, textcolor=ROOT.kGray + 3)

    # Decorations
    #c.text(qualifier=QUALIFIER, ymax=0.92, xmin=0.15)
    #c.text(["#sqrt{s} = 13 TeV", "Multijets"], ATLAS=False, textcolor=ROOT.kWhite)

    # Save
    mkdir('figures/knn/')
    c.save('figures/knn/{}_profile_{:s}_{}_{}.pdf'.format( FIT, VAR, EFF, MODEL))
    c.save('figures/knn/{}_profile_{:s}_{}_{}.eps'.format( FIT, VAR, EFF, MODEL))
    
    del c
    pass


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()  # (adversarial=True)

    # Call main function
    main(args)
    pass
