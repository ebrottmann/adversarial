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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
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

YRANGE = (50., 90.)
XRANGE = FIT_RANGE 

def func(X, a, b, c):
    """ error function"""
    Y = []
    for x in range(len(X)):
        Y.append(a * erf(b*(x+c)))
    return Y


# Main function definition
@profile
def main (args):

    # Initialise
    args, cfg = initialise(args)

    # Load data
    data, _, _ = load_data('data/' + args.input) #, test=True)
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
    weights = 1/err

    # Add k-NN variable
    knnfeat = 'knn'
    orgfeat = VAR
    add_knn(data, newfeat=knnfeat, path='models/knn/{}_{}_{}_{}.pkl.gz'.format(FIT, VAR, EFF, MODEL)) 

    # Loading KNN classifier
    knn = loadclf('models/knn/{}_{:s}_{}_{}.pkl.gz'.format(FIT, VAR, EFF, MODEL))
    #knn = loadclf('models/knn/{}_{:s}_{}_{}.pkl.gz'.format(FIT, VAR, EFF, MODEL))

    X = x.reshape(-1,1)

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
        if 'erf' in FIT:
            fit = func(fineCentres, knn[0], knn[1], knn[2])
            print "Check: ", func([1500, 2000], knn[0], knn[1], knn[2]) 
        else:
            fit = knn.predict(fineCentres.reshape(-1,1)) #centres.reshape(-1,1))

        # Fill ROOT "profile"
        profile_fit = ROOT.TH1F('profile_fit', "", len(fineBins) - 1, fineBins.flatten('C'))
        root_numpy.array2hist(fit, profile_fit)
        
        knn1 = PolynomialFeatures(degree=2)                                           
        X_poly = knn1.fit_transform(X)
        reg = LinearRegression(fit_intercept=False) #fit_intercept=False)
        reg.fit(X_poly, percs, weights)
        score = round(reg.score(X_poly, percs), 4)
        coef = reg.coef_
        intercept = reg.intercept_
        print "COEFFICIENTS: ", coef, intercept
        
        TCoef = ROOT.TVector3(coef[0], coef[1], coef[2]) 
        outFile = ROOT.TFile.Open("models/{}_jet_ungrtrk500_eff{}_stat{}_{}.root".format(FIT, EFF, MIN_STAT, MODEL),"RECREATE")
        outFile.cd()
        TCoef.Write()
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

    # MC weights are scaled with lumi. This is just for better comparison
    #if INPUT =="mc": 
    #    data.loc[:,'TotalEventWeight'] /=  139000000. 

    for sig, msk in zip([True, False], [msk_sig, msk_bkg]):

        # Define arrays
        shape   = AXIS[VARX][0]
        bins    = np.linspace(AXIS[VARX][1], AXIS[VARX][2], AXIS[VARX][0]+ 1, endpoint=True)
        #bins = np.linspace(AXIS[VARX][1], 4000, 40, endpoint=True)
        #bins = np.append(bins, [4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000])

        print "HERE: ", bins 
        
        #x, y = (np.zeros(shape) for _ in range(2))

        # Create `profile` histogram
        profile_knn = ROOT.TH1F('profile', "", len(bins) - 1, bins ) #.flatten('C') )
        profile_org = ROOT.TH1F('profile', "", len(bins) - 1, bins ) #.flatten('C') )

        # Compute inclusive efficiency in bins of `VARX`
        effs = list()
        
        for i in range(shape):
            msk_bin  = (data[VARX] > bins[i]) & (data[VARX] <= bins[i+1])
            msk_pass =  data[knnfeat] > 0 # <?
            msk_pass_org =  data[orgfeat] > 70 # <?
            num = data.loc[msk & msk_bin & msk_pass, 'TotalEventWeight'].values.sum()
            num_org = data.loc[msk & msk_bin & msk_pass_org, 'TotalEventWeight'].values.sum()
            den = data.loc[msk & msk_bin,'TotalEventWeight'].values.sum()
            if den > 0:
                eff = num/den *100.
                eff_org = num_org/den *100.
                profile_knn.SetBinContent(i + 1, eff)
                profile_org.SetBinContent(i + 1, eff_org)
                effs.append(eff)
            #else:
            #print i, "Density = 0"
            pass

        c = rp.canvas(batch=True)
        leg = ROOT.TLegend(0.2, 0.75, 0.5, 0.85)
        leg.AddEntry(profile_knn, "#it{n}_{trk}^{#varepsilon=%s%%} > 0" % ( EFF), "l")
        leg.AddEntry(profile_org, "#it{n}_{trk} > 70", "l")
        leg.Draw()

        pad = c.pads()[0]._bare()
        pad.cd()
        pad.SetRightMargin(0.10)
        pad.SetLeftMargin(0.15)
        pad.SetTopMargin(0.10)

        # Styling
        profile_knn.SetLineColor(rp.colours[1])
        profile_org.SetLineColor(rp.colours[2])
        profile_knn.SetMarkerStyle(24)
        profile_knn.GetXaxis().SetTitle( "#it{m}_{jj} [GeV]" ) #latex(VARX, ROOT=True) + "[GeV]") #+ " = log(m^{2}/p_{T}^{2})")
        #profile.GetXaxis().SetTitle("Large-#it{R} jet " + latex(VARX, ROOT=True))# + " = log(m^{2}/p_{T}^{2})")
        profile_org.GetYaxis().SetTitle("Selection efficiency (%)") # for #it{n}_{trk}^{#varepsilon=%s%%}>0" % ( EFF))

        profile_knn.GetYaxis().SetNdivisions(505)
        #profile_knn.GetXaxis().SetNdivisions(505)
        profile_knn.GetXaxis().SetTitleOffset(1.4)
        profile_knn.GetYaxis().SetTitleOffset(1.8)
        profile_knn.GetXaxis().SetRangeUser(*XRANGE)
        profile_org.GetXaxis().SetRangeUser(*XRANGE)

        yrange = (0., EFF*3) #2.0 percent
        if yrange:
            profile_knn.GetYaxis().SetRangeUser(*yrange)
            profile_org.GetYaxis().SetRangeUser(*yrange)
            pass

        # Draw
        profile_org.Draw()
        profile_knn.Draw("same")

        # Save
        mkdir('figures/knn/')
        c.save('figures/knn/{}_eff_{}_{:s}_{}_{}_stat{}.pdf'.format(FIT, 'sig' if sig else 'bkg', VAR, EFF, MODEL+INPUT, MIN_STAT))
        #c.save('figures/knn/{}_eff_{}_{:s}_{}_{}_stat{}.png'.format(FIT, 'sig' if sig else 'bkg', VAR, EFF, MODEL, MIN_STAT))
        c.save('figures/knn/{}_eff_{}_{:s}_{}_{}_stat{}.eps'.format(FIT, 'sig' if sig else 'bkg', VAR, EFF, MODEL+INPUT, MIN_STAT))
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
    #profile.GetXaxis().SetRangeUser(1000, 9000)
    #fit.GetXaxis().SetRangeUser(1000, 8000)

    if YRANGE:
        profile.GetYaxis().SetRangeUser(*YRANGE)
        pass

    # Draw Goddamn it

    #    print profile.GetBinContent(10), profile.GetNbinsX(), profile.GetEntries()

    profile.Draw("AP")
    fit.Draw("SAME") #("SAME")
    
    leg = ROOT.TLegend(0.2, 0.75, 0.5, 0.85)

    if INPUT=='data':
        leg.AddEntry(profile, "CR Data", "p")
    elif INPUT=='mcCR':
        leg.AddEntry(profile, "CR MC", "p")
    elif INPUT=='mc':
        leg.AddEntry(profile, "Full MC", "p")


    if 'knn' in FIT:
        fitLegend =  "k-NN fit "
    elif 'poly2' in FIT:
        fitLegend = "2. order polynomial fit " 
    elif 'poly3' in FIT:
        fitLegend = "3. order polynomial fit "
    elif 'erf' in FIT:
        fitLegend = "Error function fit "

    if MODEL=='data':
        fitLegend += "to CR Data"
    elif MODEL=='mcCR':
        fitLegend += "to CR MC"
    elif MODEL=='mc':
        fitLegend += "to Full MC"

    leg.AddEntry(fit, fitLegend, "l")
    leg.Draw() 


    # Save
    mkdir('figures/knn/')
    c.save('figures/knn/{}_profile_{:s}_{}_{}_stat{}.pdf'.format( FIT, VAR, EFF, MODEL+INPUT, MIN_STAT))
    #c.save('figures/knn/{}_profile_{:s}_{}_{}_stat{}.png'.format( FIT, VAR, EFF, MODEL, MIN_STAT))
    c.save('figures/knn/{}_profile_{:s}_{}_{}_stat{}.eps'.format( FIT, VAR, EFF, MODEL+INPUT, MIN_STAT))
    
    del c
    pass


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()  # (adversarial=True)

    # Call main function
    main(args)
    pass
