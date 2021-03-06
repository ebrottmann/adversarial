#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ROOT import(s)
import ROOT
import root_numpy

# Project import(s)
from .common import *
from adversarial.utils import mkdir, latex, wpercentile, signal_low, MASSBINS
from adversarial.constants import *

# Custom import(s)
import rootplotting as rp


#@showsave
def jetmasscomparison (data, args, features, eff_sig=25):
    """
    Perform study of jet mass distributions before and after subtructure cut for
    different substructure taggers.

    Saves plot `figures/jetmasscomparison__eff_sig_[eff_sig].pdf`

    Arguments:
        data: Pandas data frame from which to read data.
        args: Namespace holding command-line arguments.
        features: Features for which to plot signal- and background distributions.
        eff_sig: Signal efficiency at which to impose cut.
    """

    # Define masks and direction-dependent cut value
    msk_sig = data['sigType'] == 1
    cuts, msks_pass = dict(), dict()
    lead_features = []

    print "Features: ", features

    for feat in features:
        eff_cut = eff_sig if signal_low(feat) else 100 - eff_sig
        
        if (not 'lead' in feat) and (not 'sub' in feat):
            print "hej"

            cut = wpercentile(data.loc[msk_sig, feat].values, eff_cut, weights=data.loc[msk_sig, 'weight'].values)
            msk = (data[feat] > cut) 

            fpr, tpr, thresholds = roc_curve(data['signal'], data[feat], sample_weight=data['weight'])
            idx = np.argmin(np.abs(tpr - eff_sig/100.))
            
            print "Pass criteria:", feat, " > ", cut
            print "Background acceptance @ {:.2f}% sig. eff.: {:.5f}% ({} > {:.2f})".format(eff_sig, (fpr[idx]) * 100., feat, thresholds[idx])

            msks_pass[feat]=msk
            lead_features.append(feat)

        else:

            if 'lead' in feat:
                cut1 = wpercentile(data.loc[msk_sig, feat].values, eff_cut, weights=data.loc[msk_sig, 'weight'].values)
                msk1 = (data[feat] > cut1) 

                fpr, tpr, thresholds = roc_curve(data['signal'], data[feat], sample_weight=data['weight'])
                idx = np.argmin(np.abs(tpr - eff_sig/100.))
            
                print "H Pass criteria:", feat, " > ", cut1
                print "H Background acceptance @ {:.2f}% sig. eff.: {:.6f}% ({} > {:.2f})".format(eff_sig, (fpr[idx]) * 100., feat, thresholds[idx])

                lead_features.append(feat)

                subfeat = feat.replace("lead", "sub")
                data1 = data[msk1]
                cut2 = wpercentile(data1.loc[msk_sig, subfeat].values, eff_cut, weights=data1.loc[msk_sig, 'weight'].values)
                fpr, tpr, thresholds = roc_curve(data1['signal'], data1[subfeat], sample_weight=data1['weight'])

                idx = np.argmin(np.abs(tpr - eff_sig/100.))
                idy = np.argmin(np.abs(thresholds - cut1))
            

                print "H Pass criteria:", subfeat, " > ", cut2, idy, len(thresholds) 
                print "H Background acceptance @ {:.5f}% sig. eff.: {:.5f}% ({} > {:.5f})".format((tpr[idy])*100, (fpr[idy]) * 100., subfeat, thresholds[idy])

                #msks_pass[feat]=(data[feat]>cut1) | (data[subfeat]>cut1)
                msks_pass[feat]=(data[feat]>cut1) & (data[subfeat]>cut1)

        # Ensure correct cut direction
        if signal_low(feat):
            msks_pass[feat] = ~msks_pass[feat]
            pass
        pass
        
                        
    # Perform plotting
    #c = plot(data, args, features, msks_pass, eff_sig)

    # Perform plotting on individual figures
    c = plot_individual(data, args, lead_features, msks_pass, eff_sig)

    # Output
    path = 'figures/jetmasscomparison__eff_sig_{:d}_{}.pdf'.format(int(eff_sig), MODEL)
    path = 'figures/jetmasscomparison__eff_sig_{:d}_{}.eps'.format(int(eff_sig), MODEL)

    return c, args, path



def plot_individual (*argv):
    """
    Method for delegating plotting.
    """

    # Unpack arguments
    data, args, features, msks_pass, eff_sig = argv

    with TemporaryStyle() as style:

        # Style @TEMP?
        ymin, ymax = 5E-05, 5E+00
        scale = 0.6
        for coord in ['x', 'y', 'z']:
            style.SetLabelSize(style.GetLabelSize(coord) * 0.7 * scale, coord)
            style.SetTitleSize(style.GetTitleSize(coord) * 0.7 * scale, coord)
            pass
        style.SetTextSize      (style.GetTextSize()       * scale)
        style.SetLegendTextSize(style.GetLegendTextSize() * (scale + 0.03))
        style.SetTickLength(0.07,                     'x')
        style.SetTickLength(0.07 * (5./6.) * (2./3.), 'y')

        # Global variable override(s)
        histstyle = dict(**HISTSTYLE)
        histstyle[True]['fillstyle'] = 3554
        histstyle[True] ['linewidth'] = 4
        histstyle[False]['linewidth'] = 4
        histstyle[True] ['label'] = None
        histstyle[False]['label'] = None
        for v in ['linecolor', 'fillcolor']:
            histstyle[True] [v] = 16
            histstyle[False][v] = ROOT.kGray+2
            pass
        style.SetHatchesLineWidth(6)

        # Loop features
        ts  = style.GetTextSize()
        lts = style.GetLegendTextSize()
        for ifeat, feats in enumerate([None] + list(zip(features[::2], features[1::2])), start=-1):
            first = ifeat == -1

            # Style
            style.SetTitleOffset(1.25 if first else 1.2, 'x')
            style.SetTitleOffset(1.7  if first else 1.6, 'y')
            style.SetTextSize(ts * (0.8 if first else scale))
            style.SetLegendTextSize(lts * (0.8 + 0.03 if first else scale + 0.03))

            # Canvas
            c1 = rp.canvas(batch=not args.show, size=(300, 200)) #int(200 * (1.45 if first else 1.))))
            c2 = rp.canvas(batch=not args.show, size=(300, 200)) #int(200 * (1.45 if first else 1.))))

            if first:
                opts = dict(xmin=0.185, width=0.60) #, columns=2)
                c1.legend(header=' ', categories=[
                            ("Multijets",   histstyle[False]),
                            ("Dark jets", histstyle[True])
                        ], ymax=0.45, **opts)
                c2.legend(header=' ', categories=[
                            ("Multijets",   histstyle[False]),
                            ("Dark jets", histstyle[True])
                        ], ymax=0.45, **opts)
                c1.legend(header='Inclusive selection:',
                         ymax=0.38, **opts)
                c2.legend(header='Inclusive selection:',
                         ymax=0.38, **opts)
                #c1.pad()._legends[-2].SetTextSize(stdijetmassyle.GetLegendTextSize())
                #c1.pad()._legends[-1].SetTextSize(style.GetLegendTextSize())
                c1.pad()._legends[-2].SetMargin(0.35)
                c1.pad()._legends[-1].SetMargin(0.35)
                c2.pad()._legends[-2].SetMargin(0.35)
                c2.pad()._legends[-1].SetMargin(0.35)

                c1.text(["#sqrt{s} = 13 TeV,  Dark jet tagging",
                        "Cuts at #varepsilon_{sig}^{rel} = %.0f%%" % eff_sig,
                        ], xmin=0.2, ymax=0.80, qualifier=QUALIFIER)

                c2.text(["#sqrt{s} = 13 TeV,  Dark jet tagging",
                        "Cuts at #varepsilon_{sig}^{rel} = %.0f%%" % eff_sig,
                        ], xmin=0.2, ymax=0.80, qualifier=QUALIFIER)

            else:
                # Plots
                # -- Dummy, for proper axes
                c1.hist([ymin], bins=[1000, 6000], linestyle=0, fillstyle=0)
                c2.hist([ymin], bins=[1000, 2000], linestyle=0, fillstyle=0)

                # -- Inclusive
                base = dict(bins=MASSBINS, normalise=True)
                for signal, name in zip([False, True], ['bkg', 'sig']):
                    msk = data['signal'] == signal
                    histstyle[signal].update(base)
                    histstyle[signal]['option'] = 'HIST'
                    c1.hist(data.loc[msk, 'dijetmass'].values, weights=data.loc[msk, 'weight'].values, **histstyle[signal])
                    c2.hist(data.loc[msk, 'lead_jet_pt'].values, weights=data.loc[msk, 'weight'].values, **histstyle[signal])
                    pass

                for sig in [True, False]:
                    histstyle[sig]['option'] = 'FL'
                    pass

                # -- Tagged
                for jfeat, feat in enumerate(feats):
                    opts = dict(
                        linecolor = rp.colours[((2 * ifeat + jfeat) // 2)],
                        linestyle = 1 + 6 * (jfeat % 2),
                        linewidth = 4,
                        )
                    cfg = dict(**base)
                    cfg.update(opts)
                    msk = (data['signal'] == 0) & (msks_pass[feat])

                    print "Debugging: ", len(data.loc[msk, 'dijetmass'].values)
                    print "Debugging: ", len(data.loc[msk, 'weight'].values)

                    c1.hist(data.loc[msk, 'dijetmass'].values, weights=data.loc[msk, 'weight'].values, label=" " + latex(feat.replace("lead_", "").replace("jet_", ""), ROOT=True), **cfg)

                    c2.hist(data.loc[msk, 'lead_jet_pt'].values, weights=data.loc[msk, 'weight'].values, label=" " + latex(feat.replace("lead_", "").replace("jet_", ""), ROOT=True), **cfg)
                    pass

                # -- Legend(s)
                y =  0.825  if first else 0.85
                dy = 0.025 if first else 0.025
                c1.legend(width=0.25, xmin=0.7, ymax=y-dy)
                c1.latex("Tagged multijets:", NDC=True, x=0.87, y=y, textcolor=ROOT.kGray + 3, textsize=style.GetLegendTextSize() * 0.9, align=31)
                c1.pad()._legends[-1].SetMargin(0.35)
                c1.pad()._legends[-1].SetTextSize(style.GetLegendTextSize())

                c2.legend(width=0.25, xmin=0.7, ymax=y-dy)
                c2.latex("Tagged multijets:", NDC=True, x=0.87, y=y, textcolor=ROOT.kGray + 3, textsize=style.GetLegendTextSize() * 0.9, align=31)
                c2.pad()._legends[-1].SetMargin(0.35)
                c2.pad()._legends[-1].SetTextSize(style.GetLegendTextSize())


                opts = dict(xmin=0.7, width=0.25) #, columns=2)
                                                                                                                                             
                c1.legend(header=' ', categories=[
                            ("Multijets",   histstyle[False]),
                            ("Dark jets", histstyle[True])
                        ], ymax=y-6.5*dy , **opts)
                c2.legend(header=' ', categories=[
                            ("Multijets",   histstyle[False]),
                            ("Dark jets", histstyle[True])
                        ], ymax=y-6.5*dy, **opts)

                c1.latex("Inclusive selection:", NDC=True, x=0.88, y=y-7.5*dy, textcolor=ROOT.kGray + 3, textsize=style.GetLegendTextSize() * 0.9, align=31)
                c2.latex("Inclusive selection:", NDC=True, x=0.88, y=y-7.5*dy, textcolor=ROOT.kGray + 3, textsize=style.GetLegendTextSize() * 0.9, align=31)


                #c1.legend(header='Inclusive selection:',
                 #        ymax=y-5*dy, xmin=0.7, width=0.25)


                #c2.legend(header='Inclusive selection:',
                 #        ymax=y-5*dy, xmin=0.7, width=0.25)


                #c1.pad()._legends[-2].SetTextSize(stdijetmassyle.GetLegendTextSize())                                                                                                                       
 
                #c1.pad()._legends[-1].SetTextSize(style.GetLegendTextSize())                                                                                                                                 
                c1.pad()._legends[-2].SetMargin(0.35)
                c1.pad()._legends[-1].SetMargin(0.35)
                c2.pad()._legends[-2].SetMargin(0.35)
                c2.pad()._legends[-1].SetMargin(0.35)

                c1.text(["#sqrt{s} = 13 TeV,  Dark jet tagging",
                        "Cuts at #varepsilon_{sig}^{rel} = %.0f%%" % eff_sig,
                        ], xmin=0.25, ymax=0.8, ATLAS=False)

                c2.text(["#sqrt{s} = 13 TeV,  Dark jet tagging",
                        "Cuts at #varepsilon_{sig}^{rel} = %.0f%%" % eff_sig,
                        ], xmin=0.25, ymax=0.8, ATLAS=False)


                # Formatting pads
                tpad1 = c1.pad()._bare()
                tpad1.SetLeftMargin  (0.20)
                tpad1.SetBottomMargin(0.12 if first else 0.20)
                tpad1.SetTopMargin   (0.39 if first else 0.05)

                tpad2 = c2.pad()._bare()
                tpad2.SetLeftMargin  (0.20)
                tpad2.SetBottomMargin(0.12 if first else 0.20)
                tpad2.SetTopMargin   (0.39 if first else 0.05)

                # Re-draw axes
                tpad1.RedrawAxis()
                tpad1.Update()
                tpad2.RedrawAxis()
                tpad2.Update()

                c1.pad()._xaxis().SetAxisColor(ROOT.kWhite)  # Remove "double ticks"
                c1.pad()._yaxis().SetAxisColor(ROOT.kWhite)  # Remove "double ticks"

                c2.pad()._xaxis().SetAxisColor(ROOT.kWhite)  # Remove "double ticks"
                c2.pad()._yaxis().SetAxisColor(ROOT.kWhite)  # Remove "double ticks"

                # Decorations
                c1.xlabel("m_{jj} [GeV]") #("Large-#it{R} jet mass [GeV]")
                c1.ylabel("Fraction of jets")

                c2.xlabel("Jet p_{T} [GeV]") #("Large-#it{R} jet mass [GeV]")
                c2.ylabel("Fraction of jets")

                c1.text(qualifier=QUALIFIER, xmin=0.25, ymax=0.85)

                c1.ylim(ymin, ymax)
                c1.logy()

                c2.text(qualifier=QUALIFIER, xmin=0.25, ymax=0.85)

                c2.ylim(ymin, ymax)
                c2.logy()
                pass

            # Save
            c1.save(path = 'figures/jetmasscomparison_individual_eff_sig_{:d}_mjj_{}_{}.pdf'.format(int(eff_sig), 'legend' if first else '{}_{}'.format(*feats), MODEL)) 
            c1.save(path = 'figures/jetmasscomparison_individual_eff_sig_{:d}_mjj_{}_{}.eps'.format(int(eff_sig), 'legend' if first else '{}_{}'.format(*feats), MODEL)) 
            c2.save(path = 'figures/jetmasscomparison_individual_eff_sig_{:d}_pt_{}_{}.pdf'.format(int(eff_sig), 'legend' if first else '{}_{}'.format(*feats), MODEL)) 
            c2.save(path = 'figures/jetmasscomparison_individual_eff_sig_{:d}_pt_{}_{}.eps'.format(int(eff_sig), 'legend' if first else '{}_{}'.format(*feats), MODEL)) 
            pass
        pass  # end temprorary style

    return c1, c2

def plot (*argv):
    """
    Method for delegating plotting.
    """

    # Unpack arguments
    data, args, features, msks_pass, eff_sig = argv

    with TemporaryStyle() as style:

        # Style
        ymin, ymax = 5E-05, 5E+00
        scale = 0.8
        for coord in ['x', 'y', 'z']:
            style.SetLabelSize(style.GetLabelSize(coord) * scale, coord)
            style.SetTitleSize(style.GetTitleSize(coord) * scale, coord)
            pass
        style.SetTextSize      (style.GetTextSize()       * scale)
        style.SetLegendTextSize(style.GetLegendTextSize() * scale)
        style.SetTickLength(0.07,                     'x')
        style.SetTickLength(0.07 * (5./6.) * (2./3.), 'y')

        # Global variable override(s)
        histstyle = dict(**HISTSTYLE)
        histstyle[True]['fillstyle'] = 3554
        histstyle[True] ['label'] = None
        histstyle[False]['label'] = None
        for v in ['linecolor', 'fillcolor']:
            histstyle[True] [v] = 16
            histstyle[False][v] = ROOT.kGray+2
            pass
        style.SetHatchesLineWidth(1)

        # Canvas
        c = rp.canvas(batch=not args.show, num_pads=(2,3))

        # Plots
        # -- Dummy, for proper axes
        for ipad, pad in enumerate(c.pads()[1:], 1):
            pad.hist([ymin], bins=[1000, 6000], linestyle=0, fillstyle=0, option=('Y+' if ipad % 2 else ''))
            pass

        # -- Inclusive
        base = dict(bins=MASSBINS, normalise=True, linewidth=2)
        for signal, name in zip([False, True], ['bkg', 'sig']):
            msk = data['signal'] == signal
            histstyle[signal].update(base)
            for ipad, pad in enumerate(c.pads()[1:], 1):
                histstyle[signal]['option'] = 'HIST'
                pad.hist(data.loc[msk, 'lead_jet_pt'].values, weights=data.loc[msk, 'weight'].values, **histstyle[signal])
                pass
            pass

        for sig in [True, False]:
            histstyle[sig]['option'] = 'FL'
            pass

        c.pads()[0].legend(header='Inclusive selection:', categories=[
            ("Multijets",   histstyle[False]),
            ("Dark jets", histstyle[True])
            ], xmin=0.18, width= 0.60, ymax=0.28 + 0.08, ymin=0.001 + 0.07, columns=2)
        c.pads()[0]._legends[-1].SetTextSize(style.GetLegendTextSize())
        c.pads()[0]._legends[-1].SetMargin(0.35)

        # -- Tagged
        base['linewidth'] = 2
        for ifeat, feat in enumerate(features):
            opts = dict(
                linecolor = rp.colours[(ifeat // 2)],
                linestyle = 1 + (ifeat % 2),
                linewidth = 2,
                )
            cfg = dict(**base)
            cfg.update(opts)
            msk = (data['signal'] == 0) & msks_pass[feat]
            pad = c.pads()[1 + ifeat//2]
            pad.hist(data.loc[msk, 'lead_jet_pt'].values, weights=data.loc[msk, 'weight'].values, label=" " + latex(feat, ROOT=True), **cfg)
            pass

        # -- Legend(s)
        for ipad, pad in enumerate(c.pads()[1:], 1):
            offsetx = (0.20 if ipad % 2 else 0.05)
            offsety =  0.20 * ((2 - (ipad // 2)) / float(2.))
            pad.legend(width=0.25, xmin=0.68 - offsetx, ymax=0.80 - offsety)
            pad.latex("Tagged multijets:", NDC=True, x=0.93 - offsetx, y=0.84 - offsety, textcolor=ROOT.kGray + 3, textsize=style.GetLegendTextSize() * 0.8, align=31)
            #pad._legends[-1].SetMargin(0.35)
            #pad._legends[-1].SetTextSize(style.GetLegendTextSize())
            pass

        # Formatting pads
        margin = 0.2
        for ipad, pad in enumerate(c.pads()):
            tpad = pad._bare()  # ROOT.TPad
            right = ipad % 2
            f = (ipad // 2) / float(len(c.pads()) // 2 - 1)
            tpad.SetLeftMargin (0.05 + 0.15 * (1 - right))
            tpad.SetRightMargin(0.05 + 0.15 * right)
            tpad.SetBottomMargin(f * margin)
            tpad.SetTopMargin((1 - f) * margin)
            if ipad == 0: continue
            pad._xaxis().SetNdivisions(505)
            pad._yaxis().SetNdivisions(505)
            if ipad // 2 < len(c.pads()) // 2 - 1:  # Not bottom pad(s)
                pad._xaxis().SetLabelOffset(9999.)
                pad._xaxis().SetTitleOffset(9999.)
            else:
                pad._xaxis().SetTitleOffset(2.7)
                pass
            pass

        # Re-draw axes
        for pad in c.pads()[1:]:
            pad._bare().RedrawAxis()
            pad._bare().Update()
            pad._xaxis().SetAxisColor(ROOT.kWhite)  # Remove "double ticks"
            pad._yaxis().SetAxisColor(ROOT.kWhite)  # Remove "double ticks"
            pass

        # Decorations
        c.pads()[-1].xlabel("Large-#it{R} jet mass [GeV]")
        c.pads()[-2].xlabel("Large-#it{R} jet mass [GeV]")
        c.pads()[1].ylabel("#splitline{#splitline{#splitline{#splitline{}{}}{#splitline{}{}}}{#splitline{}{}}}{#splitline{}{#splitline{}{#splitline{}{Fraction of jets}}}}")
        c.pads()[2].ylabel("#splitline{#splitline{#splitline{#splitline{Fraction of jets}{}}{}}{}}{#splitline{#splitline{}{}}{#splitline{#splitline{}{}}{#splitline{}{}}}}")
        # I have written a _lot_ of ugly code, but this ^ is probably the worst.

        c.pads()[0].text(["#sqrt{s} = 13 TeV,  Dark jet tagging",
                    "Cuts at #varepsilon_{sig}^{rel} = %.0f%%" % eff_sig,
                    ], xmin=0.2, ymax=0.72, qualifier=QUALIFIER)

        for pad in c.pads()[1:]:
            pad.ylim(ymin, ymax)
            pad.logy()
            pass

        pass  # end temprorary style

    return c

