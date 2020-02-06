#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Basic import(s)
import h5py
import itertools
from glob import glob
import os
from ROOT import TFile

# Scientific import(s)
import numpy as np
import numpy.lib.recfunctions as rfn
import ROOT
import root_numpy

# Project import(s)
from adversarial.profile import profile


# Utility function(s)
glob_sort_list = lambda paths: sorted(list(itertools.chain.from_iterable(map(glob, paths))))

def rename (name):
    name = name.replace('lead_', '')
    name = name.replace('sub_', '')
    return name

def split (data):
    """
    Create one "event" per jet 
    """    
#    lead_jet_fields = list()
#    sub_jet_fields = list()
#    event_fields = list()

#    for name in data.dtype.names:
#        if 'lead' in name: 
#            lead_jet_fields.append(name)
#        elif 'sub' in name: 
#            sub_jet_fields.append(name)
#        else: 
#            event_fields.append(name)

#    lead_jet_fields = np.concatenate((event_fields, lead_jet_fields))
#    sub_jet_fields = np.concatenate((event_fields, sub_jet_fields))
    
#    data_leadjets   = data[lead_jet_fields]
#    data_subjets   = data[sub_jet_fields]    

#    bool_bothCR = (np.random.random_integers(0,1)<1) & (data['sub_jet_ungrtrk500']<20) & (data['lead_jet_ungrtrk500']<20)
#    bool_leadNoCR = data_leadjets['lead_jet_ungrtrk500']>20        

#    msk_lead = (bool_bothCR) | (bool_leadNoCR)

#    data_leadjets = data_leadjets[msk_lead]
#    data_subjets = data_subjets[not msk_lead]

    # Rename columns
#    data_leadjets.dtype.names = map(rename, data_leadjets.dtype.names)
#    data_subjets.dtype.names = map(rename, data_subjets.dtype.names)
    
#    return np.concatenate((data_leadjets, data_subjets))



def split (data):
    """                                                                                                                                                                                                 
    Create one "event" per jet                                                                                                                                                                          
    """
    lead_jet_fields = list()
    sub_jet_fields = list()
    event_fields = list()

    for name in data.dtype.names:
        if 'lead' in name:
            lead_jet_fields.append(name)
        elif 'sub' in name:
            sub_jet_fields.append(name)
        else:
            event_fields.append(name)

    lead_jet_fields = np.concatenate((event_fields, lead_jet_fields))
    sub_jet_fields = np.concatenate((event_fields, sub_jet_fields))

    data_leadjets   = data[lead_jet_fields]
    data_subjets   = data[sub_jet_fields]

    # Rename columns                                                                                                                                                                                    
    data_leadjets.dtype.names = map(rename, data_leadjets.dtype.names)
    data_subjets.dtype.names = map(rename, data_subjets.dtype.names)

    return np.concatenate((data_leadjets, data_subjets))

                
def unravel (data):
    """
    ...
    """
    
    if not data.dtype.hasobject:
        return data

    # Identify variable-length (i.e. per-jet) and scalar (i.e. per-event)
    # fields
    jet_fields = list()
    for field, (kind, _) in data.dtype.fields.iteritems():
        if kind.hasobject:
            jet_fields.append(field)
            pass
        pass
    jet_fields   = sorted(jet_fields)
    event_fields = sorted([field for field in data.dtype.names if field not in jet_fields])

    # Loop events, take up to `nleading` jets from each (@TODO: pT-ordered?)
    jets = list()
    data_events = data[event_fields]
    data_jets   = data[jet_fields]
    
    rows = list()
    for jets, event in zip(data_jets, data_events):
        for jet in np.array(jets.tolist()).T:
            row = event.copy()
            row = rfn.append_fields(row, jet_fields, jet.tolist(), usemask=False)
            rows.append(row)
            pass
        pass
        
    return np.concatenate(rows)


# Main function definition.
@profile
def main (sig_dir, bkg15_dir,bkg17_dir, bkg18_dir, treename_bkg = 'Nominal', treename_sig='AntiKt10LCTopoTrimmedPtFrac5SmallR20Jets', shuffle=False, sample=None, seed=21, replace=False, nleading=1, frac_train=0.7):
    """
    ...
    """

    # For reproducibility
    rng = np.random.RandomState(seed=seed)

    # Check(s)
    #if isinstance(sig, str):
    #    sig = [sig]
    #    pass
    #if isinstance(bkg_dir, str):
    #    bkg_dir = [bkg_dir]
    #    pass

    bkg15 = os.listdir(bkg15_dir)
    bkg17 = os.listdir(bkg17_dir)
    bkg18 = os.listdir(bkg18_dir)

    sig_dirs = os.listdir(sig_dir)
    sig = []

    for directory in sig_dirs:
        sig_files = os.listdir(sig_dir + directory)
        for filename in sig_files:
            print 'sigfile: ', filename
            sig.append(sig_dir + directory + '/' + filename)

    # Get glob'ed list of files for each category
    #sig = glob_sort_list(sig)
    #bkg = glob_sort_list(bkgFiles)

    print "here1"

    myBranches=[
        'weight',
        'event',
        'dijetmass',
        'lead_jet_ungrtrk500',
        'lead_jet_pt',
        'lead_jet_m',
        'lead_jet_eta',
        'sub_jet_ungrtrk500',
        'sub_jet_pt',
        'sub_jet_m',
        'sub_jet_eta',
        ]

    # Read in data
    selection_bkg = '(lead_jet_ungrtrk500<20 | sub_jet_ungrtrk500<20) & lead_jet_pt>500 & sub_jet_pt>200 & lead_jet_eta<2.0 & sub_jet_eta<2.0 & ((lead_jet_pt<1000 & lead_jet_m>30) | lead_jet_pt>1000) & ((sub_jet_pt<1000 & sub_jet_m>30) | sub_jet_pt>1000)'# & event%1==0 &

    selection_sig = 'lead_jet_pt>500 & sub_jet_pt>200 & lead_jet_eta<2.0 & sub_jet_eta<2.0 &((lead_jet_pt<1000 & lead_jet_m>30) | lead_jet_pt>1000) & ((sub_jet_pt<1000 & sub_jet_m>30) | sub_jet_pt>1000)'# & event%1==0 &

    kwargs_sig = dict(treename=treename_sig, branches=myBranches, selection=selection_sig)
    kwargs_bkg = dict(treename=treename_bkg, branches=myBranches, selection=selection_bkg)

#    data_sig = root_numpy.root2array(sig, **kwargs_sig)
#    data_bkg = root_numpy.root2array(bkg_dir, **kwargs_bkg)


    i=0
    for sigFile in sig:
        print sigFile
        data_sig_tmp = root_numpy.root2array(sigFile, **kwargs_sig)

        typeArray = (i+1)*np.ones((data_sig_tmp.shape[0],))

        data_sig_tmp = rfn.append_fields(data_sig_tmp, "sigType", typeArray, usemask=False)

        if i==0:
            data_sig = data_sig_tmp
        else:         
            data_sig = np.concatenate((data_sig, data_sig_tmp))
        i = i+1
    


    i=0
    for bkgFile in bkg15:
        print bkgFile
        inFile = TFile.Open(bkg15_dir + bkgFile)
        if ( inFile.GetSize()<11000 ):
            print "Too small" 
            continue

        else:
            data_bkg_tmp = root_numpy.root2array(bkg15_dir+bkgFile, **kwargs_bkg)
            data_bkg_tmp = rfn.append_fields(data_bkg_tmp, "sigType", np.zeros((data_bkg_tmp.shape[0],)), usemask=False)

            if i==0: 
                data_bkg = data_bkg_tmp
            else: 
                data_bkg = np.concatenate((data_bkg, data_bkg_tmp))


            i = i+1

    for bkgFile in bkg17:
        print bkgFile
        inFile = TFile.Open(bkg17_dir + bkgFile)

        if ( inFile.GetSize()<11000 ):
            print "Too small"
            continue

        else:
            data_bkg_tmp = root_numpy.root2array(bkg17_dir+bkgFile, **kwargs_bkg)
            data_bkg_tmp = rfn.append_fields(data_bkg_tmp, "sigType", np.zeros((data_bkg_tmp.shape[0],)), usemask=False)

            data_bkg = np.concatenate((data_bkg, data_bkg_tmp))


    for bkgFile in bkg18:
        print bkgFile
        inFile = TFile.Open(bkg18_dir + bkgFile)

        if ( inFile.GetSize()<11000 ):
            print "Too small"
            continue

        else:
            data_bkg_tmp = root_numpy.root2array(bkg18_dir+bkgFile, **kwargs_bkg)
            data_bkg_tmp = rfn.append_fields(data_bkg_tmp, "sigType", np.zeros((data_bkg_tmp.shape[0],)), usemask=False)

            data_bkg = np.concatenate((data_bkg, data_bkg_tmp))



    
    #    print data_sig[:5]

    # (Opt.) Unravel non-flat data
    #data_sig = unravel(data_sig)
    #dalead_jet_fieldta_bkg = unravel(data_bkg)

    # Append signal fields
    data_sig = rfn.append_fields(data_sig, "signal", np.ones ((data_sig.shape[0],)), usemask=False)
    data_bkg = rfn.append_fields(data_bkg, "signal", np.zeros((data_bkg.shape[0],)), usemask=False)

    # Concatenate sig and bkg arrays
    data = np.concatenate((data_sig, data_bkg))

    # (Opt.) Split leading and sub-leading jets into separate events 
    data = split(data)
    #print data.dtype.names

    # Object selection
    #msk = (data['lead_jet_pt'] > 1.) & (data['lead_jet_m'] > 1.) & (data['dijetmass'] > 1.)
    #msk = (data['jet_pt'] > 10.) & (data['jet_m'] > 10.) & (data['dijetmass'] > 10.)
    #data = data[msk]

    print "here3" 
    # Append rhoDDT field
    #data = rfn.append_fields(data, "rho", np.log(np.square(data['lead_jet_m']) / np.square(data['lead_jet_pt'])), usemask=False)
    #data = rfn.append_fields(data, "rho_ddt", np.log(np.square(data['lead_jet_m']) / data['lead_jet_pt']), usemask=False)

    # Append train field
    data = rfn.append_fields(data, "train", rng.rand(data.shape[0]) < frac_train, usemask=False)

    # (Opt.) Shuffle
    if shuffle:
        rng.shuffle(data)
        pass

    # (Opt.) Subsample
    if sample:
        data = rng.choice(data, sample, replace=replace)
        pass

    return data


# Main function call.
if __name__ == '__main__':
 #   basepath = '/afs/cern.ch/work/e/ehansen/public/DarkJetResonance/'
    basepath = '/eos/atlas/user/r/rjansky/DJR/LCTopoJets/'
    sig = basepath + 'Signal190624/' #user.rjansky.ntuple_DJ.ModelA_2000.root'
#    bkg_dir = basepath + 'bkgSamples/'
    bkg17 = basepath + 'user.rjansky.ntuple_DJ_data17_190710_calibration/'
    bkg18 = basepath + 'user.rjansky.ntuple_DJ_data18_190709_calibration/'
    bkg1516 = basepath + 'user.rjansky.ntuple_DJ_data1516_190709_calibration/'
    data = main(sig, bkg1516, bkg17, bkg18)
    with h5py.File('djr_LCTopo_data.h5', 'w') as hf:
        hf.create_dataset('dataset',  data=data, compression='gzip')
        pass

    pass
