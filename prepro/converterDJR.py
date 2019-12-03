#!/bin/env python
# -*- coding: utf-8 -*-

# Basic import(s)
import h5py
import itertools
import os
from glob import glob
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
def main (basedir, treename_bkg='Nominal', treename_sig='AntiKt10LCTopoTrimmedPtFrac5SmallR20Jets', shuffle=True, sample=None, seed=21, replace=False, nleading=1, frac_train=0.7): #'AntiKt10TrackCaloClusterTrimmedPtFrac5SmallR20Jets'
    """
    ...
    """

    # For reproducibility
    rng = np.random.RandomState(seed=seed)

    #sig_dirs = os.listdir(sig_dir)
    #bkg = os.listdir(bkg_dir)
    #bkg_dirs = os.listdir(bkg_dir)

    sampledirs = os.listdir(basedir)
    sigdir = "Signal190624" 
    bkg = []
    sig = []

    for directory in sampledirs:
        if "JZ" in directory:
            bkg_files = os.listdir(basedir + directory)
            #bkg.append(directory + '/' + bkg_files[0])
            for filename in bkg_files:
                bkg.append(directory+'/'+filename)
        if "Signal" in directory:
            sig_dirs = os.listdir(basedir + directory)
            for sig_dir in sig_dirs:
                sig_files = os.listdir(basedir + directory + '/'+ sig_dir)
                for filename in sig_files:
                    sig.append(directory+'/'+ sig_dir +'/'+filename)


    # Check(s)
    #    if isinstance(sig, str):
    #        sig = [sig]
    #        pass
    #    if isinstance(bkg, str):
    #        bkg = [bkg]
    #        pass

    # Get glob'ed list of files for each category
    #sig = glob_sort_list(sig)
    #bkg = glob_sort_list(bkg)

    print "here1"
    
    """
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
    """

    myBranches=[
        'weight',
        'dijetmass',
        'lead_jet_ungrtrk500',
        'lead_jet_pt',
        'lead_jet_EMFrac',
        'sub_jet_ungrtrk500',
        'sub_jet_pt',
        'sub_jet_EMFrac'
        ]
    
    
    # Read in data
    selection_bkg = 'weight<0.0002 & lead_jet_pt>500 & sub_jet_pt>200 & lead_jet_m>40 & sub_jet_m>40 & lead_jet_eta<2.0 & sub_jet_eta<2.0 & event%2!=0' 

    selection_sig = 'lead_jet_pt>500 & sub_jet_pt>200 & lead_jet_m>40 & sub_jet_m>40 & lead_jet_eta<2.0 & sub_jet_eta<2.0 & event%2!=0'

    kwargs_sig = dict(treename=treename_sig, branches=myBranches, selection=selection_sig)
    kwargs_bkg = dict(treename=treename_bkg, branches=myBranches, selection=selection_bkg)

    i=0
    n7events=0
    for bkgFile in bkg:
        print bkgFile
        if ".part" in bkgFile :
            print "file /afs/cern.ch/work/e/ehansen/public/DarkJetResonance/bkgSamples/user.rjansky.ntuple_DJ_JZ7W_MC16d_180904_calibration/user.rjansky.15306660.calibration._000001.root.part is truncated at 264241152 bytes: should be 643954789, skipping"
            continue
        inFile = TFile.Open(basedir + bkgFile)
        if inFile.GetSize()<11000 :
            print "Too small, skipping"
            continue

        #if i>3:
        #    continue

        else:
            data_bkg_tmp = root_numpy.root2array(basedir+bkgFile, **kwargs_bkg)
            #data_bkg_tmp = data_bkg_tmp1.astype('float32')

            print data_bkg_tmp.shape
            data_bkg_tmp = rfn.append_fields(data_bkg_tmp, "sigType", np.zeros((data_bkg_tmp.shape[0],)), usemask=False)
            
            
            if i==0:
                #data_bkg_tmp = rfn.append_fields(data_bkg_tmp, "weight_scaled",  data_bkg_tmp["weight"], usemask=False)
                data_bkg = data_bkg_tmp
            
                #elif "JZ7W" in bkgFile : 
                #data_bkg_tmp = rfn.append_fields(data_bkg_tmp, "weight_scaled",  data_bkg_tmp["weight"]*15040000/11760000., usemask=False)
                #n7events += data_bkg_tmp.shape[0]
                #data_bkg = np.concatenate((data_bkg, data_bkg_tmp))

            else:
                #data_bkg_tmp = rfn.append_fields(data_bkg_tmp, "weight_scaled",  data_bkg_tmp["weight"], usemask=False)
                data_bkg = np.concatenate((data_bkg, data_bkg_tmp))
            i = i+1

    print "Number of slice 7 events available: ", n7events 

    i=0
    for sigFile in sig:
        print sigFile

        #if i>3:
        #    continue
        data_sig_tmp = root_numpy.root2array(basedir+sigFile, **kwargs_sig)
        #data_sig_tmp = data_sig_tmp1.astype('float32')

        typeArray = (i+1)*np.ones((data_sig_tmp.shape[0],))
        #weightArray = np.ones((data_sig_tmp.shape[0],))

        data_sig_tmp = rfn.append_fields(data_sig_tmp, "sigType", typeArray, usemask=False)
        #data_sig_tmp = rfn.append_fields(data_sig_tmp, "weight_scaled", weightArray, usemask=False)

        if i==0:
            data_sig = data_sig_tmp
        else:
            #print data_sig_tmp
            data_sig = np.concatenate((data_sig, data_sig_tmp))
        i = i+1


#    data_sig = root_numpy.root2array(sig, **kwargs_sig)
#    data_bkg = root_numpy.root2array(bkg, **kwargs_bkg)
                           
#    print data_sig[:5]

    # (Opt.) Unravel non-flat data
    #data_sig = unravel(data_sig)
    #dajet_fieldta_bkg = unravel(data_bkg)

    # Append signal fields
    data_sig = rfn.append_fields(data_sig, "signal", np.ones ((data_sig.shape[0],)), usemask=False)
    data_bkg = rfn.append_fields(data_bkg, "signal", np.zeros((data_bkg.shape[0],)), usemask=False)

    # Concatenate sig and bkg arrays

    data = np.concatenate((data_sig, data_bkg)) # gives a memory error when using the full MC sample

    # Failed attempt to work around the memory error
    """
    Nentries = data_sig.shape[0] + data_bkg.shape[0]
    Nbranches = len(myBranches)

    print data_sig.shape, data_bkg.shape, len(myBranches)

    data = np.empty((Nentries, 11))
    
    columns = list(data_sig)
    print "column: ", columns[0]

    for indexS, row in enumerate(data_sig):
        data[indexS] = row
        #print "row: ", row

    print "indexS", indexS
    indexS += 1
    for indexB, row in enumerate(data_bkg):
        data[indexS+indexB] = row

    print "data.shape: ", data.shape

    #print "concat data.shape: ", data2.shape
    """

    # (Opt.) Split leading and sub-leading jets into separate events 
    #data = split(data)

    # Object selection
    #msk = (data['lead_jet_pt'] > 20.) & (data['lead_jet_m'] > 10.) & (data['dijetmass'] > 10.)
    #msk = (data['jet_pt'] > 10.) & (data['jet_m'] > 10.) & (data['dijetmass'] > 10.)
    #data = data[msk]


    # Append train field
    #data = rfn.append_fields(data, "train", rng.rand(data.shape[0]) < frac_train, usemask=False)

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
    basepath = '/eos/atlas/user/r/rjansky/DJR/LCTopoJets/' #'/eos/user/e/ehansen/DJR/'
#    basepath = '/afs/cern.ch/work/e/ehansen/public/DarkJetResonance/'
    sig = basepath + 'Signal190624/' #user.rjansky.ntuple_DJ.ModelA_2000.root'
    bkg = basepath + 'user.rjansky.ntuple_DJ_JZ*/'
#   bkg = '/eos/atlas/user/r/rjansky/DJR/ntuple_DJ_JZ06W.root' #*.root'
#    bkg = '/afs/cern.ch/work/e/ehansen/public/DDT-studies-FourJets/jetjet/*.root'
    data = main(basepath)#sig, bkg)
    with h5py.File('djr_LCTopo_1.h5', 'w') as hf:
        hf.create_dataset('dataset',  data=data, compression='gzip')
        pass

    pass
