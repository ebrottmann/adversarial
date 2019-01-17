#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Basic import(s)
import h5py
import itertools
from glob import glob

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
    name = name.replace('lead_jet_', '')
    return name

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
def main (sig, bkg, treename='AntiKt10TrackCaloClusterTrimmedPtFrac5SmallR20Jets', shuffle=False, sample=None, seed=21, replace=False, nleading=1, frac_train=0.8):
    """
    ...
    """

    # For reproducibility
    rng = np.random.RandomState(seed=seed)

    # Check(s)
    if isinstance(sig, str):
        sig = [sig]
        pass
    if isinstance(bkg, str):
        bkg = [bkg]
        pass

    # Get glob'ed list of files for each category
    sig = glob_sort_list(sig)
    bkg = glob_sort_list(bkg)

    print "here1"

    # Read in data
    branches = None
    selection = 'event%2==0 & weight<0.0001'
# & (data['weight'] < 0.1)
    kwargs_sig = dict(treename=treename, branches=branches) #, selection=selection)
    kwargs_bkg = dict(treename=treename, branches=branches, selection=selection)

    data_sig = root_numpy.root2array(sig, **kwargs_sig)
    data_bkg = root_numpy.root2array(bkg, **kwargs_bkg)

    print "here2" 

    # (Opt.) Unravel non-flat data
    data_sig = unravel(data_sig)
    data_bkg = unravel(data_bkg)

    # Append signal fields
    data_sig = rfn.append_fields(data_sig, "signal", np.ones ((data_sig.shape[0],)), usemask=False)
    data_bkg = rfn.append_fields(data_bkg, "signal", np.zeros((data_bkg.shape[0],)), usemask=False)

    # Concatenate arrays
    data = np.concatenate((data_sig, data_bkg))

    # Rename columns
    #data.dtype.names = map(rename, data.dtype.names)

    # Object selection
    msk = (data['lead_jet_pt'] > 10.) & (data['lead_jet_m'] > 10.) & (data['dijetmass'] > 10.)
    data = data[msk]

    print "here3" 
    # Append rhoDDT field
    data = rfn.append_fields(data, "rho", np.log(np.square(data['lead_jet_m']) / np.square(data['lead_jet_pt'])), usemask=False)
    data = rfn.append_fields(data, "rho_ddt", np.log(np.square(data['lead_jet_m']) / data['lead_jet_pt']), usemask=False)

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
    basepath = '/afs/cern.ch/work/e/ehansen/public/DarkJetResonance/'
    sig = basepath + 'sigSamples/user.rjansky.ntuple_DJ.ModelA_2000.root'
    bkg = basepath + 'bkgSamples/*.root'
#    bkg = '/afs/cern.ch/work/e/ehansen/public/DDT-studies-FourJets/jetjet/*.root'
    data = main(sig, bkg)
    with h5py.File('djr_A_2000.h5', 'w') as hf:
        hf.create_dataset('dataset',  data=data, compression='gzip')
        pass

    pass
