#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for performing training (and evaluation?) of adversarial neural networks for de-correlated jet tagging."""

# Basic import(s)
import os
import sys
import gzip
import glob
import json
import pickle
import datetime
import subprocess
from pprint import pprint
import logging as log
import itertools

# Get ROOT to stop hogging the command-line options
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

# Scientific import(s)
import numpy as np
seed = 21 # For reproducibility
np.random.seed(seed)
import root_numpy

import sklearn
from sklearn.model_selection import StratifiedKFold

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#plt.switch_backend('pdf')

# -- Explicitly ignore DeprecationWarning from scikit-learn, which we can't do
#    anything about anyway.
stderr = sys.stderr
with open(os.devnull, 'w') as sys.stderr:
    from hep_ml.reweight import GBReweighter, BinsReweighter
    pass
sys.stderr = stderr

# Project import(s)
import adversarial
from adversarial.data    import *
from adversarial.utils   import *
from adversarial.profile import *
from adversarial.plots   import *

# Global variables
PROJECTDIR='/'.join(os.path.realpath(adversarial.__file__).split('/')[:-2] + [''])


# Command-line arguments parser
import argparse

def parse_args (cmdline_args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Perform training (and evaluation?) of adversarial neural networks for de-correlated jet tagging.")

    # -- Inputs
    parser.add_argument('-i', '--input',  dest='input',   action='store', type=str,
                        default=PROJECTDIR + 'data/', help='Input directory, from which to read HDF5 data file.')
    parser.add_argument('-o', '--output', dest='output',  action='store', type=str,
                        default=PROJECTDIR + 'output/', help='Output directory, to which to write results.')
    parser.add_argument('-c', '--config', dest='config',  action='store', type=str,
                        default=PROJECTDIR + 'configs/default.json', help='Configuration file.')
    parser.add_argument('-p', '--patch', dest='patches', action='append', type=str,
                        help='Patch file(s) with which to update configuration file.')
    parser.add_argument('--devices',     dest='devices', action='store', type=int,
                        default=1, help='Number of CPU/GPU devices to use with TensorFlow.')
    parser.add_argument('--folds',       dest='folds',    action='store', type=int,
                        default=2, help='Number of folds to use for stratified cross-validation.')
    parser.add_argument('--jobname',     dest='jobname',  action='store', type=str,
                        default="", help='Name of job, used for TensorBoard output.')

    # -- Flags
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_const',
                        const=True, default=False, help='Print verbose')
    parser.add_argument('-g', '--gpu',  dest='gpu',        action='store_const',
                        const=True, default=False, help='Run on GPU')
    parser.add_argument('--tensorflow', dest='tensorflow', action='store_const',
                        const=True, default=False, help='Use TensorFlow backend')
    parser.add_argument('--train', dest='train', action='store_const',
                        const=True, default=False, help='Perform training')
    parser.add_argument('--train-classifier', dest='train_classifier', action='store_const',
                        const=True, default=False, help='Perform classifier pre-training')
    parser.add_argument('--train-adversarial', dest='train_adversarial', action='store_const',
                        const=True, default=False, help='Perform adversarial training')
    parser.add_argument('--optimise-classifier', dest='optimise_classifier', action='store_const',
                        const=True, default=False, help='Optimise stand-alone classifier')
    parser.add_argument('--optimise-adversarial', dest='optimise_adversarial', action='store_const',
                        const=True, default=False, help='Optimise adversarial network')
    parser.add_argument('--plot', dest='plot', action='store_const',
                        const=True, default=False, help='Perform plotting')
    parser.add_argument('--tensorboard', dest='tensorboard', action='store_const',
                        const=True, default=False, help='Use TensorBoard for monitoring')

    return parser.parse_args(cmdline_args)


# Main function definition
@profile
def main (args):

    # Initialisation
    # --------------------------------------------------------------------------
    with Profile("Initialisation"):

        # Add 'mode' field manually
        args = argparse.Namespace(mode='gpu' if args.gpu else 'cpu', **vars(args))

        # Set print level
        log.basicConfig(format="%(levelname)s: %(message)s",
                        level=log.DEBUG if args.verbose else log.INFO)

        # Create common colour array
        colours = map(lambda d: d['color'], list(plt.rcParams["axes.prop_cycle"]))

        #  Modify input/output directory names to conform to convention
        if not args.input .endswith('/'): args.input  += '/'
        if not args.output.endswith('/'): args.output += '/'

        # Make sure output directory exists
        if not os.path.exists(os.path.realpath(args.output)):
            print "Creating output directory:\n  {}".format(args.output)
            try:
                os.makedirs(args.output)
            except OSError:
                print "Apparently, {} ({}) already exists...".format(args.output, os.path.realpath(args.output))
                pass
            pass

        # Validate train/optimise flags
        if args.optimise_classifier and args.optimise_adversarial:
            raise Exception(("Cannot optimise stand-alone classifier and "
                             "adversarial network simultaneously."))
        elif args.optimise_classifier:

            # Stand-alone classifier optimisation
            if not args.train_classifier:
                log.warning("Setting `train_classifier` to True.")
                args.train_classifier = True
                pass
            if args.train_adversarial:
                log.warning("Setting `train_adversarial` to False.")
                args.train_adversarial = False
                pass
            if args.train:
                log.warning("Setting `train` to False.")
                args.train = False
                pass

        elif args.optimise_adversarial:

            # Adversarial network optimisation
            if args.train_classifier:
                log.warning("Setting `train_classifier` to False.")
                args.train_classifier = False
                pass
            if not args.train_adversarial:
                log.warning("Setting `train_adversarial` to True.")
                args.train_adversarial = True
                pass
            if args.train:
                log.warning("Setting `train` to False.")
                args.train = False
                pass

            pass

        # @TODO:
        # - Make `args = prepare_args  (args)` method?
        # - Make `cfg  = prepare_config(args)` method?

        # Load configuration file
        with open(args.config, 'r') as f:
            cfg = json.load(f)
            pass

        # Apply patches
        if args.patches is not None:
            for patch_file in args.patches:
                log.info("Applying patch '{}'".format(patch_file))
                with open(patch_file, 'r') as f:
                    patch = json.load(f)
                    pass
                apply_patch(cfg, patch)
                pass
            pass

        # Set adversary learning rate (LR) ratio from ratio of loss_weights
        cfg['combined']['model']['lr_ratio'] = cfg['combined']['compile']['loss_weights'][0] / \
                                               cfg['combined']['compile']['loss_weights'][1]

        # Initialise Keras backend
        initialise_backend(args)

        import keras
        import keras.backend as K
        from keras.models import load_model
        from keras.callbacks import Callback, TensorBoard, EarlyStopping
        KERAS_VERSION=int(keras.__version__.split('.')[0])
        if KERAS_VERSION == 2:
            from keras.utils.vis_utils import plot_model
        else:
            from keras.utils.visualize_util import plot as plot_model
            pass

        # Print setup information
        log.info("Running '%s'" % __file__)
        log.info("Command-line arguments:")
        pprint(vars(args))

        log.info("Configuration file contents:")
        pprint(cfg)

        log.info("Python version: {}".format(sys.version.split()[0]))
        log.info("Numpy  version: {}".format(np.__version__))
        try:
            log.info("Keras  version: {}".format(keras.__version__))
            log.info("Using keras backend: '{}'".format(K.backend()))
            if K.backend() == 'tensorflow':
                import tensorflow
                print "  TensorFlow version: {}".format(tensorflow.__version__)
            else:
                import theano
                print "  Theano version: {}".format(theano.__version__)
                pass
        except NameError: log.info("Keras not imported")

        # Save command-line argument configuration in output directory
        with open(args.output + 'args.json', 'wb') as f:
            json.dump(vars(args), f, indent=4, sort_keys=True)
            pass

        # Save configuration dict in output directory
        with open(args.output + 'config.json', 'wb') as f:
            json.dump(cfg, f, indent=4, sort_keys=True)
            pass

        # Evaluate the 'optimizer' fields for each model, once and for all
        for model in ['classifier', 'combined']:
            opts = cfg[model]['compile']
            opts['optimizer'] = eval("keras.optimizers.{}(lr={}, decay={})" \
                                     .format(opts['optimizer'],
                                             opts.pop('lr'),
                                             opts.pop('decay')))
            pass

        # If the `model/architecture` parameter is provided as an int, convert
        # to list of empty dicts
        for network in ['classifier', 'adversary']:
            if isinstance(cfg[network]['model']['architecture'], int):
                cfg[network]['model']['architecture'] = [{} for _ in range(cfg[network]['model']['architecture'])]
                pass
            pass

        # Set keras.Model.fit.verbose flag to 2 for optimisation
        if args.optimise_classifier:
            cfg['classifier']['fit']['verbose'] = 2
            pass

        if args.optimise_adversarial:
            cfg['combined']['fit']['verbose'] = 2
            pass

        # Start TensorBoard instance
        if args.tensorflow:
            tensorboard_dir = PROJECTDIR + 'logs/tensorboard/{}/'.format('-'.join(re.split('-|:| ', str(datetime.datetime.now()).replace('.', 'T'))) if args.jobname == "" else args.jobname)
            log.info("Writing TensorBoard logs to '{}'".format(tensorboard_dir))
            if args.tensorboard:
                assert args.tensorflow, "TensorBoard requires TensorFlow backend."

                log.info("Starting TensorBoard instance in background.")
                log.info("The output will be available at:")
                log.info("  http://localhost:6006")
                tensorboard_pid = subprocess.Popen(["tensorboard", "--logdir", tensorboard_dir]).pid
                log.info("TensorBoard has PID {}.".format(tensorboard_pid))
                pass
            pass

        pass


    # Loading data
    # --------------------------------------------------------------------------
    with Profile("Loading data"):

        data = load_data(args.input + 'data.h5')
        data = prepare_data(data)
        data.shuffle(seed=21)
        data.split(train=0.8, test=0.2)

        num_samples, num_features = data.train.inputs.shape
        pass


    # Re-weighting to flatness
    # --------------------------------------------------------------------------
    """
    with Profile("Re-weighting"):
        # @NOTE: This is the crucial point: If the target is flat in (m,pt) the
        # re-weighted background _won't_ be flat in (log m, log pt), and vice
        # versa. It should go without saying, but draw target samples from a
        # uniform prior on the coordinates which are used for the decorrelation.

        decorrelation_variables = ['m']#, 'pt']

        # Performing pre-processing of de-correlation coordinates
        with Profile():
            log.debug("Performing pre-processing")

            # Get number of background events and number of target events (arb.)
            N_sig = len(sig)
            N_bkg = len(bkg)
            N_tar = len(bkg)

            # Initialise and fill coordinate arrays
            P_sig = np.zeros((N_sig, len(decorrelation_variables)), dtype=float)
            P_bkg = np.zeros((N_bkg, len(decorrelation_variables)), dtype=float)
            for col, var in enumerate(decorrelation_variables):
                P_sig[:,col] = np.log(sig[var])
                P_bkg[:,col] = np.log(bkg[var])
                pass
            #P_sig[:,1] = np.log(sig['pt'])
            #P_bkg[:,1] = np.log(bkg['pt'])
            P_tar = np.random.rand(N_tar, len(decorrelation_variables))

            # Scale coordinates to range [0,1]
            log.debug("Scaling background coordinates to range [0,1]")
            P_sig -= np.min(P_sig, axis=0)
            P_bkg -= np.min(P_bkg, axis=0)
            P_sig /= np.max(P_sig, axis=0)
            P_bkg /= np.max(P_bkg, axis=0)
            log.debug("  Min (sig):", np.min(P_sig, axis=0))
            log.debug("  Max (sig):", np.max(P_sig, axis=0))
            log.debug("  Min (bkg):", np.min(P_bkg, axis=0))
            log.debug("  Max (bkg):", np.max(P_bkg, axis=0))
            pass

        # Fit, or load, regressor to achieve flatness using hep_ml library
        with Profile():
            log.debug("Performing re-weighting using GBReweighter")
            reweighter_filename = 'trained/reweighter_{}d.pkl.gz'.format(len(decorrelation_variables))
            if not os.path.isfile(reweighter_filename):
                reweighter = GBReweighter(n_estimators=80, max_depth=7)
                reweighter.fit(P_bkg, target=P_tar, original_weight=bkg['weight'])
                log.info("Saving re-weighting object to file '%s'" % reweighter_filename)
                with gzip.open(reweighter_filename, 'wb') as f:
                    pickle.dump(reweighter, f)
                    pass
            else:
                log.info("Loading re-weighting object from file '%s'" % reweighter_filename)
                with gzip.open(reweighter_filename, 'r') as f:
                    reweighter = pickle.load(f)
                    pass
                pass
            pass

        # Re-weight for uniform prior(s)
        with Profile():
            log.debug("Getting new weights for uniform prior(s)")
            new_weights  = reweighter.predict_weights(P_bkg, original_weight=bkg['weight'])
            new_weights *= np.sum(bkg['weight']) / np.sum(new_weights)
            bkg = append_fields(bkg, 'reweight', new_weights, dtypes=K.floatx())

            # Appending similary ("dummy") 'reweight' field to signal sample, for consistency
            sig = append_fields(sig, 'reweight', sig['weight'], dtypes=K.floatx())
            pass

        pass


    # Plotting: Re-weighting
    # --------------------------------------------------------------------------
    with Profile("Plotting: Re-weighting"):


        fig, ax = plt.subplots(2, 4, figsize=(12,6))

        w_bkg  = bkg['weight']
        rw_bkg = bkg['reweight']
        w_tar  = np.ones((N_tar,)) * np.sum(bkg['weight']) / float(N_tar)

        for row, var in enumerate(decorrelation_variables):
            edges = np.linspace(0, np.max(bkg[var]), 60 + 1, endpoint=True)
            nbins  = len(edges) - 1

            v_bkg  = bkg[var]     # Background  mass/pt values for the background
            rv_bkg = P_bkg[:,row] # Transformed mass/pt values for the background
            rv_tar = P_tar[:,row] # Transformed mass/pt values for the targer

            ax[row,0].hist(v_bkg,  bins=edges, weights=w_bkg,  alpha=0.5, label='Background')
            ax[row,1].hist(v_bkg,  bins=edges, weights=rw_bkg, alpha=0.5, label='Background')
            ax[row,2].hist(rv_bkg, bins=nbins, weights=w_bkg,  alpha=0.5, label='Background') # =rw_bkg
            ax[row,2].hist(rv_tar, bins=nbins, weights=w_tar,  alpha=0.5, label='Target')
            ax[row,3].hist(rv_bkg, bins=nbins, weights=rw_bkg, alpha=0.5, label='Background')
            ax[row,3].hist(rv_tar, bins=nbins, weights=w_tar,  alpha=0.5, label='Target')

            for col in range(4):
                if col < 4: # 3
                    ax[row,col].set_yscale('log')
                    ax[row,col].set_ylim(1E+01, 1E+06)
                    if row == 1:
                        ax[row,col].set_ylim(1E-01, 1E+05)
                        pass
                    pass
                ax[row,col].set_xlabel("Jet %s%s%s" % (var, " (transformed)" if col > 1 else '', " (re-weighted)" if (col + 1) % 2 == 0 else ''))
                if col == 0:
                    ax[row,col].set_ylabel("Jets / {:.1f} GeV".format(np.diff(edges)[0]))
                    pass
                pass
            pass

        plt.legend()
        plt.savefig(args.output + 'priors_1d.pdf')

        # Plot 2D prior before and after re-weighting
        if len(decorrelation_variables) == 2:
            log.debug("Plotting 2D prior before and after re-weighting")
            fig, ax = plt.subplots(1,2,figsize=(11,5), sharex=True, sharey=True)
            h = ax[0].hist2d(P_bkg[:,0], P_bkg[:,1], bins=40, weights=bkg['weight'],   vmin=0, vmax=5, normed=True)
            h = ax[1].hist2d(P_bkg[:,0], P_bkg[:,1], bins=40, weights=bkg['reweight'], vmin=0, vmax=5, normed=True)
            ax[0].set_xlabel("Scaled log(m)")
            ax[1].set_xlabel("Scaled log(m)")
            ax[0].set_ylabel("Scaled log(pt)")

            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes([0.925, 0.15, 0.025, 0.7])
            fig.colorbar(h[3], cax=cbar_ax)
            plt.savefig(args.output + 'priors_2d.pdf')
            pass
        pass
        """


    # Classifier-only fit
    # --------------------------------------------------------------------------
    # Adapted from: https://github.com/asogaard/AdversarialSubstructure/blob/master/train.py
    # Resources:
    #  [https://github.com/fchollet/keras/issues/7515]
    #  [https://stackoverflow.com/questions/43821786/data-parallelism-in-keras]
    #  [https://stackoverflow.com/a/44771313]

    with Profile("Classifier-only fit, cross-validation"):
        # @TODO:
        # - Implement checkpointing (?)
        # - Implement data generator looping over all of the background and
        # randomly sampling signal events to have equal fractions in each
        # batch. Use DataFlow from Tensorpack?

        # Define variables
        basename = 'crossval_classifier'

        # Get indices for each fold in stratified k-fold training
        # @NOTE: No shuffling is performed -- assuming that's already done above.
        skf_instance = StratifiedKFold(n_splits=args.folds)
        skf = skf_instance.split(data.train.inputs, data.train.targets)

        # Importe module creator methods and optimiser options
        from adversarial.models import classifier_model, adversary_model, combined_model

        # Create unique set of random indices to use with stratification
        random_indices = np.arange(num_samples)
        np.random.shuffle(random_indices)

        # Collection of classifiers and their associated training histories
        classifiers = list()
        histories   = list()

        # Train or load classifiers
        if args.train or args.train_classifier:
            log.info("Training cross-validation classifiers")

            # Loop `k` folds
            try:
                for fold, (train, validation) in enumerate(skf):
                    with Profile("Fold {}/{}".format(fold + 1, args.folds)):

                        # StratifiedKFold provides stratification, but since the
                        # input arrays are not randomised, neither will the
                        # folds. Therefore, the fold should be taken with respect to
                        # a set of randomised indices rather than range(N).
                        train      = random_indices[train]
                        validation = random_indices[validation]

                        # Define unique tag and name for current classifier
                        tag  = '{}of{}'.format(fold + 1, args.folds)
                        name = '{}__{}'.format(basename, tag)

                        # Get classifier
                        classifier = classifier_model(num_features, **cfg['classifier']['model'])

                        # Compile model (necessary to save properly)
                        classifier.compile(**cfg['classifier']['compile'])

                        # Create callbacks
                        callbacks = []

                        # -- Early stopping
                        #callbacks += [EarlyStopping(patience=10)]  # @NOTE: Problem for finding global minimum...

                        # -- TensorBoard
                        if args.tensorflow:
                            callbacks += [TensorBoard(log_dir=tensorboard_dir + 'classifier/fold{}/'.format(fold))]
                            pass

                        # Fit classifier model
                        result = train_in_parallel(classifier,
                                                   {'input':   data.train.inputs,
                                                    'target':  data.train.targets,
                                                    'weights': data.train.weights,
                                                    'mask':    train},
                                                   {'input':   data.train.inputs,
                                                    'target':  data.train.targets,
                                                    'weights': data.train.weights,
                                                    'mask':    validation},
                                                   config=cfg['classifier'],
                                                   num_devices=args.devices,
                                                   mode=args.mode,
                                                   seed=seed,
                                                   callbacks=callbacks)

                        histories.append(result['history'])

                        # Save classifier model and training history to file, both
                        # in unique output directory and in the directory for
                        # pre-trained classifiers
                        for destination in [args.output, PROJECTDIR + 'trained/']:
                            classifier.save        (destination + '{}.h5'        .format(name))
                            classifier.save_weights(destination + '{}_weights.h5'.format(name))
                            with open(destination + 'history__{}.json'.format(name), 'wb') as f:
                                json.dump(result['history'], f)
                                pass
                            pass

                        # Add to list of classifiers
                        classifiers.append(classifier)
                        pass
                    pass
                pass
            except KeyboardInterrupt:
                log.warning("Training was stopped early.")
                pass
        else:
            log.info("Loading cross-validation classifiers from file")

            # Load pre-trained classifiers
            classifier_files = sorted(glob.glob(PROJECTDIR + 'trained/{}__*of{}.h5'.format(basename, args.folds)))
            assert len(classifier_files) == args.folds, "Number of pre-trained classifiers ({}) does not match number of requested folds ({})".format(len(classifier_files), args.folds)
            for classifier_file in classifier_files:
                classifiers.append(load_model(classifier_file))
                pass

            # Load associated training histories
            history_files = sorted(glob.glob(PROJECTDIR + 'trained/history__{}__*of{}.json'.format(basename, args.folds)))
            assert len(history_files) == args.folds, "Number of training histories for pre-trained classifiers ({}) does not match number of requested folds ({})".format(len(history_files), args.folds)
            for history_file in history_files:
                with open(history_file, 'r') as f:
                    histories.append(json.load(f))
                    pass
                pass

            pass # end: train/load
        pass


    # Get optimal number of training epochs
    # --------------------------------------------------------------------------
    if args.train or args.train_classifier:
        epochs = 1 + np.arange(len(histories[0]['loss']))
        val_avg = np.mean([hist['val_loss'] for hist in histories], axis=0)
        opt_epochs = epochs[np.argmin(val_avg)]

        log.info("Using optimal number of {:d} training epochs".format(opt_epochs))
        pass


    # Early stopping in case of stand-alone classifier optimisation
    # --------------------------------------------------------------------------
    if args.optimise_classifier:
        return np.min(val_avg)


    # Plotting: Cost log for classifier-only fit
    # --------------------------------------------------------------------------
    if args.plot and (args.train or args.train_classifier):
        with Profile("Plotting: Cost log, cross-val."):

            # Perform plotting
            fig, ax = plt.subplots()

            for fold, hist in enumerate(histories):
                plt.plot(epochs, hist['val_loss'], color=colours[1], linewidth=0.6, alpha=0.3,
                         label='Validation (fold)' if fold == 0 else None)
                pass

            plt.plot(epochs, val_avg,   color=colours[1], label='Validation (avg.)')

            for fold, hist in enumerate(histories):
                plt.plot(epochs, hist['loss'],     color=colours[0], linewidth=1.0, alpha=0.3,
                         label='Training (fold)'   if fold == 0 else None)
                pass

            train_avg = np.mean([hist['loss'] for hist in histories], axis=0)
            plt.plot(epochs, train_avg, color=colours[0], label='Train (avg.)')

            plt.title('Classifier-only, stratified {}-fold training'.format(args.folds), fontweight='medium')
            plt.xlabel("Training epochs",    horizontalalignment='right', x=1.0)
            plt.ylabel("Objective function", horizontalalignment='right', y=1.0)

            epochs = [0] + list(epochs)
            step = max(int(np.floor(len(epochs) / 10.)), 1)

            plt.xticks(filter(lambda x: x % step == 0, epochs))
            plt.legend()

            plt.savefig(args.output + 'costlog_classifier.pdf')
            pass
        pass


    # Classifier-only fit, full
    # --------------------------------------------------------------------------
    with Profile("Classifier-only fit, full"):

        # Define variables
        name = 'full_classifier'

        if args.train or args.train_classifier:
            log.info("Training full classifier")

            # Get classifier
            classifier = classifier_model(num_features, **cfg['classifier']['model'])

            # Compile model (necessary to save properly)
            classifier.compile(**cfg['classifier']['compile'])

            # Overwrite number of training epochs with optimal number found from
            # cross-validation
            cfg['classifier']['fit']['epochs'] = opt_epochs

            # Create callbacks
            callbacks = []

            # -- TensorBoard
            if args.tensorflow:
                callbacks += [TensorBoard(log_dir=tensorboard_dir + 'classifier/')]
                pass

            # Train final classifier
            try:
                result = train_in_parallel(classifier,
                                           {'input':   data.train.inputs,
                                            'target':  data.train.targets,
                                            'weights': data.train.weights},
                                           config=cfg['classifier'],
                                           mode=args.mode,
                                           num_devices=args.devices,
                                           seed=seed,
                                           callbacks=callbacks)
            except KeyboardInterrupt:
                log.warning("Training was stopped early.")
                pass

            # Save classifier model and training history to file, both
            # in unique output directory and in the directory for
            # pre-trained classifiers
            for destination in [args.output, PROJECTDIR + 'trained/']:
                classifier.save        (destination + '{}.h5'        .format(name))
                classifier.save_weights(destination + '{}_weights.h5'.format(name))
                with open(destination + 'history__{}.json'.format(name), 'wb') as f:
                    json.dump(result['history'], f)
                    pass
                pass

        else:

            log.info("Loading full classifier from file")

            # Load pre-trained classifiers
            classifier_file = PROJECTDIR + 'trained/{}.h5'.format(name)
            classifier = load_model(classifier_file)

            # Load associated training histories
            history_file = PROJECTDIR + 'trained/history__{}.json'.format(name)
            with open(history_file, 'r') as f:
                history = json.load(f)
                pass

            pass # end: train/load

        # Save classifier model diagram to file
        plot_model(classifier, to_file=args.output + 'model_classifier.png', show_shapes=True)

        # Store classifier output as tagger variable.
        data.add_field('NN', classifier.predict(data.inputs, batch_size=2048 * 8).flatten().astype(K.floatx()))
        pass


    # Saving classifier in lwtnn-friendly format.
    # --------------------------------------------------------------------------
    def lwtnn_save(model, name, basedir=PROJECTDIR + 'trained/lwtnn/'):
        """Method for saving classifier in lwtnn-friendly format.
        See [https://github.com/lwtnn/lwtnn/wiki/Keras-Converter]
        """
        # Check(s).
        if not basedir.endswith('/'):
            basedir += '/'
            pass

        # Make sure output directory exists
        if not os.path.exists(basedir):
            print "Creating output directory:\n  {}".format(basedir)
            os.makedirs(basedir)
            pass


        # Get the architecture as a json string
        arch = model.to_json()

        # Save the architecture string to a file
        with open(basedir + name + '_architecture.json', 'w') as arch_file:
            arch_file.write(arch)
            pass

        # Now save the weights as an HDF5 file
        model.save_weights(basedir + name + '_weights.h5')

        # Save full model to HDF5 file
        model.save(basedir + name + '.h5')
        return

    lwtnn_save(classifier, 'nn')


    # Plotting ROCs (only NN)
    # --------------------------------------------------------------------------

    # Tagger variables
    variables = ['Tau21', 'D2', 'NN']

    if args.plot:
        with Profile("Plotting: ROCs (only NN)"):
            plot_roc(data.test, args, variables, name='tagger_ROCs_NN')
            pass
        pass


    # Training callbacks
    # --------------------------------------------------------------------------
    # @TODO:
    # - Move to `adversarial/callbacks.py`?

    class PosteriorCallback (Callback):
        def __init__ (self, data, args, adversary):
            self.opts = dict(data=data, args=args, adversary=adversary)
            return

        def on_train_begin (self, logs={}):
            plot_posterior(name='posterior_begin', title="Beginning of training", **self.opts)
            return

        def on_epoch_end (self, epoch, logs={}):
            plot_posterior(name='posterior_epoch_{:03d}'.format(epoch + 1), title="Epoch {}".format(epoch + 1), **self.opts)
            return
        pass


    class ProfilesCallback (Callback):
        def __init__ (self, data, args, var):
            self.opts = dict(data=data, args=args, var=var)
            return

        def on_train_begin (self, logs={}):
            plot_profiles(name='profiles_begin', title="Beginning of training", **self.opts)
            return

        def on_epoch_end (self, epoch, logs={}):
            plot_profiles(name='profiles_epoch_{:03d}'.format(epoch + 1), title="Epoch {}".format(epoch + 1), **self.opts)
            return
        pass


    # Combined fit, full (@TODO: Cross-val?)
    # --------------------------------------------------------------------------
    with Profile("Combined fit, full"):
        # @TODO:
        # - Checkpointing

        # Define variables
        name = 'full_combined'

        # Set up adversary
        adversary = adversary_model(gmm_dimensions=data.decorrelation.shape[1],
                                    **cfg['adversary']['model'])

        # Save adversarial model diagram
        plot_model(adversary, to_file=args.output + 'model_adversary.png', show_shapes=True)

        # Create callback array
        callbacks = list()

        # Create callback logging the adversary p.d.f.'s during training
        callback_posterior = PosteriorCallback(data.test, args, adversary)

        # Create callback logging the adversary p.d.f.'s during training
        callback_profiles  = ProfilesCallback(data.test, args, classifier)

        # (opt.) List all callbacks to be used
        if args.plot:
            callbacks += [callback_posterior, callback_profiles]
            pass

        # (opt.) Add TensorBoard callback
        if args.tensorflow:
            callbacks += [TensorBoard(log_dir=tensorboard_dir + 'adversarial/')]
            pass

        # Set up combined, adversarial model
        combined = combined_model(classifier,
                                  adversary,
                                  **cfg['combined']['model'])

        # Save combiend model diagram
        plot_model(combined, to_file=args.output + 'model_combined.png', show_shapes=True)

        if args.train or args.train_adversarial:
            log.info("Training full, combined model")

            # Create custom objective function for posterior: - log(p) of the
            # posterior p.d.f. This corresponds to binary cross-entropy for 1.
            def maximise (p_true, p_pred):
                return - K.log(p_pred)

            cfg['combined']['compile']['loss'][1] = maximise

            # Compile model (necessary to save properly)
            combined.compile(**cfg['combined']['compile'])

            # Train final classifier
            try:
                result = train_in_parallel(combined,
                                           {'input':   [data.inputs,  data.decorrelation],
                                            'target':  [data.targets, np.ones_like(data.targets)],
                                            'weights': [data.weights, np.multiply(data.weights, 1 - data.targets)]},
                                           # @TODO:
                                           # - Try to use [data.weights, data.weights_flat * ...]
                                           config=cfg['combined'],
                                           mode=args.mode,
                                           num_devices=args.devices,
                                           seed=seed,
                                           callbacks=callbacks)
            except KeyboardInterrupt:
                log.warning("Training was stopped early.")
                pass

            # Save combined model and training history to file, both
            # in unique output directory and in the directory for
            # pre-trained classifiers
            history = result['history']
            for destination in [args.output, PROJECTDIR + 'trained/']:
                combined.save        (destination + '{}.h5'        .format(name))
                combined.save_weights(destination + '{}_weights.h5'.format(name))
                with open(destination + 'history__{}.json'.format(name), 'wb') as f:
                    json.dump(history, f)
                    pass
                pass

        else:

            log.info("Loading full, combined model from file")

            # Improt GradientReversalLayerto allow reading of model
            from adversarial.layers import GradientReversalLayer, PosteriorLayer

            # Load pre-trained combined _weights_ from file, in order to
            # simultaneously load the embedded classifier so as to not have to
            # extract it manually afterwards.
            combined_weights_file = PROJECTDIR + 'trained/{}_weights.h5'.format(name)
            combined.load_weights(combined_weights_file)

            # Load associated training histories
            history_file = PROJECTDIR + 'trained/history__{}.json'.format(name)
            with open(history_file, 'r') as f:
                history = json.load(f)
                pass

            pass

        # Store classifier output as tagger variable.
        data.add_field('ANN', classifier.predict(data.inputs, batch_size=2048 * 8).flatten().astype(K.floatx()))
        pass

    if args.plot:
        plot_posterior(data, args, adversary, name='posterior_end', title="End of training")
        pass


    # Early stopping in case of adversarial network
    # --------------------------------------------------------------------------
    if args.optimise_adversarial:
        # @TODO: Decide on proper metric! Stratified k-fold cross-validation?
        return np.min(val_avg)


    # Saving "vanilla" classifier in lwtnn-friendly format.
    # --------------------------------------------------------------------------
    lwtnn_save(classifier, 'ann')


    # Perform DDT transform
    # --------------------------------------------------------------------------
    with Profile("DDT transform"):

        # Compute rhoDDT
        data.add_field('rhoDDT', np.log(np.square(data['m']) / data['pt'] / 1.))

        # Tau21DDT(1)
        xmin, xmax = 1.5, 4.0

        # (opt.) Plot
        intercept, slope = plot_decorrelation(data.background,
                                              args,
                                              name='tagger_decorrelation__1',
                                              fit_range=(xmin, xmax))

        # Compute tau21DDT
        Tau21DDT = data['Tau21'] - slope * (data['rhoDDT'] - xmin)

        # Append to data container
        data.add_field('Tau21DDT_1', Tau21DDT)

        # Tau21DDT(2)
        xmin, xmax = 1.5, 9999 #np.inf

        selection = lambda array: array['pt'] > 2 * array['m']
        eff_W = np.sum(data.signal.weights[selection(data.signal)]) / \
                np.sum(data.signal.weights)
        log.warning("Weighted fraction of W events passing Tau21DDT_2 selection: {:.1f}%".format(eff_W * 100.))

        msk = selection(data.background) #data.background['pt'] > 2 * data.background['m']

        # (opt.) Plot
        intercept, slope = plot_decorrelation(data.background.slice(msk),
                                              args,
                                              name='tagger_decorrelation__2',
                                              fit_range=(xmin, xmax))

        # Compute tau21DDT
        Tau21DDT = data['Tau21'] - slope * (data['rhoDDT'] - xmin)

        # Restrict kinematics (pT > 2 x m)
        Tau21DDT[~selection(data)] = np.nan

        # Append to data container
        data.add_field('Tau21DDT_2', Tau21DDT)

        # Tau21DDT(3)
        xmin, xmax = 1.5, 9999

        selection = lambda array: (array['pt'] > 2 * array['m']) & (array['rhoDDT'] > xmin)
        eff_W = np.sum(data.signal.weights[selection(data.signal)]) / \
                np.sum(data.signal.weights)
        log.warning("Weighted fraction of W events passing Tau21DDT_3 selection: {:.1f}%".format(eff_W * 100.))

        msk = selection(data.background)

        # (opt.) Plot
        intercept, slope = plot_decorrelation(data.background.slice(msk),
                                              args,
                                              name='tagger_decorrelation__3',
                                              fit_range=(xmin, xmax))

        # Compute tau21DDT
        Tau21DDT = data['Tau21'] - slope * (data['rhoDDT'] - xmin)

        # Restrict kinematics
        Tau21DDT[~selection(data)] = np.nan

        # Append to data container
        data.add_field('Tau21DDT_3', Tau21DDT)

        # @TODO: Plot comparison of tau21 and tau21DDT?

        # ...
        pass


    # Plotting: Cost log for adversarial fit
    # --------------------------------------------------------------------------
    if args.plot and (args.train or args.train_classifier):
        with Profile("Plotting: Cost log, adversarial, full"):

            fig, ax = plt.subplots()
            colours = map(lambda d: d['color'], list(plt.rcParams["axes.prop_cycle"]))
            epochs = 1 + np.arange(len(history['loss']))
            lambda_reg = cfg['combined']['model']['lambda_reg']
            lr_ratio   = cfg['combined']['model']['lr_ratio']

            classifier_loss = np.mean([loss for key,loss in history.iteritems() if key.startswith('combined') and int(key.split('_')[-1]) % 2 == 1 ], axis=0)
            adversary_loss  = np.mean([loss for key,loss in history.iteritems() if key.startswith('combined') and int(key.split('_')[-1]) % 2 == 0 ], axis=0) * lambda_reg
            combined_loss   = classifier_loss + adversary_loss

            plt.plot(epochs, classifier_loss, color=colours[0],  linewidth=1.4,  label='Classifier')
            plt.plot(epochs, adversary_loss,  color=colours[1],  linewidth=1.4,  label=r'Adversary ($\lambda$ = {})'.format(lambda_reg))
            plt.plot(epochs, combined_loss,   color=colours[-1], linestyle='--', label='Combined')

            plt.title('Adversarial training', fontweight='medium')
            plt.xlabel("Training epochs", horizontalalignment='right', x=1.0)
            plt.ylabel("Objective function",   horizontalalignment='right', y=1.0)

            epochs = [0] + list(epochs)
            step = max(int(np.floor(len(epochs) / 10.)), 1)

            plt.xticks(filter(lambda x: x % step == 0, epochs))
            plt.legend()

            plt.text(0.03, 0.95, "ATLAS",
                     weight='bold', style='italic', size='large',
                     ha='left', va='top',
                     transform=ax.transAxes)

            plt.savefig(args.output + 'costlog_combined.pdf')
            pass
        pass


    # Plotting
    # --------------------------------------------------------------------------
    # Tagger variables
    variables = ['Tau21', 'D2', 'NN', 'ANN', 'Tau21DDT_1', 'Tau21DDT_2', 'Tau21DDT_3']


    # Plotting: Distributions
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if args.plot:
        with Profile("Plotting: Distributions"):
            for var in variables:
                print "-- {}".format(var)
                plot_distribution(data, args, var)
                pass
            pass
        pass


    # Plotting: Jet mass spectra
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if args.plot:
        with Profile("Plotting: Jet mass spectra"):
            plot_jetmass_comparison(data, args, cut_eff=0.5)
            for var in variables:
                print "-- {}".format(var)
                plot_jetmass(data, args, var, cut_eff=[0.5, 0.4, 0.3, 0.2, 0.1])
                pass
            pass
        pass


    # Plotting: ROCs (NN and ANN)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if args.plot:
        with Profile("Plotting: ROCs (NN and ANN)"):
            plot_roc(data.test, args, variables, name='tagger_ROCs_ANN')
            pass
        pass


    # Plotting: Profiles
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if args.plot:
        with Profile("Plotting: Profiles"):
            for var in variables:
                print "-- {}".format(var)
                plot_profiles(data.background, args, var) # (data.test, ...)
                pass
            pass
        pass


    # Clean-up
    # --------------------------------------------------------------------------
    if args.tensorboard:
        # @TODO: Improve, using `ps`.
        log.info("TensorBoard process ({}) is running in background. Enter `q` to close it. Enter anything else to quit the program while leaving TensorBoard running.".format(tensorboard_pid))
        response = raw_input(">> ")
        if response.strip() == 'q':
            subprocess.call(["kill", str(tensorboard_pid)])
        else:
            log.info("TensorBoard process is left running. To manually kill it later, do:")
            log.info("$ kill {}".format(tensorboard_pid))
            pass
        pass

    return 0


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()

    # Call main function
    main(args)
    pass