# adversarial

Tools for training and evaluating different methods for constructing
mass-decorrelated jet substructure taggers, in particular adversarially trained
neural networks. All information, guides, etc. can be found in the project
[wiki](https://github.com/asogaard/adversarial/wiki).

## For the Dark Jet Resonance search:

1) Convert your ROOT ntuples to hdf5 files
   * python -m prepro.converterDJR 
   * python -m prepro.converterDJR_data 

1.1) If the input files are too big/many, there's an example script for submitting the jobs to the lxplus batch system in the submission folder. 

   * condor_submit submission/converter.sub

2) To construct the fixed-efficiency ntrk variable, ntrk_epsilon, we use the kNN paakage, and run the "training" on data in oder to be less sensitive to MC mismodelling. 
   * python -m run.knn.train1D --input djr_LCTopo_data.h5

3) Validate that the fixed -efficiency regression using the kNN fit is working as expected. This step also produces a ROOT files with the kNN fit stored, such that the following steps can be done in a ROOT framework instead.   
   * python -m run.knn.test1D --input djr_LCTopo_data.h5

4) Now move on to study the sensitivity of the new ntrk_epsilon variable. This we test on MC. 
   * python -m tests.sensitivity --input djr_LCTopo_1.h5

5) To study the effect of a given cut on the dijet invariant mass distribution run
   * python -m tests.mjjDistributions --input djr_LCTopo_1.h5   