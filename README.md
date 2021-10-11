# echo_classification

Data and training pipeline for classification of newborn echocardiograms,
for detection of pulmonary hypertension (ph), a functional heart defect,
in newborns.

Main code and classes are located in the ehco_ph package.

Scripts for pre-processing data, generating index files (for splitting into train 
and valid), training the networks, and analysing results is found inside 
the <code>scripts</code> directory.

To use the echo_ph package, you must run: 
    <code> pip install -e . </code>