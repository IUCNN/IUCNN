# IUCNN 3.0.0 (29.01.2024)
=========================
  * Modified all functions to remove dependencies on retired spatial packages

# IUCNN 2.1.0 (12.05.2022)
=========================
* add NA imputation in the features using missForest
* added phylogenetic eigenvectors in iucnn_phylogenetic_features
* updated citation
* upgraded dependency on R >=4.1

# IUCNN 2.0.1 (03.01.2022)
=========================
* fixed bug with the export of the iucnn_feature_importance function

# IUCNN 2.0.0 (16.08.2021)
=========================
* moved to IUCNN/IUCNN
* added iucnn_cnn_features function
* standardized function naming scheme
* added test for polygon validity to iucnn_biome_features
* add iucnn_bias_features function and sampbias as suggested package

# IUCNN 1.0.1(01.06.2021)
=========================
* minor fix for outputting val-loss and test-loss for nn-reg mode

# IUCNN 1.0.0(27.05.2021)
=========================
* final version for first release
* some minor fixes of typos in the vignette and readme

# IUCNN 0.9.9 (26.05.2021)
=========================
* final version for last release tests

# IUCNN 0.9.3
=========================

* add footprint_features function
* added option for selected variables and ranges to the clim_features function
* added option to remove empty biomes to the biom_features function
* improved spell-checking
* add rescale option for eoo and aoo as part of geo_features
* set EOO to AOO for species with less than three occurrences
* updated the normalization of climate variables

# IUCNN 0.9.2

* bug fix predict_iucn function
* bug fix train_iucn function

# IUCNN 0.9.1

* Added description information
* Add citation information
* Start Readme
* Add vignette

# IUCNN 0.0.0.9000

* Added a `NEWS.md` file to track changes to the package.
* package skeleton
