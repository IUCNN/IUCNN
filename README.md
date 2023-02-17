<!-- badges: start -->
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![DOI](https://zenodo.org/badge/293626039.svg)](https://zenodo.org/badge/latestdoi/293626039)
[![R-CMD-check](https://github.com/azizka/IUCNN/workflows/R-CMD-check/badge.svg)](https://github.com/azizka/IUCNN/actions)
<!-- badges: end -->


# IUCNN
Batch estimation of species' IUCN Red List threat status using neural networks.

# Installation
1. Install IUCNN directly from Github using devtools. 
```r
install.packages("devtools")
library(devtools)

install_github("IUCNN/IUCNN")
```

2. Since some of IUCNNs functions are run in Python, IUCNN needs to set up a Python environment. This is easily done from within R, using the `install_miniconda()` function of the package `reticulate` (this will need c. 3 GB disk space).
If problems occur at this step, check the excellent [documentation of reticulate](https://rstudio.github.io/reticulate/index.html).
```r
install.packages("reticulate")
library(reticulate)
install_miniconda()
```


3. Install the tensorflow python library. If you are using **MacOS** or **Linux** it is recommended to install tensorflow using conda:
```r
reticulate::conda_install("r-reticulate","tensorflow=2.4")
```

If you are using **Windows**, you can install tensorflow using pip:

```r
reticulate::py_install("tensorflow~=2.4.0rc4", pip = TRUE)
```

4. Finally install the npBNN python library from Github:

```r
reticulate::py_install("https://github.com/dsilvestro/npBNN/archive/refs/tags/v0.1.11.tar.gz", pip = TRUE)
```

# Usage
There are multiple models and features available in IUCNN. A vignette with a detailed tutorial on how to use those is available as part of the package: `vignette("Approximate_IUCN_Red_List_assessments_with_IUCNN")`. Running IUCNN will write files to your working directory.

A simple example run for terrestrial orchids (This will take about 5 minutes and download ~500MB of data for feature preparation into the working directory):

```r
library(tidyverse)
library(IUCNN)

#load example data 
data("training_occ") #geographic occurrences of species with IUCN assessment
data("training_labels")# the corresponding IUCN assessments
data("prediction_occ") #occurrences from Not Evaluated species to prdict

# 1. Feature and label preparation
features <- iucnn_prepare_features(training_occ) # Training features
labels_train <- iucnn_prepare_labels(x = training_labels,
                                     y = features) # Training labels
features_predict <- iucnn_prepare_features(prediction_occ) # Prediction features

# 2. Model training
m1 <- iucnn_train_model(x = features, lab = labels_train)

summary(m1)
plot(m1)

# 3. Prediction
iucnn_predict_status(x = features_predict,
                     model = m1)
```
Additional features quantifying phylogenetic relationships and geographic sampling bias are available via `iucnn_phylogenetic_features` and `iucnn_bias_features`.


With model testing

```r
library(tidyverse)
library(IUCNN)

#load example data 
data("training_occ") #geographic occurrences of species with IUCN assessment
data("training_labels")# the corresponding IUCN assessments
data("prediction_occ") #occurrences from Not Evaluated species to predict

# Feature and label preparation
features <- iucnn_prepare_features(training_occ) # Training features
labels_train <- iucnn_prepare_labels(x = training_labels,
                                     y = features) # Training labels
features_predict <- iucnn_prepare_features(prediction_occ) # Prediction features


# Model testing
# For illustration models differing in dropout rate and number of layers

mod_test <- iucnn_modeltest(x = features,
                            lab = labels_train,
                            logfile = "model_testing_results-2.txt",
                            model_outpath = "iucnn_modeltest-2",
                            mode = "nn-class",
                            dropout_rate = c(0.0, 0.1, 0.3),
                            n_layers = c("30", "40_20", "50_30_10"),
                            cv_fold = 5,
                            init_logfile = TRUE)

# Select best model
m_best <- iucnn_best_model(x = mod_test,
                          criterion = "val_acc",
                          require_dropout = TRUE)

# Inspect model structure and performance
summary(m_best)
plot(m_best)

# Train the best model on all training data for prediction
m_prod <- iucnn_train_model(x = features,
                            lab = labels_train,
                            production_model = m_best)

# Predict RL categories for target species
pred <- iucnn_predict_status(x = features_predict,
                             model = m_prod)
plot(pred)

```

Using a convolutional neural network

```r
features <- iucnn_cnn_features(training_occ) # Training features
labels_train <- iucnn_prepare_labels(x = training_labels,
                                     y = features) # Training labels
features_predict <- iucnn_cnn_features(prediction_occ) # Prediction features

```

# Citation
```r
library(IUCNN)
citation("IUCNN")
```

Zizka A, Andermann T, Silvestro D (2022). "IUCNN - Deep learning approaches to approximate species’ extinction risk." [Diversity and Distributions, 28(2):227-241 doi: 10.1111/ddi.13450](https://doi.org/10.1111/ddi.13450). 

Zizka A, Silvestro D, Vitt P, Knight T (2021). “Automated conservation assessment of the orchid family with deep
learning.” [Conservation Biology, 35(3):897-908, doi: doi.org/10.1111/cobi.13616](https://doi.org/10.1111/cobi.13616)
