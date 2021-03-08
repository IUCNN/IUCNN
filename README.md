[![Travis build status](https://travis-ci.com/azizka/IUCNN.svg?branch=master)](https://travis-ci.com/azizka/IUCNN)
[![Codecov test coverage](https://codecov.io/gh/azizka/IUCNN/branch/master/graph/badge.svg)](https://codecov.io/gh/azizka/IUCNN?branch=master)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

# IUCNN
Batch estimation of species' IUCN Red List threat status using neural networks.


# Installation
1. install IUCNN directly from Github using devtools. 
```{r}
install.packages("devtools")
library(devtools)

install_github("azizka/IUCNN")
```

2. Python needs to be installed, for instance using miniconda and reticulated from within R (this will need c. 3 GB disk space).
If problems occur at this step, check the excellent [documentation of reticulate](https://rstudio.github.io/reticulate/index.html).
```{r}
install.packages(reticulate)
library(reticulate)
install_miniconda()
```
If python has been installed before, you can specify the python version to sue with `reticulate::use_python()`


3. Install the tensorflow and matplotlib modules
```{r}
reticulate::py_install("tensorflow==2.0.0", pip = TRUE)
reticulate::py_install("matplotlib", pip = TRUE)
```

# Usage
There are multiple models and features available in IUCNN. A vignette with a detailed tutorial on how to use those is available as part of the package: `vignette("Approximate_IUCN_Red_List_assessments_with_IUCNN")`. Running IUCNN will write files to your working directory.

```{r}
library(tidyverse)
library(IUCNN)

#load example data 
data("training_occ") #geographic occurrences of species with IUCN assessment
data("training_labels")# the corresponding IUCN assessments
data("prediction_occ") #occurrences from Not Evaluated species to prdict

# Training
## Generate features
features <- prep_features(training_occ)

## Prepare training labels
labels_train <- prep_labels(training_labels)

## train the model
m1 <- train_iucnn(x = features,
                  labels = labels_train)
            
## Look at model summary and diagnostics
summary(m1)
plot(m1)

#Prediction
## Generate features
features_predict <- prep_features(prediction_occ)

## predict IUCN conservation status
predict_iucnn(x = features_predict,
              model = m1,
              model_dir = "iuc_nn_model")
```

# Citation
```{r}
library(IUCNN)
citation("IUCNN")
```

Zizka A, Silvestro D, Vitt P, Knight T (2020). “Automated conservation assessment of the orchid family with deep
learning.” _Conservation Biology_, 0, 0-0. doi: doi.org/10.1111/cobi.13616 (URL: https://doi.org/doi.org/10.1111/cobi.13616),
<URL: https://github.com/azizka/IUCNN>.
