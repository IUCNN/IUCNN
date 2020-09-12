[![Travis build status](https://travis-ci.com/azizka/IUCNN.svg?branch=master)](https://travis-ci.com/azizka/IUCNN)
[![Codecov test coverage](https://codecov.io/gh/azizka/IUCNN/branch/master/graph/badge.svg)](https://codecov.io/gh/azizka/IUCNN?branch=master)

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
If problems occur at this step, check the excellent [documentation of reticualte](https://rstudio.github.io/reticulate/index.html).
```{r}
install.packages(reticulate)
library(reticualte)
install_miniconda()
```
If python has been installed before, you can specify the python version to sue with `reticulate::use_python()`


3. Install the tensorflow module
```{r}
reticulate::py_install("tensorflow==2.0.0", pip = TRUE)
```

# Usage
A vignette with a detailed tutorial on how to use IUCNN is available as part of the package: `vignette("Approximate_IUCN_Red_List_assessments_with_IUCNN")`. Running IUCNN will write files to your working directory.

```{r}
library(tidyverse)
library(IUCNN)

#load example data 
data("orchid_occ") #geographic occurrences of species with IUCN assessment
data("labels_detail")# the corresponding IUCN assessments
data("orchid_target") #occurrences from Not Evaluated species to prdict

# Training
## Generate features
geo <- geo_features(orchid_occ) #geographic
cli <- clim_features(orchid_occ) #climate
bme <- biome_features(orchid_occ) #biomes

features <- geo %>% 
  left_join(cli) %>% 
  left_join(bme)

# train the model
train_iucnn(x = features,
            labels = labels_detail)

#Prediction
## Generate features
geo <- geo_features(orchid_target)
cli <- clim_features(orchid_target)
bme <- biome_features(orchid_target)

features_predict <- geo %>% 
  left_join(cli) %>% 
  left_join(bme)

predict_iucnn(x = "features_predict")
```

## Citation
```{r}
library(IUCNN)
citation("IUCNN")
```
