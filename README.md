[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

# IUCNN
Batch estimation of species' IUCN Red List threat status using neural networks.

# Installation
1. Install IUCNN directly from Github using devtools. 
```{r}
install.packages("devtools")
library(devtools)

install_github("azizka/IUCNN")
```

2. Since some of IUCNNs functions are run in Python, IUCNN needs to set up a Python environment. This is easily done from within R, using the `install_miniconda()` function of the package `reticulate` (this will need c. 3 GB disk space).
If problems occur at this step, check the excellent [documentation of reticulate](https://rstudio.github.io/reticulate/index.html).
```{r}
install.packages("reticulate")
library(reticulate)
install_miniconda()
```


3. Install the tensorflow python library. If you are using **MacOS** or **Linux** it is recommended to install tensorflow using conda:
```{r}
reticulate::conda_install("r-reticulate","tensorflow=2.4")
```

If you are using **Windows**, you can install tensorflow using pip:

```{r}
reticulate::py_install("tensorflow~=2.4.0rc4", pip = TRUE)
```

4. Finally install the npBNN python library from Github:

```{r}
reticulate::py_install("https://github.com/dsilvestro/npBNN/archive/v0.1.10.tar.gz", pip = TRUE)
```

# Usage
There are multiple models and features available in IUCNN. A vignette with a detailed tutorial on how to use those is available as part of the package: `vignette("Approximate_IUCN_Red_List_assessments_with_IUCNN")`. Running IUCNN will write files to your working directory.

A simple run:

```{r}
library(tidyverse)
library(IUCNN)

#load example data 
data("training_occ") #geographic occurrences of species with IUCN assessment
data("training_labels")# the corresponding IUCN assessments
data("prediction_occ") #occurrences from Not Evaluated species to prdict

# 1. Feature and label preparation
features <- prep_features(training_occ) # Training features
labels_train <- prep_labels(training_labels) # Training labels
features_predict <- prep_features(prediction_occ) # Prediction features

# 2. Model training
m1 <- train_iucnn(x = features, lab = labels_train)

summary(m1)
plot(m1)

# 3. Prediction
predict_iucnn(x = features_predict,
              model = m1)
```

A more complex approach including model testing:

```{r}
library(tidyverse)
library(IUCNN)

#load example data 
data("training_occ") #geographic occurrences of species with IUCN assessment
data("training_labels")# the corresponding IUCN assessments
data("prediction_occ") #occurrences from Not Evaluated species to prdict

# 1. Feature and label preparation
features <- prep_features(training_occ) # Training features
labels_train <- prep_labels(training_labels) # Training labels
features_predict <- prep_features(prediction_occ) # Prediction features

# 2. model testing
mt <- modeltest_iucnn(features,
                      labels_train_detail,
                      modeltest_logfile = "model_testing_example.txt",
                      seed = 1234,
                      dropout_rate = c(0.1, 0.3),
                      n_layers = c('40_20','50_30_10'),
                      cv_fold = 3,
                      mode = 'nn-class',
                      init_logfile = TRUE)

# get best model based on chosen criterion
bm <- best_model_iucnn(mt,
                      criterion = 'val_acc',
                      require_dropout = TRUE)

# get overview of best model: prediction accuracy, confusion matrix, etc.
summary(bm)
plot(bm)

# 3. train the production model with all data, using the specs from the model testing
pm <- train_iucnn(features,
                  labels_train,
                  production_model = bm,
                  overwrite = TRUE)

# make predictions
predictions <- predict_iucnn(features_predict,
                             pm,
                             target_acc = 0.6)

# plot the predicted status distribution
plot_predictions(predictions$predictions)

```


# Citation
```{r}
library(IUCNN)
citation("IUCNN")
```

Zizka A, Silvestro D, Vitt P, Knight T (2020). “Automated conservation assessment of the orchid family with deep
learning.” _Conservation Biology_, 0, 0-0. doi: doi.org/10.1111/cobi.13616 (URL: https://doi.org/doi.org/10.1111/cobi.13616),
<URL: https://github.com/azizka/IUCNN>.
