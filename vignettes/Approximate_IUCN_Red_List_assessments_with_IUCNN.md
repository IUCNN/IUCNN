---
title: Approximate IUCN Red List assessments with IUCNN
output: rmarkdown::html_vignette
vignette: >
  %\VignetteIndexEntry{Approximate_IUCN_Red_List_assessments_with_IUCNN}
  %\VignetteEngine{knitr::rmarkdown}
  %\VignetteEncoding{UTF-8}
---

```{r, include = FALSE}
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  warning=FALSE,
  message=FALSE
)
```

# Introduction
The conservation assessments of the Global Red List of the International Union for the Conservation of nature (www.iucn.org), are arguably one of the most thorough and widely used tools to assess the global extinction risk of species. However, IUCN assessments---usually performed by a group of specialists for each taxonomic group, or professional assessors---are time and data intense, and therefore a large fraction of global plant and animal species have not yet been evaluated. IUCNN implements neural networks to predict the IUCN status of so far not evaluated or data deficient species based on publicly available geographic distribution and environmental data and existing red lists from other species. A typical application example are to predict conservation status of some plant species using all evaluated species in the same family as training data.


```{r setup}
library(IUCNN)
library(magrittr)
library(dplyr)
```

# Installation
IUCNN uses R and python, so multiple steps are necessary to install IUCNN.

1. install IUCNN directly from Github using devtools. 
```{r, eval = FALSE}
install.packages("devtools")
library(devtools)

install_github("azizka/IUCNN")
library(IUCNN)
```

2. Python needs to be installed, for instance using miniconda and reticulated from within R (this will need c. 3 GB disk space).
If problems occur at this step, check the excellent [documentation of reticulate](https://rstudio.github.io/reticulate/index.html).
```{r, eval = FALSE}
install.packages(reticulate)
library("reticulate")
install_miniconda()
```
If python has been installed before, you can specify the python version to sue with `reticulate::use_python()`


3. Install the tensorflow Python module
```{r, eval = FALSE}
reticulate::conda_install("r-reticulate","tensorflow")
reticulate::py_install("matplotlib", pip = TRUE)
reticulate::py_install("https://github.com/dsilvestro/npBNN/archive/v0.1.8.tar.gz", pip = TRUE)
```

# Prepare input data
IUCNN predicts the IUCN Global Red List assessment categories of Not Evaluated and Data Deficient species based on geographic occurrence records and a set of training species for which occurrence records and IUCN assessments are available (training data). The amount of training species necessary varies with the number of categories but in general "the more, the better". Ideally, the training dataset should comprise several hundred species, so a typical scenario will be to use all available plant species from a region, or all available species from a plant family. If the availability of training species is limited, a good option can be to predict possibly threatened (IUCN categories "CR", "EN", and "VU") vs. not threatened species ("NT" and "LC").

Hence, three types of input are necessary, which are easily available for many species: 

## 1. Geographic occurrence records of training species (training occurrences)
Occurrence records might stem from a variety of databases, For example, from field collections or public databases such BIEN (https://bien.nceas.ucsb.edu/bien/) or GBIF (www.gbif.org). GBIF data can be obtained from within R via the rgbif package, See [here](https://ropensci.org/tutorials/rgbif_tutorial/) for a tutorial on how to do so. IUCNN needs a dataset with (at least) three columns, containing the species name, decimal longitude coordinates and decimal latitude coordinates. If you are interested in cleaning records from GBIF, you may want to have a look at this [blog post] () and check out the [CoordinateCleaner]() and [bRacatus]() packages. 

## 2. IUCN Global Red List assessment of the training species (training labels)
These can be obtained from IUCN, either via the webpage www.iucn.org or via the rredlist package from inside R (preferred for many species). See [here](https://ropensci.org/tutorials/rredlist_tutorial/) for a tutorial on how to use rredlist. It is important, that all target label classes are well represented in the training data, which is rarely the case for IUCN data, since for instance "VU" is rare. If the classes are to imbalanced, consider using possibly threatened (IUCN categories "CR", "EN", and "VU") vs. not threatened species ("NT" and "LC").

## 3. Geographic occurrence records of the species for which the IUCN status should be predicted (predict occurrences)
Geographic occurrence for the target species, in the same format as for the training occurrences described above.

Example dataset are available with IUCNN: `data(training_occ)` (training occurrences), `data(training_labels)` (training labels) and `data(prediction_occ)`.

## Feature preparation
IUCNN uses sets of per species traits ("features"). Necessary is an input data.frame, with a species column, and then numerical columns indicating the feature values for each species. In general, features might represent any species trait, including from taxonomy (family), anatomy (body size), ecology (e.g., feeding guild) or conservation (e. g., population dynamics). Any of these features can be provided to IUCNN. However, since most of these data are scarce for many taxonomic groups, in most cases features will be based on geographic occurrences and auxiliary data alone. The IUCNN package contains functions to obtain default features including geographic features (number of occurrences, number of unique occurrences, mean latitude, mean longitude, latitudinal range, longitudinal range, the extend of occurrence, the area of occupancy and hemisphere), climatic features (median values per species from 19 bioclim variables from www.worldclim.org) and biome features (presence in global biomes from the [WWF](https://www.worldwildlife.org/publications/terrestrial-ecoregions-of-the-world)) and human footprint features based on occurrence records. In this tutorial, we will use the example datasets from the Orchid family (Orchidaceae) provided with the IUCNN package, 

You can prepare the default features with a single call to `prep_features`
```{r, results='hide'}
data("training_occ") #geographic occurrences of species with IUCN assessment
data("prediction_occ")

features_train <- prep_features(training_occ)
features_predict <- prep_features(prediction_occ)

```

## Label preparation
IUCNN expects the labels for training as numerical categories. So, to use IUCN Red List categories, those need to be converted to numeric in the right way. This can be done using the `prepare_labels` function. The function can use with detailed categories or with broader threatened not threatened categories. See `?prepare_labels` for more information. The labels will be converted into numeric categories following the `accepted_labels` argument, so for instance, in the default case: LC -> 0 and CR -> 4. If you change the accepted labels, the match will change accordingly.

```{r}
data("training_labels")

labels_train <- prep_labels(training_labels)
```


# Running IUCNN
Running IUCNN consists of two steps: 1) training a neural network and 2) predicting the status of new species. IUCNN contains three different neural network approaches to predict the IUCN status of species, which can all be customized. We present the default approach here, see section "Customizing analyses" for details on how to train a Bayesian or regression type neural network. 

## Model training
Based on the training features and labels, IUCNN trains a neural network, using the tensorflow module. The training is done vie the `train_iucnn` function. There are multiple options to change the design of the network, including among others the number of layers, and the fraction of records used for testing and validation. The `train_iucnn` function will write a folder to the working directory containing the model and return summary statistics including cross-entropy loss and accuracy for the validation set, which can be used to compare the performance of different models.

The following code is used to set up and train a neural network model with 3 hidden layers of 60, 60, and 20 nodes, with ReLU activation function. By specifying a seed (here: 1234) we make sure the same subsets of data are designated as training, validation and test sets across different runs and model configurations (see below). The model with estimated weights will be saved in the current working directory. 

```{r}
res_1 <- train_iucnn(x = features_train,
                     lab = labels_train, 
                     path_to_output = "iucnn_model_1")
```

You can use the `summary` and `plot` methods to get an overview on the training process and model performance. 

```{r}
summary(res_1)
plot(res_1)
```

## Predict IUCN Global Red List status
You can then use the trained model to predict the conservation status of *Not Evaluated* and *Data Deficient* species with the `predict_iucnn` function. The output is a data frame with species names and numeric labels (as generated with prepare_labels).

```{r}
predictions <- predict_iucnn(x = features_predict, 
                             model = res_1)

plot(predictions)
```

It is important to remember the following points when using IUCNN:

1. The resulting IUCNN categories are predictions. While IUCNN has reached accuracies between 80 and 90% on the broad (threatened vs non-threatened) level and up to 80% on the detailed level, some species will be mis-classified.

2. IUCNN is indifferent to the provided features. On the one hand this means that any species traits for which data is available can bes used, but on the other hand this means that thought is needed in the choice of the features. The default features of IUCNN are usually a safe choice. The number of features is not limited, but currently IUCNN does not support missing values in the feature table and removes species with missing values. 

3. IUCNN is indifferent to the relation between training and test data. So it is possible to use training data from Palearctic birds to predict the conservation status of South American Nematodes. This is not recommended. Instead, a better approach will be to predict the conservation status of species, from training data of the same genus, order, or family. Alternatively, training data could be chosen on geographic region or functional aspects (e.g., feeding guilt or body size). However some inclusion of taxonomy/evolutionary history for the choice of training data is recommended.

4. The amount of training data is important. The more the better. Minimum several hundred training species with a more or less equal distribution on the label classes should be included. If training data is limited, the broader Threatened/Not threatened level is recommended. 

5. IUCNN predictions are not equivalent to full IUCN Red List assessments. We see the main purpose of IUCNN in 1) identifying species that will likely need conservation action to trigger a full IUCN assessment, and 2) provide large-scale overviews on the extinction risk in a given taxonomic group, for instance in a macro-ecological and macro-evolutionary context.

## Evaluating feature importance
# Customizing IUCNN analyses
IUCNN contains multiple options to customize the steps of the analyses to adapt to particularities of taxonomic groups and regions and to accommodate differences in data availability. Below we describe the most important options to customize 1) feature and label preparation and 2) model training and 3) status prediction.

## 1) Features and Labels
### Add and remove feature blocks
The default labels are selected based on empirical test on relevance for different taxa and regions. However, for some analyses only part of the features may be relevant. Table 1 below explains all default features. You can exclude feature blocks using the `type` argument of the `prep_features` function. For instance, to exclude the biome features:

```{r, eval = FALSE}
features_train2 <- prep_features(training_occ, type = c("geographic", "climate", "humanfootprint"))
```

### Prepare features individually
If more control over feature preparation is necessary, each feature block can be obtained by an individual function.

Table 2. Functions to obtain default features and options to customize the features.
|Feature block|Function name|Options to customize|
|---|---|---|
|Geographic|`ft_geo`|-|
|Biomes|`ft_biom`|change the reference dataset of biomes (biome_input, biome.id), remove biomes without any species occurrence (remove_zeros)|
|Climate|`ft_clim`|the amount of bioclim variables from the default source to be included (type), the resolution of the default input data (res)|
|Human footprint|`ft_foot`|chose the time points from the default source (year), the break points for the different footprint categories (breaks, by default approximately quantiles on the global footprint dataset) or a default source for human footprint (footp_input)|

For instance:

```{r, eval = FALSE}
clim_features <- ft_clim(x = training_occ, 
                         type = "selected")

clim_features2 <- ft_clim(x = training_occ, 
                         type = "all")
```

### Use custom features
It is also possible to provide features unrelated to the default features. They may contain any continuous or categorical features, but some processing will be needed. The format needs to be a data.frame with a compulsory column containing the species name. Continuous variables should be rescaled to cover a similar range, whereas categorical features should be coded binary (present/absent, as the custom biome features).

For instance:

```{r, eval = FALSE}
feat <- data.frame(species = c("Adansonia digitata", "Ceiba pentandra"),
                   max_plant_size_m = c(25, 50),
                   africa = c(1,1),
                   south_america = c(0,1),
                   fraction_of_records_in_protected_area = c(25, 75))
```

Table 2. Description of the default features included in `prep_features`. All continuous variables are rescaled to a similar range.
| Feature | Block | Name | Description |
|---|---|---|---|
|tot_occ|Geographic|Number of occurrences|The total number of occurrences available for this species|
|uni_occ|Geographic|Number of geographically unique occurrences|The number of geographically unique records available for this species|
|mean_lat|Geographic|Mean latitude|The mean latitude of all records of this species|
|mean_lon|Geographic|Mean longitude|The mean longitude of all records of this species|
|lat_range|Geographic|Latitudinal range|The latitudinal range (.95 quantile - .05 quantile).|
|lon_range|Geographic|Longitudinal range|The longitudinal range (.95 quantile - .05 quantile).|
|alt_hemisphere|Geographic|The hemisphere|0 = Southern hemisphere, 1 = Northern hemisphere|
|eoo|Geographic|Extend of Occurrence|The extend of occurrence. Calculated by rCAT. For species with less than 3 records set to AOO|
|aoo|Geographic|Area of Occupancy||
|1|Biome|Tropical & Subtropical Moist Broadleaf Forests|Are at least 5% of the species records present in this biome?|
|2|Biome|Tropical & Subtropical Dry Broadleaf Forests|Are at least 5% of the species records present in this biome?|
|3|Biome|Tropical & Subtropical Coniferous Forests|Are at least 5% of the species records present in this biome?|
|4|Biome|Temperate Broadleaf & Mixed Forests|Are at least 5% of the species records present in this biome?|
|5|Biome|Temperate Conifer Forests|Are at least 5% of the species records present in this biome?|
|6|Biome|Boreal Forests/Taiga|Are at least 5% of the species records present in this biome?|
|7|Biome|Tropical & Subtropical Grasslands, Savannas & Shrublands|Are at least 5% of the species records present in this biome?|
|8|Biome|Temperate Grasslands, Savannas & Shrublands|Are at least 5% of the species records present in this biome?|
|9|Biome|Flooded Grasslands & Savannas|Are at least 5% of the species records present in this biome?|
|10|Biome|Montane Grasslands & Shrublands|Are at least 5% of the species records present in this biome?|
|11|Biome|Tundra|Are at least 5% of the species records present in this biome?|
|12|Biome|Mediterranean Forests, Woodlands & Scrub|Are at least 5% of the species records present in this biome?|
|13|Biome|Deserts & Xeric Shrublands|Are at least 5% of the species records present in this biome?|
|14|Biome|Mangroves|Tropical & Subtropical Moist Broadleaf Forests|Are at least 5% of the species records present in this biome?|
|98|Biome|Lake|Are at least 5% of the species records present in this biome?|
|99|Biome|Rock and ice|Are at least 5% of the species records present in this biome?|
|bio1|Climate|Annual Mean Temperature|The median value of this bioclimatic layer for the occurrence records of a species. Records with NA values removed|
|bio4|Climate|Temperature Seasonality|The median value of this bioclimatic layer for the occurrence records of a species. Records with NA values removed|
|bio11|Climate|Mean Temperature of Coldest Quarter|The median value of this bioclimatic layer for the occurrence records of a species. Records with NA values removed|
|bio12|Climate|Annual Precipitation|The median value of this bioclimatic layer for the occurrence records of a species. Records with NA values removed|
|bio15|Climate|Precipitation Seasonality|The median value of this bioclimatic layer for the occurrence records of a species. Records with NA values removed|
|bio17|Climate|Precipitation of Driest Quarter|The median value of this bioclimatic layer for the occurrence records of a species. Records with NA values removed|
|range_bio1|Climate|Range of annual Mean Temperature|The range of value of this bioclimatic layer for the occurrence records of a species. Records with NA values removed. Range is the .95-.05 quantile.|
|range_bio4|Climate|Range of temperature Seasonality|The range of value of this bioclimatic layer for the occurrence records of a species. Records with NA values removed. Range is the .95-.05 quantile.|
|range_bio11|Climate|Range of mean Temperature of Coldest Quarter|The range of value of this bioclimatic layer for the occurrence records of a species. Records with NA values removed. Range is the .95-.05 quantile.|
|range_bio12|Climate|Range of annual Precipitation|The range of value of this bioclimatic layer for the occurrence records of a species. Records with NA values removed. Range is the .95-.05 quantile.|
|range_bio15|Climate|Range of precipitation Seasonality|The range of value of this bioclimatic layer for the occurrence records of a species. Records with NA values removed. Range is the .95-.05 quantile.|
|range_bio17|Climate|Range of precipitation of Driest Quarter|The range of value of this bioclimatic layer for the occurrence records of a species. Records with NA values removed. Range is the .95-.05 quantile.|
|humanfootprint_1993_1|Human footprint year 1993 lowest impact|The fraction of records in areas of the lowest category of human footprint in the year 1993. Footprint was categorized so that categorize represent roughly quantiles.|
|humanfootprint_1993_2|Human footprint|Human footprint year 1993 intermediate impact 1|The fraction of records in areas of the second lowest category of human footprint in the year 1993. Footprint was categorized so that categorize represent roughly quantiles.|
|humanfootprint_1993_3|Human footprint|Human footprint year 1993 intermediate impact 2|The fraction of records in areas of the second highest category of human footprint in the year 1993. Footprint was categorized so that categorize represent roughly quantiles.|
|humanfootprint_1993_4|Human footprint|Human footprint year 1993 highest impact|The fraction of records in areas of the highest category of human footprint in the year 1993. Footprint was categorized so that categorize represent roughly quantiles.|
|humanfootprint_2009_1|Human footprint|Human footprint year 2009 lowest impact|The fraction of records in areas of the lowest category of human footprint in the year 2009. Footprint was categorized so that categorize represent roughly quantiles.|
|humanfootprint_2009_2|Human footprint|Human footprint year 2009 intermediate impact 1|The fraction of records in areas of the second lowest category of human footprint in the year 2009. Footprint was categorized so that categorize represent roughly quantiles.|
|humanfootprint_2009_3|Human footprint|Human footprint year 2009 intermediate impact 2|The fraction of records in areas of the second highest category of human footprint in the year 2009. Footprint was categorized so that categorize represent roughly quantiles.|
|humanfootprint_2009_4|Human footprint|Human footprint year 2009 highest impact|The fraction of records in areas of the highest category of human footprint in the year 2009. Footprint was categorized so that categorize represent roughly quantiles.|

### Labels: Full categories vs Threatened/Not threatened
The `prep_labels` function may accepted any custom labels as long as they are included in the `accepted_labels` option. It also can provide a classification into threatened/non-threatened, via the `level` and `threatened` options. ON the broader level the model accuracy is usually significantly higher.

For instance:

```{r, eval = FALSE}
labels_train <- prep_labels(training_labels, 
                            level = "broad")
```

## 2) Model training - NN regression model
### Customizing model parameters
The `train_iucnn` function contains various options to customize the neural network, including among other the fraction of validation and test data, the maximum number of epochs, the number of layers and nodes, the activation function , dropout and randomization of the input data. See `?train_iucnn` for a comprehensive list of options and their description. By default, `train_iucnn` trains a neural network with 3 hidden layers with 60, 60 and 20 nodes and a sigmoid as activation function. Depending on your dataset different networks may improve performance. For instance, you can set up a different model with 1 hidden layer of 60 nodes, a sigmoidal activation function and without using a bias node in the first hidden layer.

```{r}
res_2 <- train_iucnn(x = features_train,
                   lab = labels_train, 
                   dropout_rate = 0.3,
                   path_to_output= "iucnn_model_2",
                   n_layers = "60",
                   use_bias = FALSE,
                   act_f = "sigmoid")
```

You can compare the validation loss of the models using `res_1$validation_loss` and `res_2$validation_loss`. Model 2 in this case yields a lower validation loss and is therefore preferred. Once you chose the preferred model configuration based on validation loss, we can check test accuracy of best model: `res_2$test_accuracy`. The `train_iucnn` function contains various options to adapt the model, see the section "Setting model parameters" below for details. 

### Changing the modeling algorithm
There are three neural network algorithms implemented in iucnn. Besides the default classifier approach based on a tensorflow implementation, these are a Bayesian neural network and a regression model.

The Bayesian approach has the advantage that it returns true probabilities for the classification of species into the relative output classes (e.g. 80% probability of a species to be LC). We consider this approach more suitable for classification of species into IUCN categories, than the default option. It will need more time for model training and should best be applied once you have identified the best model parameters using the default approach. You can run a BNN setting the `mode` option of `train_iucnn` to `"bnn-class"`.

```{r, eval = FALSE}
res_3 <- train_iucnn(x = features_train,
                    lab = labels_train, 
                    path_to_output = "iucnn_model_3",
                    mode = 'bnn-class')
```

IUCNN also offers the option to train a NN regression model instead of a classifier. Since the IUCN threat statuses constitute a list of categories that can be sorted by increasing threat level, we can model the task of estimating these categories as a regression problem.  You can run such a model with the `train_iucnn()` function, specifying to train a regression model by setting `mode = 'nn-reg'`.


```{r}
res_4 <- train_iucnn(x = features_train,
                    lab = labels_train, 
                    path_to_output = "iucnn_model_4",
                    mode = 'nn-reg',
                    rescale_features = TRUE)
```

### Feature importance
You can use the `feature_importance` function to gauge the importance of different feature blocks for model performance. The function will randomize individual feature blocks and return the decrease in model performance caused by the randomization. If you have used other than the default features, you can define feature blocks using the `feature_blocks` option. 
```{r, eval = FALSE}
feature_importance(x = res_1)
```

### Model testing

TOBI explain here

## 3) Status prediction
The `predict_iucnn` function offer some function to customize predictions. The most important option in many cases is `target_acc`. With this option you can set a target overall accuracy that the model needs to achieve. All species that cannot be classified with enough certainty to reach this target accuracy will be classified as DD (Data Deficient).
```{r}
pred_2 <- predict_iucnn(x = features_predict, 
                        target_acc = 0.7,
                        model = res_2)
```

Furthermore, via the `return_raw` and `return_IUCN` options you can customize the output format. With `return_raw` option you can return the raw probabilities for the classification (see `?predict_iucnn`) and with `return_IUCN` you can chose to either return the IUCN category labels or the internal labels used by the neural network.
```{r}
pred_3 <- predict_iucnn(x = features_predict, 
                        model = res_2,
                        return_raw = TRUE)
```

## 4) The number od species per category

### just get the numbers


### Account for uncertainty

