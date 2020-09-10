library(usethis)
library(tidyverse)

load("data-raw/train_example.rda")

orchid_occ <- ass_occ %>%
  select(species = canonical_name, decimallongitude, decimallatitude)

orchid_classes <- ass_occ %>%
  select(species = canonical_name, category_broad, category_detail)


## code to prepare `DATASET` dataset goes here

usethis::use_data(orchid_classes, overwrite = TRUE)

labels_detail <- read.delim("data-raw/1_main_iucn_full_clean_detailed_labels_fullds.txt", col.names = "labels")
species_id <- read.delim("data-raw/1_main_iucn_full_clean_detailed_speciesid_fullds.txt", col.names = "species")
labels_detail <- bind_cols(species_id, labels_detail)
usethis::use_data(labels_detail, overwrite = TRUE)

labels_broad <- read.delim("data-raw/2_main_iucn_full_clean_broad_labels_fullds.txt", col.names = "labels")
species_id <- read.delim("data-raw/2_main_iucn_full_clean_broad_speciesid_fullds.txt", col.names = "species")
labels_broad <- bind_cols(species_id, labels_broad)
usethis::use_data(labels_broad, overwrite = TRUE)
