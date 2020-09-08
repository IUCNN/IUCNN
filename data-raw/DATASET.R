library(usethis)
library(tidyverse)

load("data-raw/train_example.rda")

orchid_occ <- ass_occ %>%
  select(species = canonical_name, decimallongitude, decimallatitude)

orchid_classes <- ass_occ %>%
  select(species = canonical_name, category_broad, category_detail)


## code to prepare `DATASET` dataset goes here

usethis::use_data(orchid_classes, overwrite = TRUE)
