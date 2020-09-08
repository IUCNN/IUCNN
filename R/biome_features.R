
#Copy the download code from speciesgeocoder


# ## biomes from https://www.worldwildlife.org/publications/terrestrial-ecoregions-of-the-world
# wwf <- st_read(dsn = "input/additional_nnfeatures/wwf_biomes", layer = "wwf_terr_ecos")
# pts <- st_as_sf(ass_occ,
#                 coords = c("decimallongitude", "decimallatitude"),
#                 crs= st_crs(wwf))
#
# biom <- st_intersects(pts, wwf) # this gives the rownames of each point in wwf
#
# st_geometry(wwf) <- NULL
# biom2 <- wwf[as.numeric(biom),c("ECO_NAME", "BIOME")]
#
# biom <- bind_cols(ass_occ %>% dplyr::select(species = canonical_name),
#                   biom2) %>%
#   dplyr::select(-ECO_NAME) %>%
#   distinct() %>%
#   mutate(presence = 1) %>%
#   tidyr::pivot_wider(id_cols = species, names_from = BIOME, values_from = presence) %>%
#   replace(is.na(.), 0) %>%
#   dplyr::select(-`NA`)
