
cnn_features <- function(x,
                         res,
                         ras,
                         species = "species",
                         lon = "decimallongitude",
                         lat = "decimallatitude",
                         type){
  # generate the background equal area raster for madagascar
  mdg <- ne_countries(country = "Madagascar", returnclass = "sp")

  # transform to Tananarive Laborde CRS https://epsg.io/8441
  mdg2 <-  spTransform(mdg, CRS("+init=epsg:8441"))

  mdg.ras <- raster(xmn = 0,
                    xmx = 850000,
                    ymn = ymin(mdg2),
                    ymx = ymax(mdg2),
                    resolution = 10000,
                    crs = proj4string(mdg2))


  test <- rasterize(mdg2, mdg.ras)

  plot(test)

  # rasterize species occurrences
  ## load occurrence data
  load(file = "output/filtered_occurrences.rda")

  ##reprodject
  pts <- SpatialPoints(filtered_occurrences[, c("decimallongitude", "decimallatitude")], proj4string = CRS("+init=epsg:4326"))
  pts <- spTransform(pts, CRS("+init=epsg:8441"))
  pts <- data.frame(species = filtered_occurrences[, "species"], coordinates(pts))

  ## rasterize
  li <- sort(unique(pts$species))
  abundance_raster <- list()

  for( i in 1:length(li)){
    print(i)
    sub <- pts[pts$species == li[i],]
    abundance_raster[[i]] <- rasterize(sub[, 2:3], mdg.ras, fun = "count")
    abundance_raster[[i]][is.na(abundance_raster[[i]])] <-  0
  }

  names(abundance_raster) <- li
  abundance_raster <- lapply(abundance_raster, "as.matrix")

  save(abundance_raster, file = "output/iu-cnn_abundance_raster.rda")

  ## presence/absence rasters
  # presence_raster <- lapply(abundance_raster, function(k){k[k > 1] <- 1})
  presence_raster <-  abundance_raster
  for(i in 1:length(abundance_raster)){
    sub <- presence_raster[[i]]
    sub[sub > 1] <- 1
    presence_raster[[i]] <- sub
  }

  save(presence_raster, file = "output/iu-cnn_presence_raster.rda")
}
