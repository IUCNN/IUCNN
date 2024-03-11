#' @importFrom sf st_as_sf st_transform st_coordinates st_as_sf
#' st_convex_hull st_area st_union

simProjWiz <- function(thepoints, thecentre){
  #setup and set projection to WGS84
  projcrs <- "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs" # geographic and WGS 84
  thepoints <- sf::st_as_sf(x = thepoints,
                        coords = c("long", "lat"),
                        crs = projcrs)


  #depending on centre point
  if ((thecentre$lat < 70) & (thecentre$lat > -70)) {
    CRS.new <- paste("+proj=cea +lon_0=", thecentre$long,   " +lat_ts=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs",sep = "")
  } else {
    CRS.new <- paste("+proj=laea +lat_0=", thecentre$lat," +lon_0=", thecentre$long, " +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs",sep = "")
  }

  #reproject
  xysp <- sf::st_transform(thepoints, CRS.new)
  xy <- sf::st_coordinates(xysp)
  #rename to x and y as not longer lat long
  colnames(xy) <- c("x","y")
  return(xy)
}



# -----

trueCOGll <-function(thepoints){

  llrad <- deg2rad(thepoints) #to radians
  cartp <- ll2cart(llrad$lat,llrad$long) #to cartesian
  mp <- data.frame(x=mean(cartp$x),y=mean(cartp$y),z=mean(cartp$z)) #central point
  pmp <- pro2sph(mp$x,mp$y,mp$z) #projection to surface
  pmprll <- cart2ll(pmp$x,pmp$y,pmp$z) #to ll in radians
  pmpll <- rad2deg(pmprll) #to degrees
  return(data.frame(lat=pmpll$latr,long=pmpll$longr))

}




trueCOGll <- function(thepoints) {
  llrad <- deg2rad(thepoints) #to radians
  cartp <- ll2cart(llrad$lat, llrad$long) #to cartesian
  mp <-
    data.frame(x = mean(cartp$x),
               y = mean(cartp$y),
               z = mean(cartp$z)) #central point
  pmp <- pro2sph(mp$x, mp$y, mp$z) #projection to surface
  pmprll <- cart2ll(pmp$x, pmp$y, pmp$z) #to ll in radians
  pmpll <- rad2deg(pmprll) #to degrees
  return(data.frame(lat = pmpll$latr, long = pmpll$longr))

}


ll2cart <- function(latr, longr) {
  x <- cos(latr) * cos(longr)
  y <- cos(latr) * sin(longr)
  z <- sin(latr)
  return(data.frame(x, y, z))
}


cart2ll <- function(x, y, z) {
  latr <- asin(z)
  longr <- atan2(y, x)
  return(data.frame(latr, longr))
}

pro2sph <- function(x, y, z) {
  sc <- 1 / sqrt(x ^ 2 + y ^ 2 + z ^ 2)
  x <- x * sc
  y <- y * sc
  z <- z * sc
  return(data.frame(x, y, z))
}

rad2deg <- function(rad) {
  (rad * 180) / (pi)
}
deg2rad <- function(deg) {
  (deg * pi) / (180)
}
MER <- function(thepoints){
  xmin <- min(thepoints[1])
  xmax <- max(thepoints[1])
  ymin <- min(thepoints[2])
  ymax <- max(thepoints[2])
  return(c(xmin,xmax,ymin,ymax))
}


EOOarea <- function(thepoints, thecentre) {
  if ((thecentre$lat < 70) & (thecentre$lat > -70)) {
    projcrs <- paste("+proj=cea +lon_0=", thecentre$long,   " +lat_ts=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs",sep = "")
  } else {
    projcrs <- paste("+proj=laea +lat_0=", thecentre$lat," +lon_0=", thecentre$long, " +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs",sep = "")
  }
  points_sf <- sf::st_as_sf(x = as.data.frame(thepoints),
                            coords = c("x", "y"),
                            crs = projcrs)
  EOOpolyid <- sf::st_convex_hull(sf::st_union(points_sf))
  harea <- sf::st_area(EOOpolyid)
  #check for Area = NA ie when we only have one point
  if (is.na(harea)) {harea <- 0}
  return(as.numeric(harea))
}


AOOsimp <- function(thepoints,cellsize){
  bottomleftpoints <- floor(thepoints/cellsize)
  uniquecells <- unique(bottomleftpoints)
  #cellp <- data.frame(x=(uniquecells$x * cellsize), y=(uniquecells$y * cellsize))
  return(nrow(uniquecells))
}


Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}



EOORating <- function(EOOArea, abb){
  if(missing(abb)){
    abb = TRUE
  }
  #  EOOArea <- 250
  #  abb <- FALSE
  #make positive
  EOOArea <- sqrt(EOOArea * EOOArea)
  cat <- NA
  if (identical(abb,FALSE)) {
    if (EOOArea < 100){
      cat <- "Critically Endangered"
    } else if (EOOArea < 5000){
      cat <- "Endangered"
    } else if (EOOArea < 20000){
      cat <- "Vulnerable"
    } else if (EOOArea < 30000){
      cat <- "Near Threatened"
    } else
      cat <- "Least Concern"

  } else {
    if (EOOArea < 100){
      cat <- "CR"
    } else if (EOOArea < 5000){
      cat <- "EN"
    } else if (EOOArea < 20000){
      cat <- "VU"
    } else if (EOOArea < 30000){
      cat <- "NT"
    } else
      cat <- "LC"
  }
  return (cat)
}

AOORating <- function(AOOArea,abb){
  if(missing(abb)){
    abb = TRUE
  }
  cat <- NA
  cat <- EOORating(AOOArea*10,abb)
}
