################################
ConBatch <- function(taxa, lat, long, cellsize) {
  #sort data out
  mypointsll <- data.frame(taxa, lat, long)
  specieslist <- unique(mypointsll$taxa)

  #make a dataframe to store results
  resultsdf <-
    data.frame(
      taxa = character(),
      NOP = integer(),
      MER = double(),
      EOOkm2 = double(),
      AOOkm = double(),
      EOOcat = character(),
      AOOcat = character(),
      stringsAsFactors = FALSE
    )
  #rename the AOOfield to use the cellsize in km
  names(resultsdf)[5] <- paste("AOO", cellsize / 1000, "km", sep = "")

  #loop thought all taxa
  for (taxa in specieslist) {
    print(paste("Processing", taxa))
    #get just one taxa to work on, if already projected just taxa get if not projected, then project and then get taxa
    taxapointsll <-
      (mypointsll[mypointsll$taxa == taxa, c("lat", "long")])
    #reproject points
    thecenter <- trueCOGll(taxapointsll)
    taxapointsxy <- simProjWiz(taxapointsll, thecenter)

    #CALCULATE METRICS
    #number of points
    nop <- nrow(taxapointsxy)
    #Minimium enclosing rectangle
    MERps <- MER(taxapointsxy) / 1000
    MERarea <- (MERps[2] - MERps[1]) * (MERps[4] - MERps[3])
    #EOO
    EOOa <- EOOarea(taxapointsxy, thecenter) / 1000000
    #AOO with cellsize
    AOOa <- AOOsimp(taxapointsxy, cellsize) * (cellsize / 1000) ^ 2
    #EOO IUCN category
    EOOcat <- EOORating(EOOa)
    #AOO IUCn category
    AOOcat <- AOORating(AOOa)
    #population the results data frame
    resultsdf[nrow(resultsdf) + 1, ] <-
      c(taxa, nop, MERarea, EOOa, AOOa, EOOcat, AOOcat)
  } #end of loop
  return(resultsdf)
} #end of function
