
    library(rredlist)
    
    args = commandArgs(trailingOnly = TRUE)
    taxon_group = args[1]
    group_rank = args[2]
    iucn_key = args[3]
    outdir = args[4]
    exclude_extinct = FALSE
    
    # load all IUCN data
    data = c()
    for (i in seq(0, 20, 1)){
      data = c(data,c(rl_sp(key=iucn_key,page = i)))
    }
    
    # get taxon list, class list and status list from data
    taxon_list = c()
    group_list = c()
    status_list = c()
    taxon_id = c()
    for (page in data){
      if (length(page) > 1){
        target_column = which(startsWith(colnames(page),group_rank))
        taxon_list = c(taxon_list,page$scientific_name)
        group_list = c(group_list,page[,target_column])
        status_list = c(status_list,page$category)
        taxon_id = c(taxon_id,page$taxonid)
      }
    }
    
    # exclude extinct taxa if needed
    if (exclude_extinct){
      boolean = !grepl('EX|EW',status_list)
      taxon_list = taxon_list[boolean]
      group_list = group_list[boolean]
      status_list = status_list[boolean]
      taxon_id = taxon_id[boolean]
    }

    # remove all non-species level identifications
    boolean = !grepl('subsp.|ssp.|subpopulation|Subpopulation',taxon_list)
    taxon_list = taxon_list[boolean]
    group_list = group_list[boolean]
    status_list = status_list[boolean]
    taxon_id = taxon_id[boolean]
    
    # select target taxa
    selected_taxon_list = taxon_list[group_list==taxon_group]
    selected_ids = taxon_id[group_list==taxon_group]
    final_sorted_taxon_list = selected_taxon_list
    #final_taxon_list = as.data.frame(cbind(selected_taxon_list,selected_ids))
    #final_sorted_taxon_list = final_taxon_list[order(final_taxon_list$selected_taxon_list),]
    write.table(sort(final_sorted_taxon_list),file=paste0(outdir,'/',taxon_group,"_species_list.txt"), quote=F,row.names=F,sep='	',col.names = FALSE)
    
    
    # get historic data __________________________
    # create new dataframe with species as first column
    historic_assessments = selected_taxon_list
    historic_assessments = as.data.frame(historic_assessments)
    colnames(historic_assessments) = c('species')
    # find historic assessments and fill into dataframe
    log_frequency = 1000
    counter = 1
    for (i in seq(1, length(selected_taxon_list), 1)){
      species = selected_taxon_list[i]
      species_id = selected_ids[i]
      print(paste0('Downloading IUCN history: species ',counter, ' of ',length(selected_taxon_list)))
      #print(species)
      row_id = which(historic_assessments$species == species)
      hist_data <- rl_history(id=species_id,key=iucn_key)
      for (year in hist_data$result$year){
        id = which(hist_data$result$year == year)
        #some species have multiple assignments for some years
        if (length(hist_data$result$code[id])>1){
          historic_assessments[row_id,year] <- hist_data$result$code[id][1]
        }
        else{
          historic_assessments[row_id,year] <- hist_data$result$code[id]
        }
      }
    if (counter %% log_frequency == 0){
      write.table(historic_assessments,file=paste0(outdir,'/',taxon_group,"_iucn_history.txt"), quote=F,row.names=F,sep='	')
    }
      counter = counter+1
    }
    write.table(historic_assessments,file=paste0(outdir,'/',taxon_group,"_iucn_history.txt"), quote=F,row.names=F,sep='	')
    #___________________________________    
    