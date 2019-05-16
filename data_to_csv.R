### Helper script to load the raw data from the study and convert it to csv files for python use ###

source("./Script/load.R")
source("./Script/cross-validation.R")
source("./Script/traintest.R")

out=DataGenerate('./Data/Data.Rda', T)

multiomics_measures <- c('CellfreeRNA','PlasmaLuminex','SerumLuminex','Microbiome','ImmuneSystem','Metabolomics', 'PlasmaSomalogic')

multiomics_data <- list()

for (i in seq(length(out$features))){
    multiomics_data[[multiomics_measures[i]]] <- as.data.frame(out$features[i])
    write.csv(as.data.frame(out$features[i]), paste('./Data/', multiomics_measures[i], '.csv', sep = ''))
}

write.csv(out$featureweeks, './Data/featureweeks.csv', row.names=FALSE, na="")
write.csv(out$featureindex, './Data/featureindex.csv', row.names=FALSE, na="")
write.csv(out$featurepatients, './Data/featurepatients.csv', row.names=FALSE, na="")
write.csv(out$featuretimes, './Data/featuretimes.csv', row.names=FALSE, na="")
