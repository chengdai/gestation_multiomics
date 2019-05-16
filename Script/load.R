# This function loads the entire dataset.
# This includes: featureweeks (gestational age at time of sampling), featureindex (indexes of samples collected during pregnancy - excludes postpartum), features (different biological measurements), and featurepatients (patients IDs).

DataGenerate<-function(fname, flag=TRUE)
{
  datasets=c('Cellfree RNA','PlasmaLuminex','SerumLuminex','Microbiome','ImmuneSystem','Metabolomics', 'PlasmaSomalogic')
  load(fname)

  
  features=list()
  colind=list()
  if(flag==TRUE){
    featureindex=which(featuretimes!=4)
  }
  else{
    featureindex=which(featuretimes %in% seq(4))
  }
  colind[[1]] = which(colMeans(InputData[[1]])!=0)
  features[[1]] = InputData[[1]][ , colind[[1]]]
  colnames(features[[1]])=iconv(colnames(features[[1]]),"WINDOWS-1252","UTF-8")
    
  
  colind[[2]] = which(colMeans(InputData[[2]])!=0)
  features[[2]] = InputData[[2]][ , colind[[2]]]
  colnames(features[[2]])=iconv(colnames(features[[2]]),"WINDOWS-1252","UTF-8")
  
  
  colind[[3]] = which(colMeans(InputData[[3]])!=0)
  features[[3]] = InputData[[3]][ , colind[[3]]]
  colnames(features[[3]])=iconv(colnames(features[[3]]),"WINDOWS-1252","UTF-8")
  
  
  colind[[4]] = which(colMeans(InputData[[4]])!=0)
  features[[4]] = InputData[[4]][ , colind[[4]]]
  colnames(features[[4]])=iconv(colnames(features[[4]]),"WINDOWS-1252","UTF-8")
  
  
  colind[[5]] = which(colMeans(InputData[[5]])!=0)
  features[[5]] = InputData[[5]][ , colind[[5]]]
  colnames(features[[5]])=iconv(colnames(features[[5]]),"WINDOWS-1252","UTF-8")
  
  
  colind[[6]] = which(colMeans(InputData[[6]])!=0)
  features[[6]] = InputData[[6]][ , colind[[6]]]
  colnames(features[[6]])=iconv(colnames(features[[6]]),"WINDOWS-1252","UTF-8")
  
  
  colind[[7]] = which(colMeans(InputData[[7]])!=0)
  features[[7]] = InputData[[7]][ , colind[[7]]]
  colnames(features[[7]])=iconv(colnames(features[[7]]),"WINDOWS-1252","UTF-8")
  
  out = list()
  out$featureweeks=featureweeks
  out$featureindex=featureindex
  out$features=features
  out$featurepatients = featurepatients
  out$colind = colind
  out$featuretimes = featuretimes[featureindex]
  return(out)
}
