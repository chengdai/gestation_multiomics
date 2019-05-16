#coordinates sample ids across various leave-one-patient-out cross-validation layers.
crossvalid <- function(foldid, testpatient)
{
  bidx=which(foldid==min(foldid))
  eidx=which(foldid==max(foldid))
  
  indxTS = bidx-1+testpatient
  indxCV = which( !c(1:length(foldid)) %in% indxTS )

  kfold = vector()
  for (i in seq(length(bidx)) )
  {
    kfold = c(kfold, seq( (max(foldid)-1) ) )
  }
  out = list()
  out$kfold = kfold
  out$kfoldidx = indxCV
  out$Testid = indxTS
  return(out)
}
