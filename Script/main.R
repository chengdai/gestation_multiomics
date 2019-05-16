remove(list = ls())
source("load.R")
source("cross-validation.R")
source("traintest.R")

out=DataGenerate('../Data/Data.Rda', T)

foldid=as.integer(as.factor(out$featurepatients[out$featureindex]))
set.seed(1000)
featureweeks=out$featureweeks[out$featureindex]

glmn = list()
for(i in seq(length(out$features)))
{
  glmn[[i]] = TestGLM(out, featureweeks, foldid, paste0('../RDSfiles/GLM', i, '.rds'), TRUE, FALSE, i)
}

gmtrpv = vector() 
gmtstpv = vector()

gmBtrain =  glmn[[1]]$Trainpredict
gmBtest = glmn[[1]]$Testpredict

gmtrpv[1] = glmn[[1]]$Trainpval
gmtstpv[1]= glmn[[1]]$Testpval

for(i in c(2:length(out$features)))
{ 
  gmBtrain = cbind(gmBtrain, glmn[[i]]$Trainpredict)
  gmBtest = cbind(gmBtest, glmn[[i]]$Testpredict)
  gmtrpv[i] = glmn[[i]]$Trainpval
  gmtstpv[i] = glmn[[i]]$Testpval
}

gmboost=BoostGLM(gmBtrain, featureweeks, gmBtest, foldid, '../RDSfiles/GLMBoost.rds', TRUE, FALSE) 

gmtrpv[8] = gmboost$Trainpval
gmtstpv[8] = gmboost$Testpval