#################################################################################
#This function normalizes the data by subtracting mean (C) and dividing by standard deviation (SD). 
scalePR <- function(X, C, SD)
{
  return(sweep(sweep(X,2,C), 2,SD, '/'))
}

#################################################################################
#Given the training data X, and labels Y, with parameters stored in parm, trains and model and produces test performance.
MethodTrainReport <- function(X, Y, Test, parm)
{
  if(parm$method == 'glmnet')
  {  
    library(glmnet)  
    gm<-glmnet(X, Y, standardize=FALSE, alpha=parm$a, lambda=parm$l)
    if(sum(predict(gm, type="coefficients",lambda=l)[-1]!=0)<1)
    {
      TestPredict = rep(0, nrow(Test))
    }
    else
    {
      TestPredict = predict(gm, Test, lambda=parm$l)[,1]
    }
    if(parm$coef == TRUE)
    {
      out = list()
      out$TestPredict = TestPredict
      out$coefs = predict(gm, type="coefficients",lambda=l)[-1]
      return(out)
    }
  }
  if(parm$method == 'gausspr')
  {
    library(kernlab)
    gp<-gausspr(X, Y, kernel=parm$kernel, variance.model = TRUE, scaled = FALSE,  var=parm$var)
    TestPredict = predict(gp, Test, type = "response")
  }
  if(parm$method == 'ksvm')
  {
    library(kernlab)
    svr<-ksvm(X, Y, kernel=parm$kernel, kpar = "automatic", C = parm$c, cache = 100, type='eps-svr')
    TestPredict = predict(svr, Test)
  }
  if(parm$method == 'XGB')
  {
    library(xgboost)
    param <- list(max_depth = 2, silent = 1, nthread = 4, objective = "reg:linear", eval_metric = "rmse", min_child_weight=14, num_parallel_tree=4,
                  eta=parm$e, alpha=parm$a)
    dtrain = xgb.DMatrix(data=X, label= Y)
    xgb<-xgb.train(param,  booster = "gbtree", data = dtrain, nrounds = 7)
    TestPredict = predict(xgb, Test)
  }
  if(parm$method == 'RF')
  {  
    library(randomForest)
    rf=randomForest(X, Y, proximity=TRUE, importance=FALSE, norm.votes=TRUE)    
    TestPredict = predict(rf, Test)
  }  
  return(TestPredict)
}

#################################################################################
#This functions uses a two-layer leave-one-patient-out cross validation to optimize the hyper parameters of each estimator
Train2layer <- function(X, Y, foldid, parm)
{ 
  library(matrixStats)
  Xpredict = vector()
  coefs = matrix(nrow = max(foldid), ncol = ncol(X))
  for (j in seq(max(foldid))) 
  {
    kfold=crossvalid(foldid, j)
    Xtrain=X[kfold$kfoldidx,]    
    Ytrain=Y[kfold$kfoldidx]
    Xtest=X[kfold$Testid,]
    if(parm$scale == TRUE)
    {
      Ctrain=colMeans(Xtrain)
      SDtrain=colSds(Xtrain)
      ind0=which(SDtrain!=0)
      Xtrain=scalePR(Xtrain[,ind0], Ctrain[ind0], SDtrain[ind0])    
      Xtest=scalePR(Xtest[,ind0], Ctrain[ind0], SDtrain[ind0])
    }
    if(parm$coef == TRUE)
    {
      mout = MethodTrainReport(Xtrain, Ytrain, Xtest, parm)
      Xpredict[kfold$Testid] = mout$TestPredict
      if(parm$scale == TRUE)
      {
        coefs[j, ind0] = mout$coefs
      }
      else
      {
        coefs[j, ] = mout$coefs
      }
    }
    else
    {
      Xpredict[kfold$Testid] = MethodTrainReport(Xtrain, Ytrain, Xtest, parm)
    }    
  }
  if(parm$coef == TRUE)
  {
    out = list()
    out$Xpredict = Xpredict
    out$coefs = colMeans(coefs)
    return(out)
  }
  else
  {
    return(Xpredict)
  }
}

#################################################################################
#This function splits the data into training data (to be used for the second layer), and test to report the validation result.
wrapTrain1layer <- function(X, Y, foldid, parm, j)
{
  library(matrixStats)
  kfold=crossvalid(foldid, j)
  Xtrain=X[kfold$kfoldidx,]
  Ytrain=Y[kfold$kfoldidx]
  Xtest=X[kfold$Testid,]
  return(Train2layer(Xtrain, Ytrain, kfold$kfold, parm))
}

#################################################################################
#This function uses parallel computing to calculate scores for a broad range of hyperparameter values for the inner-layer cross-validation.
Train1layer <- function(X, Y, foldid, Ncpu, parm)
{ 
  library(snow)  
  clus <- makeCluster(Ncpu)
  clusterExport(clus,"scalePR", envir=environment())
  clusterExport(clus,"Train2layer", envir=environment())
  clusterExport(clus,"MethodTrainReport", envir=environment())
  clusterExport(clus,"wrapTrain1layer", envir=environment())
  clusterExport(clus,"X", envir=environment())
  clusterExport(clus,"Y", envir=environment())
  clusterExport(clus,"parm", envir=environment())
  clusterExport(clus,"crossvalid", envir=environment())
  clusterExport(clus,"foldid", envir=environment())
  Xpredict= parRapply( clus, data.frame( a=seq(max(foldid)) ), function(x) wrapTrain1layer(X, Y, foldid, parm, x[1]) )
  stopCluster(clus)
  return( Xpredict )
}

#################################################################################
#This function uses the optimized hyperparameters from the second layer of leave-one-patient-out cross-validation to predict the test data in the first layer.
wrapTest1layer <- function(X, Y, foldid, parm)
{
  library(matrixStats)
  Xpredict=vector()
  coefs = matrix(nrow = max(foldid), ncol = ncol(X))
  for (j in seq(max(foldid))) 
  {
    print(j)
    kfold=crossvalid(foldid, j)
    Xtrain=X[kfold$kfoldidx,]    
    Ytrain=Y[kfold$kfoldidx]
    Xtest=X[kfold$Testid,]
    if(parm$scale == T)
    {
      Ctrain=colMeans(Xtrain)
      SDtrain=colSds(Xtrain)
      ind0=which(SDtrain!=0)
      Xtrain=scalePR(Xtrain[,ind0], Ctrain[ind0], SDtrain[ind0])
      Xtest=scalePR(Xtest[,ind0], Ctrain[ind0], SDtrain[ind0])              
    }
    if(parm$coef == TRUE)
    {
      mout = MethodTrainReport(Xtrain, Ytrain, Xtest, parm)
      Xpredict[kfold$Testid] = mout$TestPredict
      if(parm$scale == TRUE)
      {
        coefs[j, ind0] = mout$coefs
      }
      else
      {
        coefs[j, ] = mout$coefs
      }
    }
    else
    {
      Xpredict[kfold$Testid] = MethodTrainReport(Xtrain, Ytrain, Xtest, parm)
    }    
  }
  out = list()
  out$Testpredict=Xpredict
  out$Testpval=-log10( cor.test(Xpredict, Y, method='spearman')$p.value )
  if(parm$coef == TRUE)
  {
    out$coefs = colMeans(coefs)
  }
  parm$coef = FALSE
  myout=Train1layer(X, Y, foldid, 20, parm)
  print(length(myout))
  Xpredict=matrix( 0, nrow = max(foldid), ncol = length(Y) )
  nrp=sum(foldid==1)* (max(foldid)-1)
  for (j in seq(max(foldid))) 
  {  
    kfold=crossvalid(foldid, j)
    Xpredict[j,kfold$kfoldidx]=myout[((j-1)*nrp+1):(j*nrp)]
  }
  out$Trainpredict=colMedians(Xpredict)
  out$Trainpval=-log10( cor.test(colMedians(Xpredict), Y, method='spearman')$p.value )
  return( out )  
}

#################################################################################
#This function initializes the necessary parameters of Elastic Net (particularly alpha and lambda) from the RDS files to pass them to the wrapTest1layer function.
TestGLM <- function(out, Y, foldid, wfile, scale=TRUE, coef, pind)
{
  library(matrixStats)
  parm = list()
  X=out$features[[pind]]
  if(pind==6) scale = F
  parm$scale = scale
  parm$method = 'glmnet'
  parm$coef = coef
  alpha = seq(0,1,.1)
  lambda = c( seq(0,1,.025), seq(12) )
  pval=readRDS(wfile)
  ind= which(max(pval, na.rm=T)==pval, arr.ind=TRUE)
  
  parm$a = alpha[ind[1,1]]
  parm$l = lambda[ind[1,2]]
  
  return(wrapTest1layer(X, Y, foldid, parm))
}

#################################################################################
#This function reports the validation result of the full stacked generalization model.
wrapBoost1layer <- function(X, Y, Test, foldid, parm)
{
  library(matrixStats)
  Xpredict=vector()
  for (j in seq(max(foldid))) 
  {
    print(j)
    kfold=crossvalid(foldid, j)
    Xtrain=X[kfold$kfoldidx,]    
    Ytrain=Y[kfold$kfoldidx]
    Xtest=Test[kfold$Testid,]
    if(parm$scale == T)
    {
      Ctrain=colMeans(Xtrain)
      SDtrain=colSds(Xtrain)
      ind0=which(SDtrain!=0)
      Xtrain=scalePR(Xtrain[,ind0], Ctrain[ind0], SDtrain[ind0])
      Xtest=scalePR(Xtest[,ind0], Ctrain[ind0], SDtrain[ind0])              
    }
    Xpredict[kfold$Testid]=MethodTrainReport(Xtrain, Ytrain, Xtest, parm)
  }
  out = list()
  out$Testpredict=Xpredict
  out$Testpval=-log10( cor.test(Xpredict, Y, method='spearman')$p.value )
  
  myout=Train1layer(X, Y, foldid, 20, parm)
  Xpredict=matrix( 0, nrow = max(foldid), ncol = length(Y) )
  nrp=sum(foldid==1)* (max(foldid)-1)
  for (j in seq(max(foldid))) 
  {  
    kfold=crossvalid(foldid, j)
    Xpredict[j,kfold$kfoldidx]=myout[((j-1)*nrp+1):(j*nrp)]
  }
  out$Trainpredict=colMedians(Xpredict)
  out$Trainpval=-log10( cor.test(colMedians(Xpredict), Y, method='spearman')$p.value )
  return( out )  
}

#################################################################################
#Builds the final stacked generalization model using already calculated hyperparameters.
BoostGLM <- function(X, Y, Test, foldid, wfile, scale, coef)
{
  library(matrixStats)
  parm = list()
  parm$scale = scale
  parm$method = 'glmnet'
  parm$coef= coef
  
  alpha = seq(0,1,.1)
  lambda = c( seq(0,1,.025), seq(12) )
  pval=readRDS(wfile)
  ind= which(max(pval, na.rm=T)==pval, arr.ind=TRUE)
  
  parm$a = alpha[ind[1,1]]
  parm$l = lambda[ind[1,2]]
  
  return(wrapBoost1layer(X, Y, Test, foldid, parm))
}
#################################################################################
