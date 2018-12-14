
# Coding assignment 1


set.seed(4007)
# Data Generation
csize = 10;       # number of centers
p = 2;      
s = 1;      # sd for generating the centers within each class                    
m1 = matrix(rnorm(csize*p), csize, p)*s + cbind( rep(1,csize), rep(0,csize));
m0 = matrix(rnorm(csize*p), csize, p)*s + cbind( rep(0,csize), rep(1,csize));


# Generate training data
n=100;  
# Randomly allocate the n samples for class 1  to the 10 clusters
id1 = sample(1:csize, n, replace = TRUE);
# Randomly allocate the n samples for class 1 to the 10 clusters
id0 = sample(1:csize, n, replace = TRUE);  

s= sqrt(1/5);                               # sd for generating x. 

traindata = matrix(rnorm(2*n*p), 2*n, p)*s + rbind(m1[id1,], m0[id0,])
Ytrain = factor(c(rep(1,n), rep(0,n)))


# Generate test data
N = 5000;  
id1 = sample(1:csize, N, replace=TRUE);
id0 = sample(1:csize, N, replace=TRUE); 
testdata = matrix(rnorm(2*N*p), 2*N, p)*s + 
  rbind(m1[id1,], m0[id0,])
Ytest = factor(c(rep(1,N), rep(0,N)))


#knn method
library(class) 

## Choice of the neighhood size. 
## Here I just use the values from the textbook
myk = c(151, 101, 69, 45, 31, 21, 11, 7, 5, 3, 1)
m = length(myk);

train.err.knn = rep(0,m);
test.err.knn = rep(0, m);

for( j in 1:m){
  Ytrain.pred = knn(traindata, traindata, Ytrain, k = myk[j])
  train.err.knn[j] = sum(Ytrain != Ytrain.pred)/(2*n)
  Ytest.pred = knn(traindata, testdata, Ytrain,k = myk[j])
  test.err.knn[j] = sum(Ytest != Ytest.pred)/(2*N)
}

# least square method
RegModel = lm(as.numeric(Ytrain) - 1 ~ traindata)
Ytrain_pred_LS = as.numeric(RegModel$fitted > 0.5)
Ytest_pred_LS = RegModel$coef[1] + RegModel$coef[2] * testdata[,1] + 
  RegModel$coef[3] * testdata[,2]
Ytest_pred_LS = as.numeric(Ytest_pred_LS > 0.5)

## cross tab for training data and training error
table(Ytrain, Ytrain_pred_LS);   
train.err.LS = sum(Ytrain !=  Ytrain_pred_LS) / (2*n);  

## cross tab for test data and test error
table(Ytest, Ytest_pred_LS);     
test.err.LS = sum(Ytest !=  Ytest_pred_LS) / (2*N);


# bayes error
mixnorm=function(x){
  ## return the density ratio for a point x, where each 
  ## density is a mixture of normal with 10 components
  sum(exp(-apply((t(m1)-x)^2, 2, sum)*5/2))/sum(exp(-apply((t(m0)-x)^2, 2, sum)*5/2))
}

Ytest_pred_Bayes = apply(testdata, 1, mixnorm)
Ytest_pred_Bayes = as.numeric(Ytest_pred_Bayes > 1);
table(Ytest, Ytest_pred_Bayes); 
test.err.Bayes = sum(Ytest !=  Ytest_pred_Bayes) / (2*N)


# error curve plot
pdf("AssignmentOutput_1.pdf")

plot(c(0.5,m), range(test.err.LS, train.err.LS, test.err.knn, train.err.knn),
     type="n", xlab="Degree of Freedom", ylab="Error", xaxt="n")

df = round((2*n)/myk)
axis(1, at = 1:m, labels = df)
axis(3, at = 1:m, labels = myk)

points(1:m, test.err.knn, col="red", pch=1);
lines(1:m, test.err.knn, col="red", lty=1); 
points(1:m, train.err.knn, col="blue", pch=1);
lines(1:m, train.err.knn, col="blue", lty=2);

points(3, train.err.LS, pch=2, cex=2, col="blue")
points(3, test.err.LS, pch=2, cex=2, col="red")

abline(test.err.Bayes, 0, col="purple")

legend("bottomleft",1,m,legend=c("Test", "Train"),
       col=c("red", "blue"), lty=1:2, cex=0.8)

dev.off()

