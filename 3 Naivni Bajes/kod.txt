
# RADI SA FAKTOR VARIJABLAMA!
# MOZE DA RADI I SA NUMERICKIM AKO IMAJU NORMALNU RASPODELU!
# AKO NEMAJU, MORAJU DA SE DISKRETIZUJU!

dataNB <- read.csv("winemag-data-130k-v2-custom.csv", stringsAsFactors = FALSE)
str(dataNB)

dataSub <- subset(dataNB, (country == "France" | country == "Argentina" | country == "Italy"))
summary(dataSub)

apply(dataSub, 2, FUN = function(x) length(which(x == "")))
apply(dataSub, 2, FUN = function(x) length(which(x == "-")))
apply(dataSub, 2, FUN = function(x) length(which(is.na(x))))

dataSub$region_2 <- NULL

# region 2 izbacujemo jer nam ne sluzi ni cemu
#Varijable koje imaju prazan string su designation, region1, a NA vrednosti ima price

length(unique(dataSub$designation))
# posto designation ima 1877 razl vrednosti a ukupan br observacije je 3427 nema poente 
# da je pretvorimo u faktor zato cemo je ukloniti
dataSub$designation <- NULL
length(unique(dataSub$region_1))
# posto ima mnogo faktora brisemo
dataSub$region_1 <- NULL
dataSub$description <- NULL #description nema uticaja na dalje analze

dataSub$country <- as.factor(dataSub$country)
dataSub$title <- NULL
length(unique(dataSub$province))
dataSub$province <- as.factor(dataSub$province)
str(dataSub)
length(unique(dataSub$variety))
dataSub$variety <- as.factor(dataSub$variety)
length(unique(dataSub$winery))
dataSub$winery <- NULL
str(dataSub)

shapiro.test(dataSub$price)
# nema normalnu menjamo medijanom

medijana <- median(dataSub$price, na.rm = TRUE)
dataSub$price[is.na(dataSub$price)] <- medijana

str(dataSub)
prvikvartil <- quantile(dataSub$price, 0.25)
prvikvartil
dataSub$price_category <- ifelse(dataSub$price <= prvikvartil, yes = "cheap", no = "not_cheap")
dataSub$price <- NULL
dataSub$price_category <- as.factor(dataSub$price_category)

str(dataSub)

# zavrsili smo sa sredjivanjem

shapiro.test(dataSub$points)
# Nema normalnu raspodelu, pa radimo diskretizaciju !

install.packages("bnlearn")
library(bnlearn)
dataSub$points <- as.numeric(dataSub$points)
points.df <- as.data.frame(dataSub$points)
# pretvorili smo points u numeric data frame jer funckija discretized (ova ispod)
# prima to kao parametar

discretized <- discretize(data = points.df,
                          method = "quantile",
                          breaks = c(5))
# diskretizujemo sve koje nemaju normalnu raspodelu
# npr da je bilo vise kolona onda bi bilo
# discretized <- discretize(carseats[,c(2,3,6,7,9)],method = "quantile",breaks = c(5,2,5,2,5)) stavljamo 5, ako javlja gresku onda 2

levels(discretized$`dataSub$points`)

newData <- as.data.frame(cbind(dataSub[,1:5],discretized))
# spojimo ove sa normalnom i faktor varijable sa ovom diskretizovanom u newData
str(newData)
summary(newData)
newData$points <- newData$`dataSub$points`
newData$`dataSub$points` <- NULL


#train i test
library(caret)
set.seed(10)
indexes <- createDataPartition(newData$price_category, p= 0.80, list = F)
train.set <- newData[indexes,]
test.set <- newData[-indexes,]


install.packages("e1071")
library(e1071)
nb1 <- naiveBayes(price_category ~ ., data = train.set)
nb1
nb1.pred <- predict(nb1, newdata = test.set, type = "class")
nb1.cm <- table(true = test.set$price_category, predicted = nb1.pred)
nb1.cm
#pozitivna klasa je cheap

getEvaluationMetrics <- function(cm) {
  
  TP <- cm[2,2] # true positive
  TN <- cm[1,1] # true negative
  FP <- cm[1,2] # false positive
  FN <- cm[2,1] # false negative
  
  accuracy <- sum(diag(cm)) / sum(cm)
  precision <- TP / (TP + FP)
  recall <- TP / (TP + FN)
  F1 <- (2 * precision * recall) / (precision + recall)
  
  c(Accuracy = accuracy, 
    Precision = precision, 
    Recall = recall, 
    F1 = F1)
}

nb1.eval <- getEvaluationMetrics(nb1.cm)
nb1.eval

# sad trazimo threshold preko ROC krive
# to je optimalna verovatnoca za specificity i sensitivity
# onda se pravi nova predikcija za ROC krivu, ali TYPE = RAW
nb2.pred.prob <- predict(nb1, newdata = test.set, type = "raw")
nb2.pred.prob

#kreiranje ROC krive
install.packages("pROC")
library(pROC)
nb2.roc <- roc(response = as.numeric(test.set$price_category),
               predictor = nb2.pred.prob[,2],
               levels = c(1,2))
# kada je no pozitivna klasa onda ide predictor = nb2.pred.prob[,1],levels = c(2,1)
nb2.roc$auc
# sto je area under the curve veca, to je bolja 0.8434 je u nasem slucaju
# sada pravimo plot da bismo videli threshold
plot.roc(nb2.roc, print.thres = TRUE, print.thres.best.method = "youden")
# treshold je 0.654, specificity je 0.783, a sensitivity je 0.747

nb2.coords <- coords(nb2.roc, 
                     ret = c("accuracy", "spec", "sens", "thr"), 
                     x = "local maximas")
nb2.coords
# ovo radimo da bismo izabrali najbolji threshold da maximiziramo specificity i sensitivity

prob.threshold <- nb2.coords[54,4]
prob.threshold
# ovaj je najbolji, 54. red i 4. kolona

nb2.pred <- ifelse(test = nb2.pred.prob[,2] >= prob.threshold, yes = "Cheap", no = "Not_cheap")

nb2.pred <- as.factor(nb2.pred)
nb2.cm <- table(true = test.set$price_category, predicted = nb2.pred)
nb2.cm

nb2.eval <- getEvaluationMetrics(nb2.cm)
nb2.eval
data.frame(rbind(nb1.eval, nb2.eval), row.names = c("one", "two"))



