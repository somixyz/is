
# RADI SAMO SA NUMERICKIM PODACIMA!

# da je pisalo da treba da se ucita iz ISLR paketa onda treba
# install.packages('ISLR')
# library(ISLR)
# str(imeDataSeta)
# dataSet <- imeDataSeta

data <- read.csv("Video_Games_Sales_2017_reduced.csv")

dataSub <- subset(data, (data$Platform == "PS2" | data$Platform == "PS3" | data$Platform == "PS3"))
#NA vrednosti postoje u okviru varijable Critic_Score i ima ih 1514, Critic_Count 1514, User_Score 1175 i User_Count 1344
#zakljucak je da su ovo numericke varijable sa mnogo nedostajucih vrednosti

apply(dataSub, 2, FUN = function(x) length(which(is.na(x))))
apply(dataSub, 2, FUN = function(x) length(which(x == "")))
apply(dataSub, 2, FUN = function(x) length(which(x == "-")))
str(dataSub)
# varijable User_ScoreDeveloper i Rating imaju 1175 1162 i 1191 praznih stringova respektivno

# varijable Platform, Name,Genre, Publisher cemo izostaviti iz daljeg istfrazivanja jer 
# nam je za lin reg potrebno samo num
dataSub$Name <- NULL
dataSub$Platform <- NULL
dataSub$Publisher <- NULL

length(unique(dataSub$Developer))
length(unique(dataSub$Rating))
dataSub$Rating <- NULL
dataSub$Developer <- NULL
# rating i developer imaju previse praznih stringova, 
# i nisu nam neophodne u daljem istrazivanju s obzirom da bi bile kategorijske varijable
dataSub$Genre <- NULL
# Genre nam nije potrebno za lin reg i nju izbacujemo

apply(dataSub, 2, FUN = function(x) length(which(is.na(x))))
apply(dataSub, 2, FUN = function(x) length(which(x == "N/A")))
apply(dataSub, 2, FUN = function(x) length(which(x == "-")))
# sredicemo year_of_release jer ima N/A vrednosti
# sredicemo critic_score, critic_count i user_count jer ima NA vrednosti
# sredicemo user_score jer ima prazne stringove

#YEAR OF RELEASE IMA N/A VREDNOSTI
dataSub$Year_of_Release[dataSub$Year_of_Release == "N/A"] <- NA
dataSub$Year_of_Release <- as.integer(dataSub$Year_of_Release)
dataSub$User_Score[dataSub$User_Score == ""] <- NA
dataSub$User_Score <- as.numeric(dataSub$User_Score)

apply(dataSub[,c("Year_of_Release", "Critic_Score", "Critic_Count", "User_Score", "User_Count")], 2, FUN = function(x) shapiro.test(x))
# ni jedna nema normalnu raspodelu, pa ih menjamo njihovom medijanom

medianYear <- median(dataSub$Year_of_Release, na.rm = T)
medianUserScore <- median(dataSub$User_Score, na.rm = T)
medianCriticScore <- median(dataSub$Critic_Score, na.rm = T)
medianUserCount <- median(dataSub$User_Count, na.rm = T)
medianCriticCount <- median(dataSub$Critic_Count, na.rm = T)


dataSub$Year_of_Release[is.na(dataSub$Year_of_Release)] <- medianYear
dataSub$User_Score[is.na(dataSub$User_Score)] <- medianUserScore
dataSub$Critic_Score[is.na(dataSub$Critic_Score)] <- medianCriticScore
dataSub$User_Count[is.na(dataSub$User_Count)] <- medianUserCount
dataSub$Critic_Count[is.na(dataSub$Critic_Count)] <- medianCriticCount
# zavrseno sredjivanje

library(corrplot)
matrica <- cor(dataSub)
matrica[9,] # izabrali smo 9. kolonu/varijablu User_Score
corrplot(matrica, method = "number", type = "upper")
# znacajnu korelaciju u odnosu na User_Score ima Critic_Score = 0.49, to je koeficijent korelacije

library(caret)
set.seed(1010)
indexes <- createDataPartition(dataSub$User_Score, p = 0.8, list = FALSE)
train.data <- dataSub[indexes, ]
test.data <- dataSub[-indexes, ]

# sad za lm uzimamo samo ove koje imaju jacu korelaciju, u ovom slucaju samo Critic_Score
lm1 <- lm(User_Score ~ Critic_Score, data = dataSub)
summary(lm1)
# za svako povecanje Critic_Score povecava nam se User_Score za 0.056
# izgled lin krive y = 0.056x + 3.414907
# residual predstavlja razliku izmedju predvidjenih i stvarnih vrednosti
# r-squared, nas model opisuje 24.3% varijabilieteta zavisne promenljive
# f-statistika je 868.7, a p-value < 0.05, dakle postoji zavisnost izmedju ove dve varijable

# sad proveravamo multikolinearnost AKO IMA VISE VARIJABLI, OVDE NEMA PA NECE RADITI
install.packages("car")
library(car)
vif(lm1)
# ovde imamo samo jednu varijablu, Critic_Score, pa nece raditi !

# sad pravimo novi model bez limita
lm2 <-  lm(User_Score ~ Critic_Score+NA_Sales + EU_Sales + JP_Sales+Other_Sales 
           + Global_Sales + Critic_Count + User_Count + Critic_Score,
           data = dataSub)
summary(lm2)
# komentari

# OVDE IMA VISE VARIJABLI PA RADIMO PROVERU MULTIKOLINEARNOSTI
install.packages("car")
library(car)
vif(lm2)
# ako su korelacije vece od 4, problematicne su
# postoji velika multikolinearnost izmedju NA_Sales, EU_Sales, JP_Sales, Other_Sales i Global_Sales sa User_Score
# pa zbog toga njih izbacujemo !

lm3 <-  lm(User_Score ~ Critic_Score + Critic_Count + User_Count,
           data = dataSub)
summary(lm3)
# vidimo da Critic_Count nije statisticki znacajna jer jer p > 0.05
# a ovo smo mogli da vidimo i iz summary(lm2)

lm4 <-  lm(User_Score ~ Critic_Score + User_Count,
           data = dataSub)
summary(lm4)
# komentari, R-squared nam je 24.24%

coef(lm4)

# pravimo 4 plota
par(mfrow = c(2,2))
plot(lm4)
# Prva slika govori koliko je prepostavka o linearnosti zadovoljenja, 
# predikcija se moze smatrati merodavnom jer je crvena linija blizu toga da bude ravna, odnosno tackice 
# su blizu toga da budu jednako rasporedjene

# druga slika govori o tome da li su rezidulai normalno rasporedjeni
# U ovom slucaju nisu jer odstupaju u odnos na isprekidanu liniju

# treca slika proverava da li rezidulali imaju jednake varijanse, nemaju (jer nije linija ravna)

# cetvrta da li ima observacija sa veoma velikim/malim vrednostima tj. ekstemnim vrednostima, ovde vidimo da ih ima

lm4.pred <- predict(lm4, newdata = test.data)
head(lm4.pred)

RSS <- sum((lm4.pred - test.data$User_Score)^2)
TSS <- sum((mean(train.data$User_Score) - test.data$User_Score)^2)
rsquared <- 1 - RSS / TSS
rsquared
# uporedjujemo sa rsquared nad trainom i nad testom vidimo da je veca na testu
# Ukupan objasnjeni varijabilitet je 24.24% sto je manje nego u test setu gde je 28.28%

RMSE <- sqrt(RSS/nrow(test.data))
RMSE
mean(test.data$User_Score)
RMSE/mean(test.data$User_Score)
# greska iznosi 13.74% od srednje vrednosti









