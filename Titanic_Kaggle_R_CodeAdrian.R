
# Adrian Zinovei - Titanic R Code for Own Data Science Project#


library('dplyr')
library('ggplot2')
library('ggthemes')
library('scales')
library('randomForest')
library('rpart')
library('mice')
library('nlme')
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(ggthemes)) install.packages("ggthemes", repos = "http://cran.us.r-project.org")
if(!require(scales)) install.packages("scales", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(nlme)) install.packages("nlme", repos = "http://cran.us.r-project.org")
if(!require(mice)) install.packages("mice", repos = "http://cran.us.r-project.org")


# please make sure you set the working directory and place all datasets in *** Adrian Zinovei#

traindata <- read.csv('titanic_training.csv', stringsAsFactors = F)
testdata  <- read.csv('titanic_test.csv', stringsAsFactors = F)
full <- bind_rows(traindata, testdata)
nrow(full)

totalchildren <- traindata %>%
  filter(Age <= 12)
#No of children Survived vs deceased
table(totalchildren$Survived)
adulttable <- traindata %>%
  filter(Age > 12)
#Survival of men and women 
table(adulttable$Sex, adulttable$Survived)
#Survival of passengers of different class
table(adulttable$Pclass, adulttable$Survived)

ggplot(totalchildren, aes(x=Survived)) +
  geom_bar(aes(y = (..count..)/sum(..count..)))+
  scale_y_continuous(name = "Percentage", labels=percent)+
  theme_few()

womentable <- adulttable %>%
  filter(Sex =="female")
mentable <- adulttable %>%
  filter(Sex =="male")
ggplot(womentable, aes(x=Sex, fill=factor(Survived))) +
  geom_bar(aes(y = (..count..)/sum(..count..)), position ='dodge')+
  scale_y_continuous(name = "Percentage", labels=percent)+
  theme_few()

ggplot(mentable, aes(x=Sex, fill=factor(Survived))) +
  geom_bar(aes(y = (..count..)/sum(..count..)), position ='dodge')+
  scale_y_continuous(name = "Percentage", labels=percent)+
  theme_few()

ggplot(adulttable, aes(x=Pclass, fill=factor(Survived))) +
  geom_bar(aes(y = (..count..)/sum(..count..)), position ='dodge')+
  scale_y_continuous(name = "Percentage", labels=percent)+
  theme_few()


ggplot(adulttable, aes(x=Embarked, fill=factor(Survived))) +
  geom_bar(aes(y = (..count..)/sum(..count..)), position ='dodge')+
  scale_y_continuous(name = "Percentage", labels=percent)+
  theme_few()

ggplot(adulttable, aes(Embarked, fill=factor(Sex))) +
  geom_bar(aes(y = (..count..)/sum(..count..)),position ='dodge')+
  scale_y_continuous(name = "Percentage", labels=percent)+
  theme_few()


ggplot(adulttable, aes(Embarked, fill=factor(Pclass))) +
  geom_bar(aes(y = (..count..)/sum(..count..)),position ='dodge')+
  scale_y_continuous(name = "Percentage", labels=percent)+
  theme_few()


hist(full$Age, freq=F, main ="Age Distribution", col = 'lightblue') 


# # Make variables factors into factors
factor_vars <- c('PassengerId','Pclass','Sex','Embarked')

full[factor_vars] <- lapply(full[factor_vars], function(x) as.factor(x))

# # Set a random seed
set.seed(129)

# # Perform mice imputation, excluding certain less-than-useful variables:
mice_mod <- mice(full[, !names(full) %in% c('PassengerId','Name','Ticket','Cabin','Fare','Survived')], method='rf') 

mice_output <- complete(mice_mod)

par(mfrow=c(1,2))
hist(full$Age, freq=F, main='Age: Original Data', 
     col='darkgreen', ylim=c(0,0.04))
hist(mice_output$Age, freq=F, main='Age: MICE Output', 
     col='lightgreen', ylim=c(0,0.04))

full$Age <- mice_output$Age
completedata <- full
completedata$Embarked[c(62, 830)] <- 'C'
completedata$Fare[1044] <- median(completedata[completedata$Pclass == '3' & completedata$Embarked == 'S', ]$Fare, na.rm = TRUE)

#Create new variable with different levels
completedata$ageGroup[completedata$Age<=2] <- "Baby"
completedata$ageGroup[(completedata$Age >2) & (completedata$Age<= 6)] <- "Toddler"
completedata$ageGroup[(completedata$Age >6) & (completedata$Age<= 12)] <- "Child"
completedata$ageGroup[(completedata$Age >12) & (completedata$Age<= 18)] <- "Adolescent"
completedata$ageGroup[(completedata$Age >18) & (completedata$Age<= 60)] <- "Adult"
completedata$ageGroup[(completedata$Age >60)] <- "Elderly"

##Distribution of age of women
ggplot(completedata[completedata$Sex == "female", ], aes(x = Age)) + geom_density(fill = '#99d6ff', alpha=0.4) + 
  geom_vline(aes(xintercept=median(Age)),
             colour='black', linetype='dashed', lwd=1) +
  theme_few()

#Creation of new attribute
completedata$Nwomen <- "Other"
completedata$Nwomen[completedata$Sex == 'female' & completedata$Parch > 0 & completedata$Age > 20 &  completedata$Age < 40] <- "Mother"

#comparing mothers chances across class
table(completedata$Nwomen, completedata$Pclass, completedata$Survived)


#Creation of new attribute
completedata$Nmen <- "Other"
completedata$Nmen[completedata$Sex == 'male' & completedata$Parch > 0 & completedata$SibSp ==1 & completedata$Age > 18 &  completedata$Age < 45] <- "Father"

ggplot(completedata, aes(x=Nmen, fill=factor(Survived)))+
  geom_bar(stat='count', position = 'dodge')+
  theme_few()
ferdinora <- c(34,35)



# Factorizing our predictive attributes
completedata$Pclass <- as.factor(completedata$Pclass)
completedata$Sex <- as.factor(completedata$Sex)
completedata$ageGroup <- as.factor(completedata$ageGroup)
completedata$Nwomen <- as.factor(completedata$Nwomen)
completedata$Nmen <- as.factor(completedata$Nmen)
completedata$Survived <- as.factor(completedata$Survived)
##completedata$Survived <- as.integer(completedata$Survived)

#splitting the dataset
train <- completedata[1:891,]
test <- completedata[892:1309,]

# Set a random seed
set.seed(759)

# Build the model
rf_model <- randomForest(factor(Survived) ~ Pclass + Sex + ageGroup + Nwomen, data = train) 

plot(rf_model)
legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)


# Get importance
importance    <- importance(rf_model)
varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

# Create a rank variable based on importance
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

# Use ggplot2 to visualize the relative importance of variables
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() + 
  theme_few()

# Predict using the test set
prediction <- predict(rf_model, test)

# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
solution <- data.frame(PassengerID = test$PassengerId, Survived = prediction)

# Write the solution to file
write.csv(solution, file = 'resultfinal_AdrianTitanic.csv', row.names = F)