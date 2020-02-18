import os # to change the working directory
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, cross_validation
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn import tree

plt.style.use('ggplot')
titanicdirectory = os.chdir("/Users/Caner/Desktop/python/titanic") 
#os.getcwd() to double check dir. change

####### READING DATA ##########          
traindf = pd.read_csv('train.csv')
testdf = pd.read_csv('test.csv')
results = pd.read_csv('gender_submission.csv')
result_survived = results["Survived"]

######### EXPLORATORY ANALYSIS ###########
traindf.info()

survived_ratio = sum(traindf.Survived == 1) / len(traindf) 
#%38.4

kids = traindf[traindf.Age < 18]
kids_survived = sum(kids.Survived == 1)/len(kids) 
#%54
ages18_36 = traindf[(traindf['Age'] > 18) & (traindf['Age'] <=36)]
ages18_36_survived = sum(ages18_36.Survived == 1)/len(ages18_36)
#%39
ages36_54 = traindf[(traindf['Age'] > 36) & (traindf['Age'] <=54)]
ages36_54_survived = sum(ages36_54.Survived == 1)/len(ages36_54)
#%38.6
agesenior = traindf[(traindf['Age'] > 54)]
agesenior_survived = sum(agesenior.Survived ==1)/len(agesenior)
#%31

survivebyage = [kids_survived,ages18_36_survived,ages36_54_survived,agesenior_survived]
x = np.arange(4)
plt.subplot(2,2,1)
plt.bar(x, height = survivebyage)
plt.xticks(x, ['0-18','18-36','36-54','54+'])
plt.ylabel('Survival Ratio')
plt.xlabel('Age')
plt.title('Survival by Age')
plt.text(x[0]-0.1,survivebyage[0],'{:.1%}'.format(survivebyage[0]))
plt.text(x[1]-0.1,survivebyage[1],'{:.1%}'.format(survivebyage[1]))
plt.text(x[2]-0.1,survivebyage[2],'{:.1%}'.format(survivebyage[2]))
plt.text(x[3]-0.1,survivebyage[3],'{:.1%}'.format(survivebyage[3]))
# There is a negative relationship between age and survival ratio.


female =traindf[traindf.Sex == 'female']
female_survived = sum(female.Survived == 1)/len(female)

male = traindf[traindf.Sex == 'male']
male_survived = sum(male.Survived == 1)/len(male)

survivebysex = [female_survived,male_survived]
x = np.arange(2)
plt.subplot(2,2,2)
plt.bar(x, height = survivebysex, color = 'c')
plt.xticks(x,['female','male'])
plt.ylabel('Survival Ratio')
plt.xlabel('Sex')
plt.title('Survival by Sex')
plt.text(x[0]-0.05,survivebysex[0],'{:.1%}'.format(survivebysex[0]))
plt.text(x[1]-0.05,survivebysex[1],'{:.1%}'.format(survivebysex[1]))



traindf.groupby('Pclass').count() 
#passenger counts by class 216,184,491

first_surv = sum(traindf[traindf.Pclass == 1].Survived == 1)/len(traindf[traindf.Pclass == 1])
#1st class survived ratio
second_surv = sum(traindf[traindf.Pclass == 2].Survived == 1)/len(traindf[traindf.Pclass == 2])
#2nd class survived ratio
third_surv = sum(traindf[traindf.Pclass == 3].Survived == 1)/len(traindf[traindf.Pclass == 3])
#3rd class survived ratio

survivebyclass = [first_surv,second_surv,third_surv]
x= np.arange(3)
plt.subplot(2,2,3)
plt.bar(x, height = survivebyclass,color = 'm')
plt.xticks(x,['1st','2nd','3rd'])
plt.ylabel('Survival Ratio')
plt.xlabel('Class')
plt.title('Survival by Class')
plt.text(x[0]-0.05,survivebyclass[0],'{:.1%}'.format(survivebyclass[0]))
plt.text(x[1]-0.05,survivebyclass[1],'{:.1%}'.format(survivebyclass[1]))
plt.text(x[2]-0.05,survivebyclass[2],'{:.1%}'.format(survivebyclass[2]))

embarks = sum(traindf[traindf.Embarked == "S"].Survived == 1)/len(traindf[traindf.Embarked == "S"])
embarkc = sum(traindf[traindf.Embarked == "C"].Survived == 1)/len(traindf[traindf.Embarked == "C"])
embarkq = sum(traindf[traindf.Embarked == "Q"].Survived == 1)/len(traindf[traindf.Embarked == "Q"])

survivebyembark = [embarks,embarkc,embarkq]
x= np.arange(3)
plt.subplot(2,2,4)
plt.bar(x, height = survivebyembark, color = 'b')
plt.xticks(x, ['S','C','Q'])
plt.ylabel('Survival Ratio')
plt.xlabel('Embark')
plt.title('Survival by Embark(boarding location)')
plt.text(x[0]-0.05,survivebyembark[0],'{:.1%}'.format(survivebyembark[0]))
plt.text(x[1]-0.05,survivebyembark[1],'{:.1%}'.format(survivebyembark[1]))
plt.text(x[2]-0.05,survivebyembark[2],'{:.1%}'.format(survivebyembark[2]))

plt.tight_layout()

################################## PREPROCESSING ##################################
### dropping unnecessary variables ###
traindf.drop(['PassengerId', 'Name','Ticket','Cabin'],axis = 1, inplace = True)
testdf.drop(['PassengerId','Name','Ticket','Cabin'],axis = 1, inplace = True)
### na treatment ###
traindf.Embarked.ffill(inplace = True)
testdf.Fare.ffill(inplace = True)
#traindf.Age.fillna(random.gauss(traindf.Age.mean(),traindf.Age.std()),inplace=True)
#2 values on embarked col forward filled
random.seed(5)
def ngauss(mu,std,n): 
	container = []
	for i in range(n):
		number_generated = round(random.gauss(mu,std))
		if number_generated > 0:
			container.append(number_generated)
		else:
			number_generated = round(random.gauss(mu,std))
			if number_generated > 0:
				container.append(number_generated)
			else:
				ngauss(mu,std,n)
	return(container)
#above func. generates n gaussian numbers
missingages_train = ngauss(traindf.Age.mean(),traindf.Age.std(),sum(traindf.Age.isnull()))
missingages_test =  ngauss(testdf.Age.mean(),testdf.Age.std(),sum(testdf.Age.isnull()))

traindf.loc[traindf.Age.isnull(),'Age'] = missingages_train 
testdf.loc[testdf.Age.isnull(),'Age'] = missingages_test
#Replacing null values with gaussian numbers generated

### Categorical Value Transformation ###
traindf = pd.get_dummies(traindf, columns = ['Sex','Embarked'])
testdf = pd.get_dummies(testdf, columns = ['Sex','Embarked'])

X_train = traindf.drop('Survived',1)
y_train = traindf.Survived 
X_test = testdf
#At this point dataframes are ready for model fitting.



############################################### MODEL FITTING ###############################################
### Logistic Regression ###
logreg = LogisticRegression()
predicted_log = cross_validation.cross_val_predict(logreg, X_train, y_train, cv=10)
print (metrics.accuracy_score(y_train, predicted))
#10 fold cv accuracy score on train data ~80%
fit = logreg.fit(X_train,y_train)
test_pred = fit.predict(X_test)
logreg.score(X_test, result_survived)
#94% accuracy on the test set !

### Support Vector Machine ###
svmfit = svm.SVC()
predicted_svm = cross_validation.cross_val_predict(svmfit, X_train, y_train, cv=10)
print (metrics.accuracy_score(y_train, predicted_svm))
#10 fold cv accuracy score on train data ~72% No need to fit to test data.

### Perceptron ### 
perfit = Perceptron()
predicted_per = cross_validation.cross_val_predict(perfit, X_train, y_train, cv=10)
print (metrics.accuracy_score(y_train, predicted_per))
#10 fold cv accuracy score on train data ~41% No need to fit to test data.

### Decision Tree ###
treefit = tree.DecisionTreeClassifier()
predicted_tree = cross_validation.cross_val_predict(treefit, X_train, y_train, cv=10)
print (metrics.accuracy_score(y_train, predicted_tree))
#10 fold cv accuracy score on train data ~75% Still not better than logistic regression.

forestfit = tree.DecisionTreeClassifier()
predicted_forest = cross_validation.cross_val_predict(forestfit, X_train, y_train, cv=10)
print (metrics.accuracy_score(y_train, predicted_forest))
#10 fold cv accuracy score on train data ~76% Still not better than logistic regression.





