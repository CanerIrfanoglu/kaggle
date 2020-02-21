import pandas as pd
import random
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# See more than 5 cols in pycharm
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 50)




# Load data
results = pd.read_csv("gender_submission.csv")
testdf_raw = pd.read_csv("test.csv")
testdf = testdf_raw.copy()
traindf = pd.read_csv("train.csv")
result_survived = results["Survived"]


### FEATURE ENG ###
# Dropping unnecessary variables ###
traindf.drop(['Ticket', 'Cabin'], axis=1, inplace=True)
testdf.drop(['Ticket', 'Cabin'], axis=1, inplace=True)
combine = [traindf, testdf]

# Creating variable from titles
for dataset in combine:
	dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# before
# pd.crosstab(traindf['Title'], traindf['Sex'])

# Handling the less common titles
for dataset in combine:
	dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
												 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

	dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
	dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
	dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# after
# pd.crosstab(traindf['Title'], traindf['Sex'])

# Convert categorigal to ordinal
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
	dataset['Title'] = dataset['Title'].map(title_mapping)
	dataset['Title'] = dataset['Title'].fillna(0)


traindf = traindf.drop(['Name', 'PassengerId'], axis=1)
testdf = testdf.drop(['Name'], axis=1)
combine = [traindf, testdf]

# NA treatment
traindf.Embarked.ffill(inplace = True)
testdf.Fare.ffill(inplace = True)

random.seed(5)
# Generates n gaussian numbers
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

missingages_train = ngauss(traindf.Age.mean(),traindf.Age.std(),sum(traindf.Age.isnull()))
missingages_test =  ngauss(testdf.Age.mean(),testdf.Age.std(),sum(testdf.Age.isnull()))

# Replacing null values with gaussian numbers generated
traindf.loc[traindf.Age.isnull(),'Age'] = missingages_train
testdf.loc[testdf.Age.isnull(),'Age'] = missingages_test

# Categorical Value Transformation ###
traindf = pd.get_dummies(traindf, columns = ['Sex','Embarked'])
testdf = pd.get_dummies(testdf, columns = ['Sex','Embarked'])

# At this point dataframes are ready for model fitting
X_train = traindf.drop('Survived',1)
y_train = traindf.Survived
X_test  = testdf.drop("PassengerId", axis=1).copy()

###################

### MODEL FITTING ###
# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, y_train) * 100, 2)
acc_log

# See coefficients of log-reg
coeff_df = pd.DataFrame(traindf.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
coeff_df.sort_values(by='Correlation', ascending=False)


# Support Vector Machines
svc = SVC()
svc.fit(X_train, y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, y_train) * 100, 2)
acc_svc

# k-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, y_train) * 100, 2)
acc_knn

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)
acc_gaussian

#Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, y_train) * 100, 2)
acc_perceptron

# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)
acc_linear_svc

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, y_train) * 100, 2)
acc_sgd

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
Y_pred = decision_tree.predict(X_test)
Y_pred_dt = Y_pred.copy()
acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
acc_decision_tree


# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
acc_random_forest

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes', 'Perceptron',
              'Stochastic Gradient Decent', 'Linear SVC',
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log,
              acc_random_forest, acc_gaussian, acc_perceptron,
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


submission = pd.DataFrame({
        "PassengerId": testdf_raw["PassengerId"],
        "Survived": Y_pred_dt
    })

# submission.to_csv('submission.csv', index=False)










