import pandas as pd

data1 = pd.read_csv("apple.csv")



news = data1.iloc[:,1:2]
news.iloc[:,0:1].replace("[^a-zA-Z]"," ",regex=True, inplace=True)

news["Headlines"] = news["Headlines"].str.lower()

    
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

## implement BAG OF WORDS
countvector=CountVectorizer(ngram_range=(1,1))
traindataset=countvector.fit_transform(news["Headlines"])


# implement RandomForest Classifier
randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(traindataset,data1['Label'])

test = pd.read_csv('test.csv')

test.iloc[:,0:1].replace("[^a-zA-Z]"," ",regex=True, inplace=True)

test["Headlines"] = test["Headlines"].str.lower()
test_dataset = countvector.transform(test["Headlines"])
predictions = randomclassifier.predict(test_dataset)

## Import library to check accuracy
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

matrix=confusion_matrix(test['Label'],predictions)
print(matrix)
score=accuracy_score(test['Label'],predictions)
print(score)
report=classification_report(test['Label'],predictions)
print(report)

'''

## Predict for the Test Dataset
test_transform= []
for row in range(0,len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))


'''