from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.ensemble import RandomForestClassifier

training_data = []
labels = []
testing_data = []
list_of_labels = ['dslr canon','mathematics','nike-deodrant','physics','sony cybershot','spoken english','timex watch','titan watch','axe deo','best-seller books','calvin klein','c programming','data structures algorithms','dell laptops','tommy watch','camcorder','camera','chemistry','chromebook','written english']


with open('training.txt') as f:
    for i in range(int(f.readline())):
        data=f.readline().strip()
        y=data.find("\t")
        training_data.append(data[:y])
        labels.append(list_of_labels.index(data[y+1:]))

vectorizer = HashingVectorizer(stop_words='english')
train = vectorizer.fit_transform(training_data)
model = RandomForestClassifier()
model.fit(train,labels)


response = int(input())
for j in range(response):
    t=input().strip()
    testing_data.append(t)
test = vectorizer.transform(testing_data)
test_label=model.predict(test)
for k in test_label:
    print(list_of_labels[k])
