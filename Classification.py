'''
Created on Nov 18, 2022

@author: LAP13307-local
'''

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder



df = pd.read_csv('C:\\resources\\data.csv')
Class = df.keys()[-1]
df.head()
Class = df.keys()[-1]


inputs = df.drop(Class, axis=1)
target = df[Class]

Colunm1_Age = LabelEncoder()
Colunm2_Exp = LabelEncoder()
Colunm3_Rank = LabelEncoder()
Colunm4_Nationality = LabelEncoder()

inputs['Age_n'] = Colunm1_Age.fit_transform(inputs['Age'])
inputs['Exp_n'] = Colunm1_Age.fit_transform(inputs['Experience'])
inputs['Rank_n'] = Colunm1_Age.fit_transform(inputs['Rank'])
inputs['Nation_n'] = Colunm1_Age.fit_transform(inputs['Nationality'])

inputs_n = inputs.drop(['Age','Experience','Rank' ,'Nationality'], axis = 'columns')
print('input la:\n', inputs)
# print('input_n la:\n', inputs_n)
# print('target la: \n',target)

print(df[Class].value_counts())

classifier = DecisionTreeClassifier()
classifier.fit(inputs_n, target)
classifier.score(inputs_n, target)
print('score:',classifier.score(inputs_n, target))

predict_result = classifier.predict([[0,1,0,0]])
if predict_result == 0:
    result = "ít khán giả - thất bại"
else:
    result = "nhiều khán giả - thành công"

print('Dự đoán buổi biểu diễn khi có thông tin của một nghệ sĩ với [Age = Young, Experience = Medium, Rank = Normal, Nationality = Others] thì kết quả sẽ là: ',result)
#


