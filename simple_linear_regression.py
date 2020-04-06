''' Part 1 : Data Preprocessing '''


#---------- three common libraries we import before any algorithm-------------------------------------------------
import numpy as np #for math
import matplotlib.pyplot as plt #for plotting
import pandas as pd #to organise my datasets and imort them
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
#------------------------------importing the dataset and defining the independant and dependant variable(s)--------------
dataset = pd.read_csv('Salary_Data.csv') #using panda to import our dataset
x = dataset.iloc[:,:-1] #take all the data except (-1) last column (label).
y = dataset.iloc[:,1]
# for data in dataset['YearsExperience']:
#     if data <=30.00:
#         data +=1.00
#         dataset=dataset.append( pd.DataFrame({'YearsExperience': data, }, index=[0]), ignore_index=True)
df = pd.DataFrame({'YearsExperience': []})
for i in range(11,31):
    df = df.append({'YearsExperience': i}, ignore_index=True)

print(df)

# for i in np.arange(0, 4):
#     if i % 2 == 0:
#         data = dataset.append(pd.DataFrame({'A': i, 'B': i + 1}, index=[0]), ignore_index=True)
#     else:
#         data = dataset.append(pd.DataFrame({'A': i}, index=[0]), ignore_index=True)

# print(dataset.head())
#---------------------------split data set to training and test sets --------------------------------------------------


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)

#no need to make feature scaling in linear regression as the library does that.



''' part 2 : Linear Regression '''


#then we will create and object of LinearRegression() class :

regressor = LinearRegression()

# then we need to fit our object on our trainig set : 
regressor.fit(x_train, y_train) # with this line of code it means it fitted and learned.

#now let's see how it will predicte : 
y_pred = regressor.predict(x_test)
acc=regressor.score(x_test,y_test)
print(acc)
pred = regressor.predict(df)
print(pred)
df['Salary']=list(pred)
df.to_csv('newPotential.csv')

new_potential = pd.read_csv("newPotential.csv",)


#Data visualisation , generally we use fuction plt.scatter
 #first step we will plot the actual data ( real observation ) : 
plt.scatter(x,y,color='red') #real observition 
plt.plot(x_test,y_pred,color='blue')
plt.xlabel('experince')
plt.ylabel('salary')
plt.title('linear regression')
# plt.show()



