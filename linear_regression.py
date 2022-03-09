import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import linear_model 

df = pd.read_csv("linearregression.csv",sep= ";")
print(df)
plt.xlabel("Alan",color="red")
plt.ylabel("Fiyat",color="red")

x=df.iloc[:,:1]
y=df.iloc[:,1:]

plt.scatter(x,y,color="red",marker="+")

reg =linear_model.LinearRegression()
reg.fit(x,y)
tahmin = reg.predict(x)
#coef_ ile bu veri setimizin eğimini buluruz.
#print(reg.coef_)
#intercept_ ile de sabit değeri buluruz yani y = a+bx ise b eğim; a ise sabit değerdir..
#print(reg.intercept_)
plt.plot(x,tahmin,color="blue")