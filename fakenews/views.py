#Libraries
from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import emoji
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.linear_model import PassiveAggressiveClassifier
import os

import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import pyttsx3
import pandas as pd
from sklearn import \
	preprocessing
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import PySimpleGUI as sg
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import pymysql




def speak(
		audio):
	engine = pyttsx3.init('sapi5')
	voices = engine.getProperty('voices')
	rate = engine.getProperty('rate')
	engine.setProperty('rate', rate - 20)
	engine.setProperty('voice', voices[0].id)		 
	engine.say(audio)
	engine.runAndWait()
	#del engine

def reg(request):
	if request.method=='POST':
		if request.POST.get('username') and request.POST.get('password') and request.POST.get('email') and request.POST.get('phone') and request.POST.get('address'):
			db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'crop',charset='utf8')
			db_cursor = db_connection.cursor()
			student_sql_query = "INSERT INTO user(username,password,email,phone,address) VALUES('"+request.POST.get('username')+"','"+request.POST.get('password')+"','"+request.POST.get('email')+"','"+request.POST.get('phone')+"','"+request.POST.get('address')+"')"
			db_cursor.execute(student_sql_query)
			db_connection.commit()	  
			print('helllllllllllllllllllllllllllllllllllllllllllllllllloo')
			return render(request,'loginpage.html') 
		return render(request,'loginpage.html') 
def loginuser(request):
	if request.method=='POST':
		print('hiiiiiiiiiiiiiiiiii')
		username = request.POST.get('username', False)
		password = request.POST.get('password', False)		
		print('hiiiiiiiiiiiiiiiiii1111111111111')
		con = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'crop',charset='utf8')
		utype = 'none'
		with con:
			print('hiiiiiiiiiiiiiiiiii33333333333333333333')
			cur = con.cursor()
			cur.execute("select * FROM user")
			rows = cur.fetchall()
			for row in rows:
				if row[1] == request.POST.get('username') and row[1] == request.POST.get('password'):
					utype = 'success'
					#status_data = row[5] 
					break
		if utype == 'success':
			print('hiiiiiiiiiiiiiiiiii11111111111122222222222222222222')
			return render(request, 'loginpage.html')
		if utype == 'none':
			return render(request, 'index.html')	
	return render(request,'index.html')
################ Home #################
def home(request):
	return render(request,'cropyield.html')

######## SVM ######
def nvb(request):
	data = pd.read_csv('Ez:/soil_kroptor/crop.csv')
	from sklearn import preprocessing		
	labelencoder_X = preprocessing.LabelEncoder()
	X = data.iloc[:, 1:8].values
	y = data.iloc[:, 9].values	   
	X.shape
	y.shape

	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2)

	A_test=[[1997,598400,24.243,42.3484,84,217000,1]]
	#testing  
	from sklearn import linear_model
	reg = linear_model.LinearRegression()
	reg.fit(X_train,y_train)
	pred = reg.predict(X_test)
	pred1 = reg.predict(A_test)
	print(pred1)
	score = reg.score(X_train,y_train)
	print("R-squared:", score)
	d = {'a': score}
	#print(reg.score(X_test,y_test))
	#acclogistic=reg.score(X_test,y_test)
	return render(request,'NaiveBayes.html',d)
def rf(request):
	data = pd.read_csv('D:/soil_kroptor/crop.csv')
	X = data.iloc[:, 1:8].values
	y = data.iloc[:, 9].values	   
	X.shape
	y.shape

	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2)

	A_test=[[1997,598400,24.243,42.3484,84,217000,1]]
	#testing  
	from sklearn.datasets import make_regression
	from sklearn.ensemble import RandomForestClassifier
	regr = RandomForestClassifier()
	regr.fit(X_train,y_train)
	pred = regr.predict(X_test)
	pred1 = regr.predict(A_test)
	print(pred1)
	score = metrics.accuracy_score(y_test, pred)
	print("accuracy:   %0.3f" % score)
	d = {'a': score}
	#print(reg.score(X_test,y_test))
	#acclogistic=reg.score(X_test,y_test)
	return render(request,'NaiveBayes.html',d)
	
def svr(request):
	data = pd.read_csv('D:/soil_kroptor/crop.csv')
	X = data.iloc[:, 1:8].values
	y = data.iloc[:, 9].values	   


	#testing  
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2)

	A_test=[[1997,598400,24.243,42.3484,84,217000,1]]
	
	from sklearn.svm import SVR
	regressor = SVR(kernel = 'rbf')
	
	regressor.fit(X_train,y_train)
	pred = regressor.predict(X_test)
	pred1 = regressor.predict(A_test)
	print(pred1)
	score = regressor.score(X_train,y_train)
	print("R-squared:", score)
	'''score = metrics.accuracy_score(y_test, pred)
	print("accuracy:   %0.3f" % score)'''
	d = {'a': score}
	#print(reg.score(X_test,y_test))
	#acclogistic=reg.score(X_test,y_test)
	return render(request,'NaiveBayes.html',d)
	
def pac(request):	
	return render(request,'NaiveBayes.html')
def svm(request):	
	return render(request,'NaiveBayes.html')
					
def accuracy(request):
	return render(request,'index.html')
  
def test(request):
	if request.method=='POST':
		headline1= request.POST.get('headline1')
		headline2= request.POST.get('headline2')
		headline3= request.POST.get('headline3')
		headline4= request.POST.get('headline4')
		headline5= request.POST.get('headline5')
		headline6= request.POST.get('headline6')
		
		from sklearn import preprocessing		
		labelencoder_X = preprocessing.LabelEncoder()
		headline6 = labelencoder_X.fit_transform([[headline6]])
		
		headline7= request.POST.get('headline7')
			
		print(headline1)
			
			
		headline1= int(headline1)
		headline2 = int(headline2)
		headline3 = float(headline3)
		headline4 = float(headline4)
		headline5 = int(headline5)
		headline6 = int(headline6)
		headline7 = int(headline7)
			
		data = pd.read_csv('D:/soil_kroptor/crop.csv')

		X = data.iloc[:, 1:8].values
		y = data.iloc[:, 9].values	   
		X.shape
		y.shape

		X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2)
		A_test=[[headline1,headline2,headline3,headline4,headline5,headline6,headline7]]
		#testing  
		from sklearn.datasets import make_regression
		from sklearn.ensemble import RandomForestClassifier
		reg = RandomForestClassifier()
		reg.fit(X_train,y_train)
		pred = reg.predict(X_test)
		pred1 = reg.predict(A_test)
		print(pred1)
		print('------------------------------------------------')
		print(pred)
						
		fakefalse=''
		if pred1==0:
			fakefalse='less crop yield'
		else:
			fakefalse='high crop yield'
				
		score = metrics.accuracy_score(y_test, pred)
		print("accuracy:   %0.3f" % score)
		d = {'a':pred1,'crop':request.POST.get('headline6')}		   
		print('hellllllllllllllllllllllllllllllllo')
		return render(request,'NaiveBayes.html',d);

def fertilizerform(request):
	return render(request,'fertilizerform.html')
def fertilizerRf(request):
	if request.method=='POST':
		headline1= request.POST.get('headline1')
		headline2= request.POST.get('headline2')
		headline3= request.POST.get('headline3')
		headline4= request.POST.get('headline4')
		headline5= request.POST.get('headline5')
		headline6= request.POST.get('headline6')
		headline7= request.POST.get('headline7')
		
		from sklearn import preprocessing		
		labelencoder_X = preprocessing.LabelEncoder()
			
		print(headline1)
			
			
		nitrogen_content= float(headline1)
		phosphorus_content = float(headline2)
		potassium_content = float(headline3)
		temperature_content = float(headline4)
		humidity_content = float(headline5)
		ph_content = float(headline6)
		rainfall = float(headline7)
			
		data = pd.read_excel('D:/soil_kroptor/cropnew.xlsx')

		X = data.iloc[:, 0:7].values
		y = data.iloc[:, 8].values	   
		X.shape
		y.shape

		X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2)
		A_test=[[nitrogen_content,phosphorus_content,potassium_content,temperature_content,humidity_content,ph_content,rainfall]]
		#testing  
		from sklearn.datasets import make_regression
		from sklearn.ensemble import RandomForestClassifier
		reg = RandomForestClassifier()
		reg.fit(X_train,y_train)
		pred = reg.predict(X_test)
		predict1 = reg.predict(A_test)
		print(predict1)
		crop_name =''
		cp = ''
		phlevel=''
		if predict1 == 0:
			crop_name = 'Apple(सेब)'
			cp="No alternatives"
		elif predict1 == 1:
			crop_name = 'Banana(केला)'
			cp='Coconut'
		elif predict1 == 2:
			crop_name = 'Blackgram(काला चना)'
			cp='No alternative'
		elif predict1 == 3:
			crop_name = 'Chickpea(काबुली चना)'
			cp="GreenPeas,Soyabeans"
		elif predict1 == 4:
			crop_name = 'Coconut(नारियल)'
			cp = 'Banana,Pepper'
		elif predict1 == 5:
			crop_name = 'Coffee(कॉफ़ी)'
			cp='Black Pepper,Cardamom'
		elif predict1 == 6:
			crop_name = 'Cotton(कपास)'
			cp='Mung seeds,Peas'
		elif predict1 == 7:
			crop_name = 'Grapes(अंगूर)'
			cp='Clover Plants'
		elif predict1 == 8:
			crop_name = 'Jute(जूट)'
			cp='No alternatives'
		elif predict1 == 9:
			crop_name = 'Kidneybeans(राज़में)'
			cp='No alternatives'
		elif predict1 == 10:
			crop_name = 'Lentil(मसूर की दाल)'
			cp='jowar'
		elif predict1 == 11:
			crop_name = 'Maize(मक्का)'
			cp='Bajra'
		elif predict1 == 12:
			crop_name = 'Mango(आम)'
			cp='Lemon,Guava'
		elif predict1 == 13:
			crop_name = 'Mothbeans(मोठबीन)'
			cp='Cotton'
		elif predict1 == 14:
			crop_name = 'Mungbeans(मूंग)'
		elif predict1 == 15:
			crop_name = 'Muskmelon(खरबूजा)'
			cp='Cucumber,Watermelon'
		elif predict1 == 16:
			crop_name = 'Orange(संतरा)'
			cp='Soyabeans,Peas'
		elif predict1 == 17:
			crop_name = 'Papaya(पपीता)'
			cp='Potatoes,Onions,Carrots'
		elif predict1 == 18:
			crop_name = 'Pigeonpeas(कबूतर के मटर)'
			cp='Green Lentils'
		elif predict1 == 19:
			crop_name = 'Pomegranate(अनार)'
			cp='No alternative'
		elif predict1 == 20:
			crop_name = 'Rice(चावल)'
			cp="raagi,sajji"
		elif predict1 == 21:
			crop_name = 'Watermelon(तरबूज)'
			cp='Muskmelon,Tomatoes,Chillies'
		if int(humidity_content) >= 1 and int(
				humidity_content) <= 33:
			humidity_level = 'low humid'
		elif int(humidity_content) >= 34 and int(humidity_content) <= 66:
			humidity_level = 'medium humid'
		else:
			humidity_level = 'high humid'

		if int(temperature_content) >= 0 and int(
				temperature_content) <= 6:
			temperature_level = 'cool'
		elif int(temperature_content) >= 7 and int(temperature_content) <= 25:
			temperature_level = 'warm'
		else:
			temperature_level = 'hot'

		if int(rainfall) >= 1 and int(
				rainfall) <= 100:
			rainfall_level = 'less'
		elif int(rainfall) >= 101 and int(rainfall) <= 200:
			rainfall_level = 'moderate'
		elif int(rainfall) >= 201:
			rainfall_level = 'heavy rain'

		if int(nitrogen_content) >= 1 and int(
				nitrogen_content) <= 50:
			nitrogen_level = 'less'
		elif int(nitrogen_content) >= 51 and int(nitrogen_content) <= 100:
			nitrogen_level = 'not to less but also not to high'
		elif int(nitrogen_content) >= 101:
			nitrogen_level = 'high'

		if int(phosphorus_content) >= 1 and int(
				phosphorus_content) <= 50:
			phosphorus_level = 'less'
		elif int(phosphorus_content) >= 51 and int(phosphorus_content) <= 100:
			phosphorus_level = 'not to less but also not to high'
		elif int(phosphorus_content) >= 101:
			phosphorus_level = 'high'

		if int(potassium_content) >= 1 and int(
				potassium_content) <= 50:
			potassium_level = 'less'
		elif int(potassium_content) >= 51 and int(potassium_content) <= 100:
			potassium_level = 'not to less but also not to high'
		elif int(potassium_content) >= 101:
			potassium_level = 'high'
		
		if float(ph_content) >= 0 and float(ph_content) <= 5:
			phlevel = 'acidic'
		elif float(ph_content) >= 6 and float(ph_content) <= 8:
			phlevel = 'neutral'
		elif float(ph_content) >= 9 and float(ph_content) <= 14:
			phlevel = 'alkaline'

		print(crop_name)
		print(humidity_level)
		print(temperature_level)
		print(rainfall_level)
		print(nitrogen_level)
		print(phosphorus_level)
		print(potassium_level)
		print(phlevel)


		speak(
			"Sir according to the data that you provided to me. The ratio of nitrogen in the soil is  " + nitrogen_level + ". The ratio of phosphorus in the soil is  " + phosphorus_level + ". The ratio of potassium in the soil is  " + potassium_level + ". The temperature level around the field is  " + temperature_level + ". The humidity level around the field is  " + humidity_level + ". The ph type of the soil is  " + phlevel + ". The amount of rainfall is  " + rainfall_level)  # Making our program to speak about the data that it has received about the crop in front of the user.

		speak("The best crop that you can grow is  " + crop_name)
		speak("The other alternatives are " + cp)
		
							
		
		
				
		score = metrics.accuracy_score(y_test, pred)
		print("accuracy:   %0.3f" % score)
		d = {'a':crop_name,'b':cp,'score':score}		   
		print('hellllllllllllllllllllllllllllllllo')
	return render(request,'fres.html',d)
def simple_upload(request):
	return render(request,'indexfile1.html')
def simple(request):
	return render(request,'indexfile1.html')	
def fileshow(request):		
	return render(request,'indexfile1.html')
def fileshow1(request):
	return render(request,'indexfile2.html')
def loginpage(request):
	return render(request,'index.html')
def register(request):
	return render(request,'register.html')

def input(request):
	return render(request,'input.html')	 
