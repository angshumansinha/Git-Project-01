#crop recommendation app
#app.py page
from flask import Flask,render_template,request
#for api
import requests
import config

from flask_cors import cross_origin
import numpy as np
import pickle


#fetch the saVED MODEL
crop_recommendation_model=pickle.load(open("App\model.pkl","rb"))
#creating app object
app=Flask(__name__)

#using weather api
def weather_fetch(city_name): #for fetching weather of city 
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key #acquiring api key
    base_url = "http://api.openweathermap.org/data/2.5/weather?" #fetching api

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None



@app.route('/')#main web page
def home():
    return render_template('home.html')


@app.route('/crop_predict',methods=['GET','POST']) #rendering prediction page
@cross_origin()
def crop_predict():
    if request.method=='POST':
        N=int(request.form['Nitrogen'])
        P=int(request.form['Phosphorous'])
        K=int(request.form['Potassium'])
        ph=float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        city=str(request.form['city'])
        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            my_prediction = crop_recommendation_model.predict([[N, P, K, temperature, humidity, ph, rainfall]])
            final_prediction = my_prediction[0]
            return render_template('predict.html', prediction=final_prediction)
        else:
            return render_template('try.html')


#running the app
if __name__ == '__main__':
    app.run(debug=True,port=5500)