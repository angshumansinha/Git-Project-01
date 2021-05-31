#crop recommendation app
from flask import Flask,render_template,request
import requests
import config# for using weather api
import numpy as np
import pickle

#fetch the saVED MODEL
path=r"C:\Users\angsh\OneDrive\Desktop\Machine Learning Projects\Git Project 01\Models\naivebayes.pkl"
crop_recommendation_model=pickle.load(path,'rb')

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


#creating app object
app=Flask(__name__)

@app.route('/')#main web page
def home():
    title="Crop Recommender System"
    render_template('index.html',title=title)

@app.route('/crop_recommend')#render crop recommend page
def crop_rec():
    title="Crop Recommender System"
    render_template('crop.html',title=title)

@app.route('/crop-predict',methods=['POST']) #rendering prediction page
def crop_predict():
    title='Crop prediction'
    if request.method=='POST':
        N=int(request.form['Nitrogen'])
        P=int(request.form['Phosphorous'])
        K=int(request.form['Potassium'])
        ph=float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        #fetch temperature by city name
        city=request.form.get('city')
        if weather_fetch(city)!=None:
            temp,humidity=weather_fetch(city)
            data=np.array(([[N, P, K, temp, humidity, ph, rainfall]]))
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]
            return render_template('crop-result.html', prediction=final_prediction, title=title)
        else:
             return render_template('try_again.html', title=title)


#running the app
if __name__ == '__main__':
    app.run(debug=False)