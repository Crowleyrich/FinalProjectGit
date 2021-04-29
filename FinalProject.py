# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 14:11:14 2021

@author: Richard Crowley


"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pydeck as pdk
import statsmodels.api as sm
from sklearn import linear_model




# Secondary Functions
def read_and_clean_data(fname):
    df = pd.read_csv(fname)
    columns = ['manufacturer','model','year','price','odometer','state','lat','long']
    df = df.filter(items = columns)    
    df = df.dropna()
    return df

              

def read_data_list(fname):
    df = read_and_clean_data('craigslist.csv')
    
    list = []
    
    columns = ['manufacturer','model','year','price','odometer','state','lat','long']
    
    for index, row in df.iterrows():
        sub = []
        for col in columns:
            index_no = df.columns.get_loc(col)
            sub.append(row[index_no])
        list.append(sub)
    return list
 


def manufacturers_list(data):
    manufacturers = []
    
    for i in range(len(data)):
        if data[i][0] not in manufacturers:
            manufacturers.append(data[i][0])
    
    manufacturers.sort()
    return manufacturers


def states_list(data):
    states = []
    
    for i in range(len(data)):
        if data[i][5] not in states:
            states.append(data[i][5])
    states.sort()
    return states


def models_list(data, manufacturer):
    models = []
    
    for i in range(len(data)):
        if data[i][1] not in models and data[i][0] in manufacturer: # need to make sure that the model is by manufacturer
            models.append(data[i][1])
    models.sort()
    return models


def freq_data_price(data, manufacturers_list, price = 100000000):
    freq_dict = {}
    

    for entry in manufacturers_list:
        freq = 0
        for i in range(len(data)):
            if data[i][0] == entry and price >= data[i][3]:
                freq += 1
        freq_dict[entry] = freq  
    return freq_dict


def bar_chart(freq_dict_price):
    x = freq_dict_price.keys()
    y = freq_dict_price.values()
    
    plt.bar(x,y)
    plt.xticks(rotation = 45)
    plt.xlabel('Manufacturers')
    plt.ylabel("Count of Manufacturers")
    title = 'Frequencies of :'
    for key in freq_dict_price.keys():
        title  += ' ' + key
        
    plt.title(title)
    return plt


def filter_DataFrame(df, manufacturer, model, state):
    if len(manufacturer) != 0:
        df = df[df['manufacturer'].isin(manufacturer)]
    if len(model) != 0:
        df = df[df['model'].isin(model)]
    if len(state) != 0:
        df = df[df['state'].isin(state)]
        
    df.rename(columns = {'long': 'lon'}, inplace = True)
    
    return df

    
def statistics(df):
    max_price = df['price'].max()
    average_price = df['price'].mean()
    cheap_price = df['price'].min()
    average_miles = df['odometer'].mean()
    st.write(f"The highest price for a car in this data subset is: ${max_price:,.2f}")
    st.write(f"The average price for a car in this data subset is: ${average_price:,.2f}")
    st.write(f"The cheapest price for a car in this data subset is: ${cheap_price:,.2f}")
    if cheap_price == 0:
        st.write(f"Wow a free car!")
    st.write(f"The average number of miles for a car in this data subset is: {average_miles:,.0f} miles")
    
    
def price_predictor(df, year_choice, odometer_choice): # takes in arguments and calculates a regression
    if df.empty is False:
        X = df[['year', 'odometer']]
        Y = df['price']
        regr = linear_model.LinearRegression()
        regr.fit(X, Y)
        prediction = regr.predict([[year_choice , odometer_choice]])
        st.write(f'Predicted Car Listing Price: ${int(prediction):,.2f}' )

    
    
     
# Main Function
def main():
    st.header("Craigslist Car Listings Final Project")

    # Sidebar Things
    st.sidebar.header("Functions and Filters")
    st.sidebar.subheader("Clean Data Table")
    manufacturer1 = st.sidebar.multiselect('Select manufacturer', manufacturers_list(read_data_list('craigslist.csv')), key = "manufacturer 1")
    model = st.sidebar.multiselect('Select model', models_list(read_data_list('craigslist.csv'), manufacturer1), key = "model 1")
    state = st.sidebar.multiselect('Select state', states_list(read_data_list('craigslist.csv')), key = "state 1")
    st.sidebar.write("-------------------------")
    
    st.sidebar.subheader("Frequency Table for Manufacturers")
    manufacturer2 = st.sidebar.multiselect('Select a manufacturer', manufacturers_list(read_data_list('craigslist.csv')), key = "Manufacturer 2")
    price_limit = st.sidebar.slider('Enter a price limit', 0.00,500000.00, 30000.00)
    st.sidebar.write("-------------------------")
    
    st.sidebar.subheader("Magic Price Estimator")
    manufacturer3 = st.sidebar.multiselect('Select a manufacturer', manufacturers_list(read_data_list('craigslist.csv')),key ="Manufacturer 3")
    model2 = st.sidebar.multiselect('Select model', models_list(read_data_list('craigslist.csv'), manufacturer3), key = "model2")
    state2 = st.sidebar.multiselect('Select state', states_list(read_data_list('craigslist.csv')), key = "state 2")
    st.sidebar.write("-------------------------")
    
    # Data and Figures
    st.subheader("Craigslist Clean Data")
    st.dataframe(filter_DataFrame(read_and_clean_data("craigslist.csv"), manufacturer1, model, state))
    statistics(filter_DataFrame(read_and_clean_data('craigslist.csv'), manufacturer1, model, state))
    st.subheader("Map Displaying Clean Data")
    st.map(filter_DataFrame(read_and_clean_data("craigslist.csv"), manufacturer1, model, state))
    
    st.markdown("***")
    st.subheader('Frequency Table for Manufacturers')
    st.pyplot(bar_chart(freq_data_price(read_data_list('craigslist.csv'), manufacturer2, price_limit)))
    
    st.markdown("***")
    st.subheader('Magic Price Estimator')
    st.write("What should YOU price your car at on Craigslist?")
    year = st.number_input("Enter the model year: ")
    odometer = st.number_input("Enter how many miles your car has: ")
    if st.button("Estimate"):
        price_predictor(filter_DataFrame(read_and_clean_data('craigslist.csv'), manufacturer3, model2, state2), year, odometer)
    

    
if __name__ == '__main__':
    main()






