# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 17:40:57 2020

@author: akshay
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

#Source of data: John Hopkins Git URL
url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
df = pd.read_csv(url, error_bad_lines=False)
data = df.to_numpy()

def logistic_function(x,a ,b, c, d ):
    #Basic logistic function
    return a/(1+np.exp(-c*(x-d))) + b 

def plateau(x, y, params, function, diff):
    #Aim identifies the number of days for the curve to flatten (go below some threshold of cases per day)
    #input: x - the array of days
    #       y -  array of cumulative cases
    #       params - best fit parameters for the fit function
    #       diff - threshold on cases per day(0 => the curve has totally flattened)
    casePerDay = y[-1]-y[-2]
    now = x[-1]
    days = 0
    while casePerDay > diff:
        days += 1
        confirmed_then = function(now + days-1, *params) 
        confirmed_now = function(now + days, *params) 
        casePerDay = confirmed_now - confirmed_then
    return days, confirmed_now

def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

def half_max_x(x, y):
    half = min(y)+((max(y)-min(y))/2.0)
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]

def richards_curve(x,a,b,c,k,m,q):
    return (k-a)/((c+q*np.exp(-b*x))**(1/m)) + a 


def predict_corona_curve(data, country, function, diff,hm):
    #Aim: predicts the corona curve and it's features
    #Input: country - name of the country(string, 'India'). If 'None', data of Kerala will be processed
    #       function - function used for modeling the data(eg: logistic_function)
    #       diff - threshold on cases per day, to estimate the flatterning time scale
    #       hm - whether FWHM to be calculated (boolean)
    #Output: Predicted curve, peak, time to flatten, half max, etc
    if country != 'None' :
        countries = data[:,1]
        idxCountry = np.int32(np.where(countries == country))
        print(country)
        casesCountry = np.asarray(data[idxCountry,4:-1])
        c1 = np.transpose(np.concatenate(casesCountry).astype(None))
        #d1 = np.sum(c1, axis = 1)
        cumulativeCases = np.sum(c1, axis = 1)#
    else:
        country = 'Kerala' 
        cumulativeCases = np.array([20, 22, 24, 27, 27, 27, 28, 40, 52, 67, 95, 109, 118, 137, 176, 182, 202, 234, 241, 265, 286, 295, 306, 314, 327, 335, 344, 356, 363, 373 , 375, 378, 386, 387, 394, 395, 399, 401, 407, 426, 437, 447, 450, 457, 468, 481, 485, 495, 497, 497, 499, 499, 499, 502, 502, 502, 503, 505, 512, 519, 524, 534, 560, 576, 587, 601, 630, 642, 666, 690, 732, 794, 847, 896, 963, 1003, 1088, 1150, 1208, 1269, 1326, 1412, 1494, 1588, 1699, 1807, 1914, 2005, 2096, 2161, 2244, 2322, 2407, 2461, 2543, 2622, 2794, 2912, 3039, 3172, 3310, 3451, 3603, 3726, 3877, 4071, 4189, 4311, 4442, 4593, 4753, 4964, 5204, 5429, 5622, 5894, 6195, 6534, 6950, 7438, 7873, 8322, 8930, 9553, 10275, 11066, 11659, 12480, 13274, 13994, 15032, 16110, 16995, 18098, 19025, 19727, 20896, 21797, 22303, 23612, 24742, 25911, 26873, 27956, 29151, 30449, 31699, 33120, 34331, 35514, 36932, 38144, 39708, 41277, 42885, 44415, 46140, 47898, 50231, 52199, 54182, 56354, 58262, 59504, 61879, 64355, 66761, 69304, 71701, 73855, 75385, 76525, 78072, 79625, 82104, 84759, 87841, 89489, 92515, 95917, 99266, 102254, 105139, 108278, 110818, 114033, 117863, 122394, 126561, 131205, 135901, 138811, 142936, 148132, 154636, 161113, 167939, 175384, 179922, 187276, 196106, 204241, 213499, 221333, 229886, 234928, 242799, 253405, 258850, 268100, 279855, 289202, 295132, 303896, 310140, 317929, 325212, 334228, 341859, 346881, 353472, 361841, 369323, 377834, 386087, 392930, 397217, 402674, 411464, 418484, 425122, 433105, 440130, 444268, 451130, 459646, 466466, 473468, 480669, 486109, 489702, 495712, 502719, 508256, 514060, 520417, 524998, 527708, 533500, 539919, 545641, 551669, 557441, 562695, 566452, 571601, 578092, 583470, 587436, 593686, 599600, 602982, 608357])
 
    lastDays = optimum_day_range(cumulativeCases)
    y = np.array(cumulativeCases[len(cumulativeCases)-lastDays:len(cumulativeCases)])
    x = np.arange(len(y))
    if function == logistic_function:
        initialGuess = [np.amax(y), 20, 1, 50]
        
    if function == richards_curve:
        initialGuess = [np.amin(y), 20, 5, np.amax(y),1,10]
        
    params,_ = curve_fit(function,x,y, initialGuess, maxfev = 10000)
     
    #Goodness of fit
    residuals = y- function(x, *params)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    dailyCases = np.diff(y)
    peakDay = np.int32(np.where(dailyCases == np.amax(dailyCases)) - np.int32(lastDays))
    #peakDay =  np.array(peakDay.astype('None'))
    print(f"Peak at {peakDay} days, peak cases: {np.int32(np.amax(dailyCases))}")
    future_dailyCases = np.diff(function(np.arange(int(2*lastDays + 1)),*params))
    if hm:
        hmx = half_max_x(np.arange(int(2*lastDays + 1))-lastDays, future_dailyCases)
        fwhm = hmx[1] - hmx[0]
        print(f"Peak spread over {int(fwhm)} days")
        half = max(future_dailyCases)/2.0
    #Plotting
    fig, ax1 = plt.subplots()
    
    
    ax1.plot(x-lastDays,y, '.', color = 'c', label = "Cumulative Cases")
    ax1.plot(x-lastDays, function(x,*params),color = 'k',label = "Fit, $R^{2} = $"  + f" {np.floor(r_squared*100)/100}")
    ax1.plot(np.linspace(0,lastDays,lastDays+1), logistic_function(np.linspace(lastDays,2*lastDays,lastDays+1),*params), color = 'g', label = "Predicted CDF")
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Cumulative Cases")
    ax1.legend(loc = 'upper left')
    
    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(np.diff(y)))-lastDays,np.diff(y),'.',color = 'r', label = 'Daily Cases')
    if hm == 0:
        ax2.plot(np.arange(int(2*lastDays))-lastDays, future_dailyCases, color = 'b', label = "PDF")
    if hm:
        ax2.plot(np.arange(int(2*lastDays))-lastDays, future_dailyCases, color = 'b', label = "PDF, FWHM:{:.2f}".format(fwhm))
        ax2.plot(hmx, [half, half], color = 'y')
        ax2.plot([hmx[1], hmx[1]],[0 , np.amax(future_dailyCases)],'k--', label = f"Days to Half Max: {int(np.ceil(hmx[1]))}")
        print(f"Days to Half Max: {int(np.ceil(hmx[1]))}")
    ax2.set_ylabel("Daily Cases")
    ax2.legend(loc = 2)
    
    plt.legend()
    plt.title( country + " : Corona Curve Prediction as of " +  str(np.datetime64('today')))
    plt.show()
    

    days, confirmed = plateau(
        x, y, params, logistic_function, diff=diff
    )
    print(f"{days} days until cases per day is less than {diff}")
    print(f"Number of cases will be {int(confirmed)}")


def optimum_day_range(cumulativeCases):
    #Aim: identifies the onset of most recent wave of corona outbreak
    #Input: cumulativeCases (numpy array with cumulative corona cases)
    #Output: The day when the curve of corona cases per day started going up
    intervalDays = np.linspace(50, int(len(cumulativeCases)), dtype = 'int')
    slopes = np.zeros((len(intervalDays),3))
    for i in range(len(intervalDays)):
        y = np.diff(np.array(cumulativeCases[len(cumulativeCases)-intervalDays[i]:len(cumulativeCases)]))
        x = np.arange(len(y))
        z = np.polyfit(x, y, 1)
        slopes[i,0] =  z[0]
        slopes[i,1] =  intervalDays[i]
    slopes = slopes[slopes[:,0]>0]#only growth
    if  slopes.size == 0:
        print('Curve has flattened')
        return 100
    else: 
        slopes[:,2] = slopes[:,1]/slopes[:,0]
        opt_DayRange =  int(slopes[slopes[:,2] == np.amin(slopes[:,2]),1])#slowest growth period
        print(f"Curve started going up {opt_DayRange} days ago")
        return opt_DayRange

#Example: Predicting the corona curve for Germany    
predict_corona_curve(data, 'Germany', logistic_function, diff = 100, hm =  1)
        
    
        






