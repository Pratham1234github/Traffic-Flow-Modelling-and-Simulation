#!/usr/bin/env python
# coding: utf-8

# In[1]:


##LETS IMPORT ALL THE LIB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


## Density was calculate in the csv file by creating a new column and using the ( (occupancy * 1000) / 6 )
## lets store the given data in pandas dataframe
data = pd.read_csv("DATA.csv")


# In[3]:


data.head()


# In[4]:


##creating new dataframe containing only density and flow
new = pd.DataFrame(data , columns= ["Density" , "Flow"] ) 


# In[5]:


new.head()


# In[6]:


##convert pandas dataframe to numpy
plot_point = new.to_numpy()


# In[7]:


plot_point.shape


# In[8]:


plot_point[:2]


# In[9]:


#CONVERT THE ARRAY INTO TO x AND y COORDINATES TO PLOT ON GRAPH   
def give_coord (a):
    return a[:,0] , a[:,1]


# In[10]:


##ploting the data
x , y = give_coord(plot_point)

##overall graph 
plt.scatter(x , y , cmap = "winter")
plt.xlim((0,100))
plt.ylim((0,1200))
plt.show()

print("\n")

##first part plot 

plt.scatter(x , y , cmap = "winter")
plt.xlim((0 , 18))
plt.ylim((0,1200))
plt.show()

print("\n")

##second part plot 

plt.scatter(x , y , cmap = "winter")
plt.xlim(( 18 , 100))
plt.ylim((0,1200))
plt.show()

print("\n")


# In[11]:


##storing the density and flow value in 2 diff set beacause of division
point_set1 = []
point_set2 = []
for i in plot_point:
    if i[0] <= 18:
        point_set1.append(i)
    else:
        point_set2.append(i)
##convert them in numpy array
set1 = np.array(point_set1)
set2 = np.array(point_set2)

len(set1), len(set2)


# In[12]:


##importing lib for 2nd degree polynomial curve fitting 
from scipy.optimize import curve_fit


# In[30]:


x1 , y1    = give_coord(set1)
fit_pramas = np.polyfit(x1 , y1 , 2)
print(fit_pramas)
print("\n")
plt.plot(x1 , y1 ,"o" , label = "original" )
fitted_point= np.poly1d(fit_pramas)
plt.plot(x1 , fitted_point(x1) , label ="fit_pramas" );
plt.legend();
def f_set1(x):
    y = fit_pramas[0]*(x**2) + fit_pramas[1]*x + fit_pramas[2]
    return y
y_predicted = f_set1(x1)

print("model eq is ---->")
print('-0.94647126(X**2) + 57.53360268(X) - 10.35872747')


# In[31]:


x2 , y2    = give_coord(set2)
fit_pramas2 = np.polyfit(x2 , y2 , 2)
print(fit_pramas2)
print("\n")
plt.plot(x2 , y2 ,"o" , label = "original" )
fitted_point= np.poly1d(fit_pramas2)
plt.plot(x2 , fitted_point(x2) , label ="fit_pramas" );
plt.legend();
def f_set2(x):
    y = fit_pramas2[0]*(x**2) + fit_pramas2[1]*x + fit_pramas2[2]
    return y
y_predicted = f_set2(x2)
print("model eq is ---->")
print('1.68144119e-03(X**2) -3.16419615e+00(X) + 8.26335412e+02')


# APPLY LAX-FRIEDRICHS FORMUALA TO THE FIT!!!!!!!!!!!
# 
# 
# 

# In[15]:


## creating a zero numpy array beacause it can contain a garbage value in term of getting the actual data
st_curve = np.zeros(shape=(200,1500) , dtype = float , order = "C" )

##just satifying the intial and boundaries condition
##fill in reverse order beacuse of python indexing 
## 0--->100 = 50
##100--->150 = 350
##150--->200 = st line
for i in range(200):
    if i > 100:
        st_curve[i][0] = 50
    
    elif i <= 100 and i >= 50:
        st_curve[i][0] = 350
    
    else:
        left_i = 7*i
        st_curve[i][0] = left_i

        
##for rows its the same

for i in range(1500):
    if i <= 166 :
        st_curve[199][i] = 0
    elif i > 166 and i < 333:
        st_curve[199][i] = 75
    else:
        st_curve [199][i] = 50
    
    


# In[21]:


## APPLY LAX-FRIEDRICHS FORMUALA 

##global variable
val1 = 0
val2 = 0
##
for i in range (1 , 1499):
    ## 199 is already filled
    for j in range(198 , 0 , -1):
        if st_curve[j+1][i-1] > 18:
            val1 = f_set2(st_curve[j+1][i-1])
        elif st_curve[j+1][i-1] <= 18:
            val1 = f_set1(st_curve[j+1][i-1])
        
        if st_curve[j-1][i-1] > 18:
            val2 = f_set2(st_curve[j-1][i-1])
        elif st_curve[j-1][i-1] <= 18:
            val2 = f_set1(st_curve[j-1][i-1])
            
        
        t_1 = ( (st_curve[j+1][i-1] + st_curve[j-1][i-1]) / 2 )
        t_2 = (val1 - val2)
        ## del(t) / 2*del(x) == 0.005 
        st_curve[j][i] = t_1 - 0.005 * t_2


# In[22]:


import csv 

with open ("created_with_lax.csv" , "w" , newline = "") as file:
    writer = csv.writer(file)
    writer.writerows(st_curve)


# In[23]:


import plotly.graph_objs as go


# In[24]:


# Generate x, y, and z coordinates
x = np.arange(st_curve.shape[1])  # space dimension
y = np.arange(st_curve.shape[0])  # time dimension
X, Y = np.meshgrid(x, y)
Z = st_curve

# Create trace
trace = go.Surface( x=X , y=Y , z=Z)

# Create layout
layout = go.Layout( scene = dict ( xaxis_title='Time', yaxis_title='Space', zaxis_title='Density') )

# Create figure
fig = go.Figure( data = [trace] , layout=layout)

# Show figure
fig.show()


# In[25]:


# assume you have the following variables defined:
# st_curve: the space-time density matrix
# time_index: the index of the desired time instance

#store the required time instances in an array
##since python index starts from [0,1,2,3.....] hence use time slape are (166,500,999,1499)
time_inst = [165, 499, 999, 1499]

for i in time_inst:
    # extract the density values at the desired time slapes
    density_curve = st_curve[:, i]

    # create a scatter plot of the density curve
    fig = go.Figure(data = [go.Scatter (x = np.arange(len(density_curve)) , y = density_curve)])

    # customize the plot layout
    fig.update_layout( title=f"Density Variation with Space at Time = {i}",
                       xaxis_title="Space",
                       yaxis_title="Density")

    # show the plot
    fig.show()

    print('\n')


# In[ ]:




