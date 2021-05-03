#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 18:58:53 2021

@author: Rohit K S S Vuppala
         Graduate Student, 
         Mechanical and Aerospace Engineering,
         Oklahoma State University.

@email: rvuppal@okstate.edu

modified from MetPy examples Advanced Sounding
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import metpy.calc as mpcalc
from metpy.cbook import get_test_data
from metpy.plots import add_metpy_logo, SkewT
from metpy.units import units

from datetime import datetime
from siphon.simplewebservice.wyoming import WyomingUpperAir


#%%
"""
Get data from server using Siphon Package
"""
#Specifier
#sname = "2004-05-30-UTC12Z"
#date = datetime(2004, 5, 30, 0)
#station = '72357'   #Norman Station
#df = WyomingUpperAir.request_data(date, station)

#sname = "2021-04-24-UTC12Z"
#date = datetime(2021, 4, 24, 0)
#station = '72355'   #Fort SIll
#df = WyomingUpperAir.request_data(date, station)

"""
Get data from local file
"""
sname = "2004-05-30-UTC0Z"
col_names = ['pressure', 'height', 'temperature','dewpoint', 'relh', 'qv', 'direction', 'speed','theta']
df = pd.read_fwf('../2004-05-29/'+sname+'.txt',skiprows=5, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8], names=col_names)


"""
Example Data set
"""
#col_names = ['pressure', 'height', 'temperature', 'dewpoint', 'direction', 'speed']

#df = pd.read_fwf(get_test_data('may4_sounding.txt', as_file_obj=False),
#                 skiprows=5, usecols=[0, 1, 2, 3, 6, 7], names=col_names)


# Drop any rows with all NaN values for T, Td, winds
df = df.dropna(subset=('temperature', 'relh', 'qv','dewpoint', 'direction', 'speed','theta'), how='all'
               ).reset_index(drop=True)

###########################################
#Data and units

h = df['height'].values * units.m
p = df['pressure'].values * units.hPa
T = df['temperature'].values * units.degC
Td = df['dewpoint'].values * units.degC
wind_speed = df['speed'].values * units.knots
wind_dir = df['direction'].values * units.degrees
u, v = mpcalc.wind_components(wind_speed, wind_dir)

###########################################
# Create a new figure. The dimensions here give a good aspect ratio.

fig = plt.figure(figsize=(9, 9))
skew = SkewT(fig, rotation=45)

# Plot the data using normal plotting functions, in this case using
# log scaling in Y, as dictated by the typical meteorological plot.
skew.plot(p, T, 'r',label='T')
skew.plot(p, Td, 'g',label='T_DwP')
skew.plot_barbs(p, u, v)
skew.ax.set_ylim(1000, 100)
skew.ax.set_xlim(-40, 60)

#Calculate relevant
lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])
skew.plot(lcl_pressure, lcl_temperature, 'ko', markersize=10,markerfacecolor='black',label='LCL')
lfc_pressure, lfc_temperature = mpcalc.lfc(p, T, Td)
skew.plot(lfc_pressure, lfc_temperature, 'bo', markersize=10,markerfacecolor='blue',label='LFC')
el_pressure, el_temperature = mpcalc.el(p, T, Td)
skew.plot(el_pressure, el_temperature, 'ro', markersize=10,markerfacecolor='red',label='EL')

# Calculate full parcel profile and add to plot as black line
prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
skew.plot(p, prof, 'k', linewidth=2,label='Parcel Profile')

# Shade areas of CAPE and CIN
skew.shade_cin(p, T, prof, Td,label='CIN')
skew.shade_cape(p, T, prof,label='CAPE')

CAPE,CIN = mpcalc.cape_cin(p,T,Td,prof)

# Add the relevant special lines
skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines()

# Show the plot
plt.legend(loc='upper right', fontsize=14)
plt.title("Norman"+str(sname))
plt.show()


#%%
"""
Units required     : u,v = m/s; h = m; p = mb; theta = K; qv = g/kg; direction = deg
Units in sounding  : u,v = knots; h = m; p = mb; theta = K; qv = g/kg
"""
col_names = ['pressure', 'height', 'temperature','dewpoint', 'relh', 'qv', 'direction', 'speed','theta']
df = pd.read_fwf('../2004-05-29/'+sname+'.txt',skiprows=5, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8], names=col_names)

h    = df['height'].values
p    = df['pressure'].values
qv   = df['qv'].values
theta= df['theta'].values 

#Convet the windspeed to velocities
wind_speed = df['speed'].values * units.knot
wind_dir = df['direction'].values * units.degrees
u, v = mpcalc.wind_components(wind_speed, wind_dir)

u = u.to(units('m/s'))
v = v.to(units('m/s'))

#Strip the units
u = u.magnitude
v = v.magnitude

sfc_p     = p[0]
sfc_theta = theta[0]
sfc_qv    = qv[0]

#Replace NaN in u,v with 0.0
u = np.nan_to_num(u)
v = np.nan_to_num(v)


fname = "input_sounding"+sname
ln = p.size

with open(fname,'w') as f:
    f.write(str(sfc_p)+'\t'+str(sfc_theta)+'\t'+str(sfc_qv)+'\n')
    for i in range(ln):
        f.write(str(round(h[i],3))+'\t'+str(round(theta[i],3))+'\t'+str(round(qv[i],3))+'\t'+str(round(u[i],3))+'\t'+str(round(v[i],3))+'\n')

f.close()


#%%
"""
Calculate the storm tracking velocities
75% of average speed till 6km
"""

        
def find_nearest(h, h_m):
    h_lim = 6000.0
    
    diff_g = 99999.0
    for i in range(h.shape[0]):
        diff = h[i] - h_lim
        if(abs(diff) < abs(diff_g)):
            diff_g = diff 
            indx   = i
            val    = h[i]     
    #IF less than h_m take next indx
    if(h[indx] < h_m):
        indx = indx + 1
        val  = h[indx]
        
    return val,indx

#%%
def getpoly_val(p,x,deg):
    
    y = 0.0
    for i in range(deg+1):
        y = y + p[i]*(x**(deg-i))
        
    return y
    
#%%    
    
h_near,h_indx = find_nearest(h,6000.0)

#Find the mean value and return the STRM Speed
u_m = u[:h_indx+1]
v_m = v[:h_indx+1]
h_m = h[:h_indx+1]

#Do a polynomial fit first to find mean
#Deg of poly
deg = 10

p_u = np.polyfit(h_m,u_m,deg)
p_v = np.polyfit(h_m,v_m,deg)


#Specify large nh for good mean values
nh = 1000
h_sp = np.linspace(h_m[0],h_m[-1],nh)

#Get the values from fit for the plot
u_sp = np.zeros(nh)
v_sp = np.zeros(nh)
for i in range(nh):
    u_sp[i] = getpoly_val(p_u,h_sp[i],deg)
    v_sp[i] = getpoly_val(p_v,h_sp[i],deg)


#Plot and check
plt.figure()
plt.title("U-Vel Fit vs Actual")
plt.plot(u_sp,h_sp,linestyle='--',color='green',label='Fit')
plt.plot(u_m,h_m,linestyle='-',color='black',label='Actual Values')
plt.grid()
plt.savefig("Ufit.jpg",dpi=300)
plt.legend()

plt.figure()
plt.title("V-Vel Fit vs Actual")
plt.plot(v_sp,h_sp,linestyle='--',color='green',label='Fit')
plt.plot(v_m,h_m,linestyle='-',color='black',label='Actual Values')
plt.grid()
plt.savefig("Vfit.jpg",dpi=300)
plt.legend()

u_s= np.sum(u_sp)/(u_sp.size) * 0.75
v_s= np.sum(v_sp)/(v_sp.size) * 0.75

print("STRM Tracking Approx u,v:",round(u_s,3),round(v_s,3))















