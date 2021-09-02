#!/bin/python3
# Author: Adam Lozinsky
# Date: June 18, 2019
# Description: Generates optimal antenna array designs based on 
#	       perturbed reuleaux triangles (Keto, 1997).


import numpy as np
import matplotlib.pyplot as mpl
import matplotlib.animation as animation
import scipy.constants as sci
import scipy.signal as sig
import random
import csv
from mpl_toolkits.mplot3d import Axes3D
from sklearn.utils import shuffle
import cv2

# Constants
PI = np.pi
SPEED_OF_LIGHT = sci.c
FREQUENCY = 49500000
WAVELENGTH = SPEED_OF_LIGHT / FREQUENCY
FLAG = False

# Setup 
NUM_ANTENNAS = int(10)
UNIQUE_BASELINES_REQ = int(40)
UNIFROM_RADIUS = 0.89 #0.83 - 0.905
LIMIT = 10000
DATA_DUMP = True
FPS = 0.01


def reuleaux_boundary(n=3, A=48, resolution=0.001):
	b = np.arange(0, 2 * PI, resolution)	# Simplification
	a = np.floor(n*b/(2*PI))				# Simplification
	A = A * 0.5773502811646716				# Width
	r = -PI/2								# Rotation
	x = A * 2 * np.cos(PI/(2*n)) * np.cos(0.5*(b+PI/n*(2*a+1))+r) - A * np.cos(PI/n*(2*a+1)+r)
	y = A * 2 * np.cos(PI/(2*n)) * np.sin(0.5*(b+PI/n*(2*a+1))+r) - A * np.sin(PI/n*(2*a+1)+r)
	return x,y


def initial_positions(x_bound, y_bound):
	initial_posx = x_bound[np.arange(NUM_ANTENNAS) * int(len(x_bound) / NUM_ANTENNAS)]
	initial_posy = y_bound[np.arange(NUM_ANTENNAS) * int(len(y_bound) / NUM_ANTENNAS)]
	weight_x = np.zeros(NUM_ANTENNAS)
	weight_y = np.zeros(NUM_ANTENNAS)
	return initial_posx, initial_posy, weight_x, weight_y


def choose_position(weight_x, weight_y, mod_x, mod_y, x_bound, y_bound):
	ant_posx = np.zeros(NUM_ANTENNAS)
	ant_posy = np.zeros(NUM_ANTENNAS)
	antenna_list = np.arange(NUM_ANTENNAS)
	np.random.shuffle(antenna_list)
	for i in antenna_list:
		weight_x[i] = weight(weight_x[i], mod_x[i], 0)
		weight_y[i] = weight(weight_y[i], mod_y[i], 1)
	ant_posx = weight_x
	ant_posy = weight_y
	return ant_posx, ant_posy, weight_x, weight_y


def weight(w, m, ax):
	B = np.array([-24.0,24.0,-28.0,20.0])
	if ax == 0:
		m_neg = max([B[0]-w, -m])
		m_pos = min([B[1]-w, m])
		w2 = w + np.random.uniform(m_neg, m_pos)
	elif ax == 1:
		m_neg = max([B[2]-w, -m])
		m_pos = min([B[3]-w, m])
		w2 = w + np.random.uniform(m_neg, m_pos)
	return w2


def uv_space(ant_posx, ant_posy):
	u = np.array([])
	v = np.array([])
	for i in range(NUM_ANTENNAS):
		for j in range(i+1, NUM_ANTENNAS):
			u = np.append(u, (ant_posx[i] - ant_posx[j]) / WAVELENGTH)
			v = np.append(v, (ant_posy[i] - ant_posy[j]) / WAVELENGTH)
			u = np.append(u, (ant_posx[j] - ant_posx[i]) / WAVELENGTH)
			v = np.append(v, (ant_posy[j] - ant_posy[i]) / WAVELENGTH)
	return u, v


def evaluate(ant_posx, ant_posy, u, v, si):
	condition1 = design_conditions(ant_posx, ant_posy)
	condition2 = unique_baselines(ant_posx, ant_posy)
	condition3, mod_x_condition, mod_y_condition = uniformity(u, v, si)
	if condition1 and condition2 and condition3:
		global FLAG
		FLAG = True
		return mod_x_condition, mod_y_condition
	else:
		return mod_x_condition, mod_y_condition


def unique_baselines(ant_posx, ant_posy):
	b = np.array([])
	for i in range(NUM_ANTENNAS):
		for j in range(i+1, NUM_ANTENNAS):
			d = ((ant_posx[i]-ant_posx[j])**2 + (ant_posy[i]-ant_posy[j])**2)**0.5
			b = np.append(b, d)
	ub = len(b)
	for i in range(NUM_ANTENNAS):
		for j in range(i+1, NUM_ANTENNAS):
			if b[i] <= b[j]+0.5 and b[i] >= b[j]-0.5:
					ub = ub - 1
	if ub >= UNIQUE_BASELINES_REQ:
		return True
	else:
		print("FAILED: Not enough unique baselines.")
		return False


def design_conditions(ant_posx, ant_posy):
		return True


def smart_index():
	N = NUM_ANTENNAS*(NUM_ANTENNAS-1)
	si = np.zeros([N,2], dtype=int)
	x = -2
	for i in range(NUM_ANTENNAS):
		for j in range(i+1, NUM_ANTENNAS):
			x += 2
			si[x,0] = i
			si[x,1] = j
			si[x+1,0] = j
			si[x+1,1] = i
	return si


def uniformity(u, v, si):
	save_index = np.array([], dtype=int)
	for i in range(len(u)):
		for j in range(i+1, len(u)):
			r = ((u[i]-u[j])**2 + (v[i]-v[j])**2)**0.5
			if r < UNIFROM_RADIUS:
				save_index = np.append(save_index, i)
				save_index = np.append(save_index, j)
	mod_x_condition = np.array([False] * NUM_ANTENNAS)
	mod_y_condition = np.array([False] * NUM_ANTENNAS)
	for i in save_index:
		mod_x_condition[si[i,0]] = True
		mod_x_condition[si[i,1]] = True
		mod_y_condition[si[i,0]] = True
		mod_y_condition[si[i,1]] = True
	if not any(mod_x_condition) and not any(mod_y_condition):
		condition = True
	else:
		condition = False
	return condition, mod_x_condition, mod_y_condition


def update_modifier(mod_x, mod_y, mod_x_condition, mod_y_condition):
	D = 1.02	# Driving term, make the position have greater pertubation range.	
	S = 10.0	# Dampening term, make the position have less pertubation range.
	for i in range(NUM_ANTENNAS):
		if mod_x_condition[i] == True:
			mod_x[i] = mod_x[i] * D
			if mod_x[i] > 48.0:
				mod_x[i] = 48.0
		if mod_x_condition[i] == False:
			mod_x[i] = mod_x[i] / S
			if mod_x[i] < 0.01:
				mod_x[i] = 0.01
		if mod_y_condition[i] == True:
			mod_y[i] = mod_y[i] * D
			if mod_y[i] > 48.0:
				mod_y[i] = 48.0
		if mod_y_condition[i] == False:
			mod_y[i] = mod_y[i] / S
			if mod_x[i] < 0.01:
				mod_x[i] = 0.01
	return mod_x, mod_y


def export_design(ant_posx, ant_posy):
	print("Optimized Antenna Pattern:")
	print("--------------------------")
	print("X Positions:")
	print(ant_posx)
	print("Y Positions:")
	print(ant_posy)
	print("--------------------------")
	return None


def circle_boundary(r=1.0):
	t = np.arange(0, 2 * PI, 0.001)
	x = r * np.cos(t)
	y = r * np.sin(t)
	return x,y 


if __name__ == "__main__":
	si = smart_index()
	x_bound, y_bound = reuleaux_boundary(3, 48, 0.001)
	initial_posx, initial_posy, weight_x, weight_y = initial_positions(x_bound, y_bound)
	weight_x = initial_posx
	weight_y = initial_posy
	mod_x = np.random.choice([1.0,2.0,3.0], NUM_ANTENNAS)
	mod_y = np.random.choice([1.0,2.0,3.0], NUM_ANTENNAS)
	
	# Keep element 0 and 5 stationary as we desire that baseline.
	mod_x[0] = 0
	mod_y[0] = 0
	mod_x[5] = 0
	mod_y[5] = 0
	mod_x[3] = 0
	mod_y[3] = 0
	mod_x[7] = 0
	mod_y[7] = 0

	# Begin Pertubations
	count = 0
	while FLAG is not True:
		count += 1
		ant_posx, ant_posy, weight_x, weight_y = choose_position(weight_x, weight_y, mod_x, mod_y, x_bound, y_bound)
		u, v = uv_space(ant_posx, ant_posy)
		mod_x_condition, mod_y_condition = evaluate(ant_posx, ant_posy, u, v, si)
		mod_x, mod_y = update_modifier(mod_x, mod_y, mod_x_condition, mod_y_condition)

		if DATA_DUMP == True:
			print("======================================================================")
			print("Iteration:", count)
			print("Modifier State:")
			print(mod_x_condition)
			print("Modifier Value:")
			print(mod_x)
			print("Weights:")
			print(weight_x, weight_y)
			print("======================================================================")
		
			mpl.figure(3)
			mpl.clf()
			mpl.plot(x_bound, y_bound)
			mpl.title("Reuleaux Triangle")
			mpl.axis([-40,40,-30,25])
			mpl.scatter(ant_posx, ant_posy, marker='v', color='red')
			mpl.draw()

			mpl.figure(4)
			mpl.clf()
			a, b = circle_boundary(8)
			mpl.scatter(u,v, color='red')
			mpl.plot(a,b)
			mpl.title("U,V Space")
			mpl.axis([-12,12,-9,9])
			
			mpl.draw()
			mpl.pause(FPS)

		print("Iteration: "+str(count), end='\r')
		if count >= LIMIT:
			break

	print("Total iterations:", count)

	# Save design results
	export_design(ant_posx, ant_posy)

	# Create Plots
	mpl.figure(1)
	mpl.plot(x_bound, y_bound)
	mpl.title("Reuleaux Triangle")
	mpl.scatter(ant_posx, ant_posy, marker='v', color='red')
	mpl.axis('equal')

	mpl.figure(2)
	a, b = circle_boundary(8)
	mpl.scatter(u,v, color='red')
	mpl.plot(a,b)
	mpl.plot(0,0, marker='o')
	mpl.title("U,V Space")
	mpl.axis('equal')

	# Save figures
	# mpl.savefig()

	# Show the results
	mpl.show()
