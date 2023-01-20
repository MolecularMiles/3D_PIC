import math
from random import random
import matplotlib
from matplotlib import pyplot
import matplotlib.cm
import numpy
import imageio.v2 as imageio
import os
from math import pi
import scipy.linalg
from scipy.linalg import solve, inv
from numpy import matmul

#_-_-_-_-_-Global simulation variables-_-_-_-_-_-_
#Control switches for the simluation itself. Imagine a box with dials and levers controlling a small chamber full of plasma. Yes, I am indeed imagining that I had
#the experimental skills to set up such a device. Oh to be talented. Alas I have to content myself with simulating that world.
#The electron and ion parameters.---------

#The lattice we'll assign charges to.---------
side_number = 10
n_x_points = int(side_number)
n_y_points = int(side_number)
n_z_points = int(side_number) 

x_spacing = 1e-5
y_spacing = x_spacing
z_spacing = x_spacing

box_area = (n_x_points*x_spacing)*(n_y_points*y_spacing)*(n_z_points*z_spacing)
wall_width = 1

probe_x = 0
probe_y = 0
probe_charge = 0



n_atoms = 100


m_e = 9.109*(10**(-31))
m_i = 1.602*(10**(-27))

e_charge = float(-1.661*(10**(-19)))
i_charge = float(1.661*(10**(-19)))



#Physical constants. 
boltz = 1.380649*(10**(-23.0))
eps_o = 8.86*(10**(-12))

#Time controls---------
n_steps = 1000

e_number_density = n_atoms/box_area
print("number density: " + str(e_number_density))
e_plasma_freq = (((e_number_density*(e_charge**2.0))/(m_e*eps_o))**0.5)/(2*pi) #Hz
t = 0
dt = 1/(1*e_plasma_freq)
t_f = dt*n_steps





#building the lattice------
lattice = numpy.zeros((n_x_points, n_y_points, n_z_points))

#atom data arrays----------
e_vel_deviation = 0.01
e_positions = numpy.array([])
e_velocities = numpy.random.normal(0, e_vel_deviation, size = (n_atoms, 3))
e_accelerations = numpy.zeros((n_atoms, 3))

i_vel_deviation = 0.01
i_positions = numpy.array([])
i_velocities = numpy.random.normal(0, i_vel_deviation, size = (n_atoms, 3))
i_accelerations = numpy.zeros((n_atoms, 3))

i = 0
while i < n_atoms:
    x = (wall_width*x_spacing) + (x_spacing*(n_x_points-(wall_width)))*random()
    y = (wall_width*y_spacing) + (y_spacing*(n_y_points-(wall_width)))*random()
    z = (wall_width*z_spacing) + (z_spacing*(n_z_points-(wall_width)))*random()
    if (x < float((n_x_points*x_spacing) - (wall_width*x_spacing))) & (x > float((wall_width*x_spacing))):
        if (y < float((n_y_points*y_spacing) - (wall_width*y_spacing))) & (y > float((wall_width*y_spacing))):
            if (z < float((n_z_points*z_spacing) - (wall_width*z_spacing))) & (z > float((wall_width*z_spacing))):
                i += 1
                temp = numpy.array([x])
                temp = numpy.append(temp, y)
                temp = numpy.append(temp, z)
                e_positions = numpy.append(e_positions, temp)

i = 0
while i < n_atoms:
    x = (wall_width*x_spacing) + (x_spacing*(n_x_points-(wall_width)))*random()
    y = (wall_width*y_spacing) + (y_spacing*(n_y_points-(wall_width)))*random()
    z = (wall_width*z_spacing) + (z_spacing*(n_z_points-(wall_width)))*random()
    if (x < float((n_x_points*x_spacing) - (wall_width*x_spacing))) & (x > float((wall_width*x_spacing))):
        if (y < float((n_y_points*y_spacing) - (wall_width*y_spacing))) & (y > float((wall_width*y_spacing))):
            if (z < float((n_z_points*z_spacing) - (wall_width*z_spacing))) & (z > float((wall_width*z_spacing))):
                i += 1
                temp = numpy.array([x])
                temp = numpy.append(temp, y)
                temp = numpy.append(temp, z)
                i_positions = numpy.append(i_positions, temp)

print("Assigned all positions...")
e_positions = e_positions.reshape(n_atoms, 3)
i_positions = i_positions.reshape(n_atoms, 3)
print("Reshaped the position vector...")

print(e_positions)
print(n_x_points*x_spacing)

#Assigning magnetic field (pointed along z) values to the lattice points.
def B_z_field(x, y, z, B_o): #mess with the B_z function to impose any functional form you want, but adjust the timestep accordingly. 
    return(B_o) 
 
b_z_lattice = (numpy.ones((n_y_points, n_x_points, n_z_points)))
for j in range(0, n_x_points):
    for i in range(0, n_y_points):
        for k in range(0, n_z_points):
            b_z_lattice[j, i, k] = B_z_field(x_spacing*(i-(n_x_points/2)), y_spacing*(j-(n_y_points/2)), z_spacing*(k-(n_z_points/2)), 0.000000001)

print("Built the external magnetic field...")


#And now the fun bit: solving the 3d poisson equation. Basically, inverting a N^3 by N^3 matrix that operates on a N^3 long column vector. I encode
#the lattice points in the column vector as [(000), ... (N00), (010), ... (N10), (020), ... (N20), ... ... (0N0), ... (NN0), (0N1), ... ,(NN1), (0N2), ... (NN2), ... (NNN)]

#Building the matrix that applies finite differences (see notes in notebook 4). Essentially using kronecker products of the 1d poisson solver with identity matrices
#to build up the whole solution.

#different boundary conditions we need 3 different seed matrices.
floor_boundary = 0*(-2)
ceiling_boundary = 0*(-2)
right_boundary = 0*(-2)
left_boundary = 0*(-2)
back_boundary = 0*(-2)
front_boundary = 0*(-2)


seed_matrix = numpy.add(numpy.ones((n_x_points, n_x_points)), (-1)*numpy.tri(n_x_points, n_x_points, -2))
seed_matrix = numpy.add(seed_matrix, numpy.transpose((-1)*numpy.tri(n_x_points, n_x_points, -2)))
seed_matrix = numpy.add(seed_matrix, (-3)*numpy.identity(n_x_points))

seed_matrix[0,1] = 0
seed_matrix[n_x_points-1, n_x_points-2] = 0
identity_seed = numpy.identity(n_x_points)

print("Built the seed matrix...")
#now that we have the seed matrix, we can use the kronecker product to build the full 3d poisson solver. M = SxIxI + IxSxI + IxIxS (x = tensor product)

solver_matrix = numpy.add(numpy.add(numpy.kron(numpy.kron(seed_matrix, identity_seed), identity_seed), numpy.kron(numpy.kron(identity_seed, seed_matrix), identity_seed)), numpy.kron(numpy.kron(identity_seed, identity_seed), seed_matrix))
print("Built the laplacian matrix...")
solver_matrix = inv(solver_matrix)

print("Built the finite difference matrix...")

#building the potential and charge vectors.

potential_vector = numpy.zeros((int(n_x_points**3.0), 1))

charge_vector = numpy.zeros((int(n_x_points**3.0), 1))
#inserting the boundary conditions.
boundary_counter = 0 
for i in range(0, n_x_points-1):
    for j in range(0, n_y_points-1):
        for k in range(0, n_z_points-1):
            if i == 0:
                charge_vector[(i + (j*n_x_points) + (k*n_x_points*n_x_points))] = left_boundary
            elif i == n_x_points-1:
                charge_vector[(i + (j*n_x_points) + (k*n_x_points*n_x_points))] = right_boundary
            elif j == 0:
                charge_vector[(i + (j*n_x_points) + (k*n_x_points*n_x_points))] = back_boundary
            elif j == n_x_points-1:
                charge_vector[(i + (j*n_x_points) + (k*n_x_points*n_x_points))] = front_boundary
            elif k == 0:
                charge_vector[(i + (j*n_x_points) + (k*n_x_points*n_x_points))] = floor_boundary
            elif k == n_x_points-1:
                charge_vector[(i + (j*n_x_points) + (k*n_x_points*n_x_points))] = ceiling_boundary
            else:
                boundary_counter+=1
            charge_vector[(i + (j*n_x_points) + (k*n_x_points*n_x_points))] = float(charge_vector[(i + (j*n_x_points) + (k*n_x_points*n_x_points))])

charge_vector_base = charge_vector #between each step, we reset the charge vector to this base state to enforce the boundary conditions. 


#Now we define the functions that we will call in the main simulation. -_-_-_-_-_-____----

#grid assign assigns the electron and ion charges to grid points, weighting by proximity. atom_property[i][dimension] is how we have stored vel, pos, acc, force...

def grid_assign(electron_positions, ion_positions):
    for i in range(0, n_atoms):
        x_pedestal = electron_positions[i, 0]
        y_pedestal = electron_positions[i, 1]
        z_pedestal = electron_positions[i, 2]
        x_float_index, x_int_index = math.modf(x_pedestal/x_spacing) #splitting the lattice assignment into integer and float parts
        y_float_index, y_int_index = math.modf(y_pedestal/y_spacing)
        z_float_index, z_int_index = math.modf(z_pedestal/z_spacing)

        x_forward_weight = float(x_float_index) #defining forwards as away from the bottom left point, backwards as away from the right/up.
        x_backwards_weight = float(1 - x_float_index)
        y_forwards_weight = float(y_float_index)
        y_backwards_weight = float(1 - y_float_index)
        z_forwards_weight = float(z_float_index)
        z_backwards_weight = float(1 - z_float_index)
        
        #the bottom row. 
        #the bottom left point
        x_1 = int(x_int_index) 
        y_1 = int(y_int_index)
        z_1 = int(z_int_index)
        w_1 = float(x_backwards_weight*y_backwards_weight*z_backwards_weight) #weighting the charge assignment by area.

        #extrapolating to the other points. defined moving clockwise from the bottom left point (point 1), then going up a row. 
        #point 2
        x_2 = int(x_int_index)
        y_2 = int(y_int_index+1)
        z_2 = int(z_int_index)
        w_2 = float(x_backwards_weight*y_forwards_weight*z_backwards_weight)
        #point 3
        x_3 = int(x_int_index+1)
        y_3 = int(y_int_index+1)
        z_3 = int(z_int_index)
        w_3 = float(x_forward_weight*y_forwards_weight*z_backwards_weight)
        #point 4
        x_4 = int(x_int_index+1)
        y_4 = int(y_int_index)
        z_4 = int(z_int_index) 
        w_4 = float(x_forward_weight*y_backwards_weight*z_backwards_weight)
        #upper row:

        #point 5
        #the bottom left point
        x_5 = int(x_int_index) 
        y_5 = int(y_int_index)
        z_5 = int(z_int_index+1)
        w_5 = float(x_backwards_weight*y_backwards_weight*z_forwards_weight) #weighting the charge assignment by area.


        #point 6
        x_6 = int(x_int_index)
        y_6 = int(y_int_index+1)
        z_6 = int(z_int_index+1)
        w_6 = float(x_backwards_weight*y_forwards_weight*z_forwards_weight)
        #point 7
        x_7 = int(x_int_index+1)
        y_7 = int(y_int_index+1)
        z_7 = int(z_int_index+1)
        w_7 = float(x_forward_weight*y_forwards_weight*z_forwards_weight)
        #point 8
        x_8 = int(x_int_index+1)
        y_8 = int(y_int_index)
        z_8 = int(z_int_index+1) 
        w_8 = float(x_forward_weight*y_backwards_weight*z_forwards_weight)

        charge_vector[((n_x_points-1)*y_1)+ x_1+ ((n_x_points-1)*(n_x_points-1)*z_1)] += ((w_1*e_charge)) 
        charge_vector[((n_x_points-1)*y_2)+ x_2+ ((n_x_points-1)*(n_x_points-1)*z_2)] += ((w_2*e_charge))
        charge_vector[((n_x_points-1)*y_3)+ x_3+ ((n_x_points-1)*(n_x_points-1)*z_3)] += ((w_3*e_charge))
        charge_vector[((n_x_points-1)*y_4)+ x_4+ ((n_x_points-1)*(n_x_points-1)*z_4)] += ((w_4*e_charge))
        charge_vector[((n_x_points-1)*y_5)+ x_5+ ((n_x_points-1)*(n_x_points-1)*z_5)] += ((w_5*e_charge)) 
        charge_vector[((n_x_points-1)*y_6)+ x_6+ ((n_x_points-1)*(n_x_points-1)*z_6)] += ((w_6*e_charge))
        charge_vector[((n_x_points-1)*y_7)+ x_7+ ((n_x_points-1)*(n_x_points-1)*z_7)] += ((w_7*e_charge))
        charge_vector[((n_x_points-1)*y_8)+ x_8+ ((n_x_points-1)*(n_x_points-1)*z_8)] += ((w_8*e_charge))


    for i in range(0, n_atoms):
        x_pedestal = ion_positions[i, 0]
        y_pedestal = ion_positions[i, 1]
        z_pedestal = ion_positions[i, 2]
        x_float_index, x_int_index = math.modf(x_pedestal/x_spacing) #splitting the lattice assignment into integer and float parts
        y_float_index, y_int_index = math.modf(y_pedestal/y_spacing)
        z_float_index, z_int_index = math.modf(z_pedestal/z_spacing)

        x_forward_weight = float(x_float_index) #defining forwards as away from the bottom left point, backwards as away from the right/up.
        x_backwards_weight = float(1 - x_float_index)
        y_forwards_weight = float(y_float_index)
        y_backwards_weight = float(1 - y_float_index)
        z_forwards_weight = float(z_float_index)
        z_backwards_weight = float(1 - z_float_index)
        
        #the bottom row. 
        #the bottom left point
        x_1 = int(x_int_index) 
        y_1 = int(y_int_index)
        z_1 = int(z_int_index)
        w_1 = float(x_backwards_weight*y_backwards_weight*z_backwards_weight) #weighting the charge assignment by area.

        #extrapolating to the other points. defined moving clockwise from the bottom left point (point 1), then going up a row. 
        #point 2
        x_2 = int(x_int_index)
        y_2 = int(y_int_index+1)
        z_2 = int(z_int_index)
        w_2 = float(x_backwards_weight*y_forwards_weight*z_backwards_weight)
        #point 3
        x_3 = int(x_int_index+1)
        y_3 = int(y_int_index+1)
        z_3 = int(z_int_index)
        w_3 = float(x_forward_weight*y_forwards_weight*z_backwards_weight)
        #point 4
        x_4 = int(x_int_index+1)
        y_4 = int(y_int_index)
        z_4 = int(z_int_index) 
        w_4 = float(x_forward_weight*y_backwards_weight*z_backwards_weight)
        #upper row:

        #point 5
        #the bottom left point
        x_5 = int(x_int_index) 
        y_5 = int(y_int_index)
        z_5 = int(z_int_index+1)
        w_5 = float(x_backwards_weight*y_backwards_weight*z_forwards_weight) #weighting the charge assignment by area.


        #point 6
        x_6 = int(x_int_index)
        y_6 = int(y_int_index+1)
        z_6 = int(z_int_index+1)
        w_6 = float(x_backwards_weight*y_forwards_weight*z_forwards_weight)
        #point 7
        x_7 = int(x_int_index+1)
        y_7 = int(y_int_index+1)
        z_7 = int(z_int_index+1)
        w_7 = float(x_forward_weight*y_forwards_weight*z_forwards_weight)
        #point 8
        x_8 = int(x_int_index+1)
        y_8 = int(y_int_index)
        z_8 = int(z_int_index+1) 
        w_8 = float(x_forward_weight*y_backwards_weight*z_forwards_weight)

        charge_vector[((n_x_points-1)*y_1)+ x_1+ ((n_x_points-1)*(n_x_points-1)*z_1)] += ((w_1*i_charge)) 
        charge_vector[((n_x_points-1)*y_2)+ x_2+ ((n_x_points-1)*(n_x_points-1)*z_2)] += ((w_2*i_charge))
        charge_vector[((n_x_points-1)*y_3)+ x_3+ ((n_x_points-1)*(n_x_points-1)*z_3)] += ((w_3*i_charge))
        charge_vector[((n_x_points-1)*y_4)+ x_4+ ((n_x_points-1)*(n_x_points-1)*z_4)] += ((w_4*i_charge))
        charge_vector[((n_x_points-1)*y_5)+ x_5+ ((n_x_points-1)*(n_x_points-1)*z_5)] += ((w_5*i_charge)) 
        charge_vector[((n_x_points-1)*y_6)+ x_6+ ((n_x_points-1)*(n_x_points-1)*z_6)] += ((w_6*i_charge))
        charge_vector[((n_x_points-1)*y_7)+ x_7+ ((n_x_points-1)*(n_x_points-1)*z_7)] += ((w_7*i_charge))
        charge_vector[((n_x_points-1)*y_8)+ x_8+ ((n_x_points-1)*(n_x_points-1)*z_8)] += ((w_8*i_charge))
    #charge_lattice = numpy.flip(charge_lattice, 1)
    #charge_lattice = numpy.flip(charge_lattice, 0)

    charge_vector[probe_y, probe_x] = probe_charge#chucking a charge in there to see if we get debye screening. 
    return(charge_vector)


#Now we calculate the E field at each lattice point, and store those in vectors. 

def calculate_electric_field(charge_vector):
    pot_vector = numpy.matmul(solver_matrix, charge_vector)
    e_x_vector = numpy.array([])
    e_y_vector = numpy.array([])
    e_z_vector = numpy.array([])
    for k in range(0, n_x_points):
        for j in range(0, n_x_points):
            for i in range(0, n_x_points):
                e_x_vector = numpy.append(e_x_vector, (pot_vector[(i-1) + ((n_x_points-1)*j) + ((n_x_points-1)*(n_x_points-1)*k)] - pot_vector[(i+1) + ((n_x_points-1)*j) + ((n_x_points-1)*(n_x_points-1)*k)])/(2*x_spacing*x_spacing))
                e_y_vector = numpy.append(e_y_vector, (pot_vector[(i) + ((n_x_points-1)*(j-1)) + ((n_x_points-1)*(n_x_points-1)*k)] - pot_vector[(i) + ((n_x_points-1)*(j+1)) + ((n_x_points-1)*(n_x_points-1)*k)])/(2*x_spacing*x_spacing))               
                e_z_vector = numpy.append(e_z_vector, (pot_vector[(i) + ((n_x_points-1)*j) + ((n_x_points-1)*(n_x_points-1)*(k-1))] - pot_vector[(i) + ((n_x_points-1)*j) + ((n_x_points-1)*(n_x_points-1)*(k+1))])/(2*x_spacing*x_spacing))
    return(e_x_vector, e_y_vector, e_z_vector)



def get_accelerations(electron_positions, electron_velocities, ion_positions, ion_velocities, e_x, e_y, e_z, b_z):
    #the field felt by a charge is weighted in the same way as their charge was assigned to grid positions. 
    #calculating the forces on the electrons. 
    electron_accelerations = numpy.array([])
    for i in range(0, n_atoms):
        x_pedestal = electron_positions[i, 0]
        y_pedestal = electron_positions[i, 1]
        z_pedestal = electron_positions[i, 2]
        vx_pedestal = electron_velocities[i, 0]
        vy_pedestal = electron_velocities[i, 1]
        vz_pedestal = electron_velocities[i, 2]
        x_float_index, x_int_index = math.modf(x_pedestal/x_spacing) #splitting the lattice assignment into integer and float parts
        y_float_index, y_int_index = math.modf(y_pedestal/y_spacing)
        z_float_index, z_int_index = math.modf(z_pedestal/z_spacing)

        x_int_index = int(x_int_index)
        y_int_index = int(y_int_index)
        z_int_index = int(z_int_index)


        x_forward_weight = x_float_index #defining forwards as away from the bottom left point, backwards as away from the right/up.
        x_backwards_weight = 1 - x_float_index
        y_forwards_weight = y_float_index
        y_backwards_weight = 1 - y_float_index
        z_forwards_weight = z_float_index
        z_backwards_weight = 1 - z_float_index

        e_x_felt = e_x[(x_int_index) + ((y_int_index)*(n_x_points-1)) + ((z_int_index)*(n_x_points-1)*(n_x_points-1))]*x_backwards_weight*y*y_backwards_weight*z_backwards_weight + e_x[(x_int_index+1) + ((y_int_index)*(n_x_points-1)) + ((z_int_index)*(n_x_points-1)*(n_x_points-1))]*x_forward_weight*y*y_backwards_weight*z_backwards_weight + e_x[(x_int_index+1) + ((y_int_index+1)*(n_x_points-1)) + ((z_int_index)*(n_x_points-1)*(n_x_points-1))]*x_forward_weight*y*y_forwards_weight*z_backwards_weight + e_x[(x_int_index+1) + ((y_int_index+1)*(n_x_points-1)) + ((z_int_index+1)*(n_x_points-1)*(n_x_points-1))]*x_forward_weight*y*y_forwards_weight*z_forwards_weight + e_x[(x_int_index) + ((y_int_index+1)*(n_x_points-1)) + ((z_int_index)*(n_x_points-1)*(n_x_points-1))]*x_backwards_weight*y*y_forwards_weight*z_backwards_weight + e_x[(x_int_index) + ((y_int_index+1)*(n_x_points-1)) + ((z_int_index+1)*(n_x_points-1)*(n_x_points-1))]*x_backwards_weight*y*y_forwards_weight*z_forwards_weight + e_x[(x_int_index) + ((y_int_index)*(n_x_points-1)) + ((z_int_index+1)*(n_x_points-1)*(n_x_points-1))]*x_backwards_weight*y*y_backwards_weight*z_forwards_weight + e_x[(x_int_index+1) + ((y_int_index)*(n_x_points-1)) + ((z_int_index+1)*(n_x_points-1)*(n_x_points-1))]*x_forward_weight*y*y_backwards_weight*z_forwards_weight
        e_y_felt = e_y[(x_int_index) + ((y_int_index)*(n_x_points-1)) + ((z_int_index)*(n_x_points-1)*(n_x_points-1))]*x_backwards_weight*y*y_backwards_weight*z_backwards_weight + e_y[(x_int_index+1) + ((y_int_index)*(n_x_points-1)) + ((z_int_index)*(n_x_points-1)*(n_x_points-1))]*x_forward_weight*y*y_backwards_weight*z_backwards_weight + e_y[(x_int_index+1) + ((y_int_index+1)*(n_x_points-1)) + ((z_int_index)*(n_x_points-1)*(n_x_points-1))]*x_forward_weight*y*y_forwards_weight*z_backwards_weight + e_y[(x_int_index+1) + ((y_int_index+1)*(n_x_points-1)) + ((z_int_index+1)*(n_x_points-1)*(n_x_points-1))]*x_forward_weight*y*y_forwards_weight*z_forwards_weight + e_y[(x_int_index) + ((y_int_index+1)*(n_x_points-1)) + ((z_int_index)*(n_x_points-1)*(n_x_points-1))]*x_backwards_weight*y*y_forwards_weight*z_backwards_weight + e_y[(x_int_index) + ((y_int_index+1)*(n_x_points-1)) + ((z_int_index+1)*(n_x_points-1)*(n_x_points-1))]*x_backwards_weight*y*y_forwards_weight*z_forwards_weight + e_y[(x_int_index) + ((y_int_index)*(n_x_points-1)) + ((z_int_index+1)*(n_x_points-1)*(n_x_points-1))]*x_backwards_weight*y*y_backwards_weight*z_forwards_weight + e_y[(x_int_index+1) + ((y_int_index)*(n_x_points-1)) + ((z_int_index+1)*(n_x_points-1)*(n_x_points-1))]*x_forward_weight*y*y_backwards_weight*z_forwards_weight   
        e_z_felt = e_z[(x_int_index) + ((y_int_index)*(n_x_points-1)) + ((z_int_index)*(n_x_points-1)*(n_x_points-1))]*x_backwards_weight*y*y_backwards_weight*z_backwards_weight + e_z[(x_int_index+1) + ((y_int_index)*(n_x_points-1)) + ((z_int_index)*(n_x_points-1)*(n_x_points-1))]*x_forward_weight*y*y_backwards_weight*z_backwards_weight + e_z[(x_int_index+1) + ((y_int_index+1)*(n_x_points-1)) + ((z_int_index)*(n_x_points-1)*(n_x_points-1))]*x_forward_weight*y*y_forwards_weight*z_backwards_weight + e_z[(x_int_index+1) + ((y_int_index+1)*(n_x_points-1)) + ((z_int_index+1)*(n_x_points-1)*(n_x_points-1))]*x_forward_weight*y*y_forwards_weight*z_forwards_weight + e_z[(x_int_index) + ((y_int_index+1)*(n_x_points-1)) + ((z_int_index)*(n_x_points-1)*(n_x_points-1))]*x_backwards_weight*y*y_forwards_weight*z_backwards_weight + e_z[(x_int_index) + ((y_int_index+1)*(n_x_points-1)) + ((z_int_index+1)*(n_x_points-1)*(n_x_points-1))]*x_backwards_weight*y*y_forwards_weight*z_forwards_weight + e_z[(x_int_index) + ((y_int_index)*(n_x_points-1)) + ((z_int_index+1)*(n_x_points-1)*(n_x_points-1))]*x_backwards_weight*y*y_backwards_weight*z_forwards_weight + e_z[(x_int_index+1) + ((y_int_index)*(n_x_points-1)) + ((z_int_index+1)*(n_x_points-1)*(n_x_points-1))]*x_forward_weight*y*y_backwards_weight*z_forwards_weight

        #b_felt = b_z[(x_int_index) , ((y_int_index)) , ((z_int_index))]*x_backwards_weight*y*y_backwards_weight*z_backwards_weight + b_z[(x_int_index+1) , ((y_int_index)) , ((z_int_index))]*x_forward_weight*y*y_backwards_weight*z_backwards_weight + b_z[(x_int_index+1) , ((y_int_index+1)) , ((z_int_index))]*x_forward_weight*y*y_forwards_weight*z_backwards_weight + b_z[(x_int_index+1) , ((y_int_index+1)) , ((z_int_index+1))]*x_forward_weight*y*y_forwards_weight*z_forwards_weight + b_z[(x_int_index) , ((y_int_index+1)) , ((z_int_index))]*x_backwards_weight*y*y_forwards_weight*z_backwards_weight + b_z[(x_int_index) , ((y_int_index+1)) , ((z_int_index+1))]*x_backwards_weight*y*y_forwards_weight*z_forwards_weight + b_z[(x_int_index) , ((y_int_index)) , ((z_int_index+1))]*x_backwards_weight*y*y_backwards_weight*z_forwards_weight + b_z[(x_int_index+1) , ((y_int_index)) , ((z_int_index+1))]*x_forward_weight*y*y_backwards_weight*z_forwards_weight
        #e_a_x_from_b = vy_pedestal*b_felt*(e_charge/m_e)
        #e_a_y_from_b = -vx_pedestal*b_felt*(e_charge/m_e)
        e_a_z_from_b = 0  #just in case I add more directions for the field to point along.
        e_a_x_from_b = 0
        e_a_y_from_b = 0
        e_a_x_from_e = e_x_felt*(e_charge/m_e) 
        e_a_y_from_e = e_y_felt*(e_charge/m_e) 
        e_a_z_from_e = e_z_felt*(e_charge/m_e) 

        e_a_x = e_a_x_from_b + e_a_x_from_e
        e_a_y = e_a_y_from_b + e_a_y_from_e
        e_a_z = e_a_z_from_b + e_a_z_from_e
        temp = numpy.array([])
        temp = numpy.append(temp, e_a_x)
        temp = numpy.append(temp, e_a_y)
        temp = numpy.append(temp, e_a_z)
        electron_accelerations = numpy.append(electron_accelerations, temp)
    electron_accelerations = electron_accelerations.reshape(n_atoms, 3)
    #now doing the same calculation for the ions. 
    ion_accelerations = numpy.array([])
    for i in range(0, n_atoms):
        x_pedestal = ion_positions[i, 0]
        y_pedestal = ion_positions[i, 1]
        z_pedestal = ion_positions[i, 2]
        vx_pedestal = ion_velocities[i, 0]
        vy_pedestal = ion_velocities[i, 1]
        vz_pedestal = ion_velocities[i, 2]
        x_float_index, x_int_index = math.modf(x_pedestal/x_spacing) #splitting the lattice assignment into integer and float parts
        y_float_index, y_int_index = math.modf(y_pedestal/y_spacing)
        z_float_index, z_int_index = math.modf(z_pedestal/z_spacing)

        x_int_index = int(x_int_index)
        y_int_index = int(y_int_index)
        z_int_index = int(z_int_index)

        x_forward_weight = x_float_index #defining forwards as away from the bottom left point, backwards as away from the right/up.
        x_backwards_weight = 1 - x_float_index
        y_forwards_weight = y_float_index
        y_backwards_weight = 1 - y_float_index
        z_forwards_weight = z_float_index
        z_backwards_weight = 1 - z_float_index

        e_x_felt = e_x[(x_int_index) + ((y_int_index)*(n_x_points-1)) + ((z_int_index)*(n_x_points-1)*(n_x_points-1))]*x_backwards_weight*y*y_backwards_weight*z_backwards_weight + e_x[(x_int_index+1) + ((y_int_index)*(n_x_points-1)) + ((z_int_index)*(n_x_points-1)*(n_x_points-1))]*x_forward_weight*y*y_backwards_weight*z_backwards_weight + e_x[(x_int_index+1) + ((y_int_index+1)*(n_x_points-1)) + ((z_int_index)*(n_x_points-1)*(n_x_points-1))]*x_forward_weight*y*y_forwards_weight*z_backwards_weight + e_x[(x_int_index+1) + ((y_int_index+1)*(n_x_points-1)) + ((z_int_index+1)*(n_x_points-1)*(n_x_points-1))]*x_forward_weight*y*y_forwards_weight*z_forwards_weight + e_x[(x_int_index) + ((y_int_index+1)*(n_x_points-1)) + ((z_int_index)*(n_x_points-1)*(n_x_points-1))]*x_backwards_weight*y*y_forwards_weight*z_backwards_weight + e_x[(x_int_index) + ((y_int_index+1)*(n_x_points-1)) + ((z_int_index+1)*(n_x_points-1)*(n_x_points-1))]*x_backwards_weight*y*y_forwards_weight*z_forwards_weight + e_x[(x_int_index) + ((y_int_index)*(n_x_points-1)) + ((z_int_index+1)*(n_x_points-1)*(n_x_points-1))]*x_backwards_weight*y*y_backwards_weight*z_forwards_weight + e_x[(x_int_index+1) + ((y_int_index)*(n_x_points-1)) + ((z_int_index+1)*(n_x_points-1)*(n_x_points-1))]*x_forward_weight*y*y_backwards_weight*z_forwards_weight
        e_y_felt = e_y[(x_int_index) + ((y_int_index)*(n_x_points-1)) + ((z_int_index)*(n_x_points-1)*(n_x_points-1))]*x_backwards_weight*y*y_backwards_weight*z_backwards_weight + e_y[(x_int_index+1) + ((y_int_index)*(n_x_points-1)) + ((z_int_index)*(n_x_points-1)*(n_x_points-1))]*x_forward_weight*y*y_backwards_weight*z_backwards_weight + e_y[(x_int_index+1) + ((y_int_index+1)*(n_x_points-1)) + ((z_int_index)*(n_x_points-1)*(n_x_points-1))]*x_forward_weight*y*y_forwards_weight*z_backwards_weight + e_y[(x_int_index+1) + ((y_int_index+1)*(n_x_points-1)) + ((z_int_index+1)*(n_x_points-1)*(n_x_points-1))]*x_forward_weight*y*y_forwards_weight*z_forwards_weight + e_y[(x_int_index) + ((y_int_index+1)*(n_x_points-1)) + ((z_int_index)*(n_x_points-1)*(n_x_points-1))]*x_backwards_weight*y*y_forwards_weight*z_backwards_weight + e_y[(x_int_index) + ((y_int_index+1)*(n_x_points-1)) + ((z_int_index+1)*(n_x_points-1)*(n_x_points-1))]*x_backwards_weight*y*y_forwards_weight*z_forwards_weight + e_y[(x_int_index) + ((y_int_index)*(n_x_points-1)) + ((z_int_index+1)*(n_x_points-1)*(n_x_points-1))]*x_backwards_weight*y*y_backwards_weight*z_forwards_weight + e_y[(x_int_index+1) + ((y_int_index)*(n_x_points-1)) + ((z_int_index+1)*(n_x_points-1)*(n_x_points-1))]*x_forward_weight*y*y_backwards_weight*z_forwards_weight   
        e_z_felt = e_z[(x_int_index) + ((y_int_index)*(n_x_points-1)) + ((z_int_index)*(n_x_points-1)*(n_x_points-1))]*x_backwards_weight*y*y_backwards_weight*z_backwards_weight + e_z[(x_int_index+1) + ((y_int_index)*(n_x_points-1)) + ((z_int_index)*(n_x_points-1)*(n_x_points-1))]*x_forward_weight*y*y_backwards_weight*z_backwards_weight + e_z[(x_int_index+1) + ((y_int_index+1)*(n_x_points-1)) + ((z_int_index)*(n_x_points-1)*(n_x_points-1))]*x_forward_weight*y*y_forwards_weight*z_backwards_weight + e_z[(x_int_index+1) + ((y_int_index+1)*(n_x_points-1)) + ((z_int_index+1)*(n_x_points-1)*(n_x_points-1))]*x_forward_weight*y*y_forwards_weight*z_forwards_weight + e_z[(x_int_index) + ((y_int_index+1)*(n_x_points-1)) + ((z_int_index)*(n_x_points-1)*(n_x_points-1))]*x_backwards_weight*y*y_forwards_weight*z_backwards_weight + e_z[(x_int_index) + ((y_int_index+1)*(n_x_points-1)) + ((z_int_index+1)*(n_x_points-1)*(n_x_points-1))]*x_backwards_weight*y*y_forwards_weight*z_forwards_weight + e_z[(x_int_index) + ((y_int_index)*(n_x_points-1)) + ((z_int_index+1)*(n_x_points-1)*(n_x_points-1))]*x_backwards_weight*y*y_backwards_weight*z_forwards_weight + e_z[(x_int_index+1) + ((y_int_index)*(n_x_points-1)) + ((z_int_index+1)*(n_x_points-1)*(n_x_points-1))]*x_forward_weight*y*y_backwards_weight*z_forwards_weight

        #b_felt = b_z[(x_int_index) , ((y_int_index)) , ((z_int_index))]*x_backwards_weight*y*y_backwards_weight*z_backwards_weight + b_z[(x_int_index+1) , ((y_int_index)) , ((z_int_index))]*x_forward_weight*y*y_backwards_weight*z_backwards_weight + b_z[(x_int_index+1) , ((y_int_index+1)) , ((z_int_index))]*x_forward_weight*y*y_forwards_weight*z_backwards_weight + b_z[(x_int_index+1) , ((y_int_index+1)) , ((z_int_index+1))]*x_forward_weight*y*y_forwards_weight*z_forwards_weight + b_z[(x_int_index) , ((y_int_index+1)) , ((z_int_index))]*x_backwards_weight*y*y_forwards_weight*z_backwards_weight + b_z[(x_int_index) , ((y_int_index+1)) , ((z_int_index+1))]*x_backwards_weight*y*y_forwards_weight*z_forwards_weight + b_z[(x_int_index) , ((y_int_index)) , ((z_int_index+1))]*x_backwards_weight*y*y_backwards_weight*z_forwards_weight + b_z[(x_int_index+1) , ((y_int_index)) , ((z_int_index+1))]*x_forward_weight*y*y_backwards_weight*z_forwards_weight
        #i_a_x_from_b = vy_pedestal*b_felt*(i_charge/m_i)
        #i_a_y_from_b = -vx_pedestal*b_felt*(i_charge/m_i)
        i_a_z_from_b = 0  #just in case I add more directions for the field to point along.
        i_a_x_from_b = 0
        i_a_y_from_b = 0
        i_a_x_from_e = e_x_felt*(i_charge/m_i) 
        i_a_y_from_e = e_y_felt*(i_charge/m_i) 
        i_a_z_from_e = e_z_felt*(i_charge/m_i) 

        i_a_x = i_a_x_from_b + i_a_x_from_e
        i_a_y = i_a_y_from_b + i_a_y_from_e
        i_a_z = i_a_z_from_b + i_a_z_from_e
        temp = numpy.array([])
        temp = numpy.append(temp, i_a_x)
        temp = numpy.append(temp, i_a_y)
        temp = numpy.append(temp, i_a_z)
        ion_accelerations = numpy.append(ion_accelerations, temp)
    ion_accelerations = ion_accelerations.reshape(n_atoms, 3)
    return(electron_accelerations, ion_accelerations)


#Now the function that performs the forward stepping in time. 
def Verlet_Step(electron_position, electron_veloc, electron_accel, ion_position, ion_veloc, ion_accel, dt):
    for i in range(0, n_atoms):
        electron_position[i,0] += dt*electron_veloc[i,0] + 0.5*(dt**2.0)*electron_accel[i, 0]
        electron_position[i,1] += dt*electron_veloc[i,1] + 0.5*(dt**2.0)*electron_accel[i, 1]
        electron_position[i,2] += dt*electron_veloc[i,2] + 0.5*(dt**2.0)*electron_accel[i, 2]
        ion_position[i,0] += dt*ion_veloc[i,0] + 0.5*(dt**2.0)*ion_accel[i, 0]
        ion_position[i,1] += dt*ion_veloc[i,1] + 0.5*(dt**2.0)*ion_accel[i, 1]
        ion_position[i,2] += dt*ion_veloc[i,2] + 0.5*(dt**2.0)*ion_accel[i, 2]
    electron_old_accel = electron_accel
    ion_old_accel = ion_accel

    new_field_x, new_field_y, new_field_z = calculate_electric_field(grid_assign(electron_position, ion_position))
    electron_accel, ion_accel = get_accelerations(electron_position, electron_veloc, ion_position, ion_veloc, new_field_x, new_field_y, new_field_z, b_z_lattice)
    for i in range(0, n_atoms):
        electron_veloc[i, 0] += 0.5*(electron_old_accel[i, 0] + electron_accel[i, 0])*dt
        electron_veloc[i, 1] += 0.5*(electron_old_accel[i, 1] + electron_accel[i, 1])*dt
        electron_veloc[i, 2] += 0.5*(electron_old_accel[i, 2] + electron_accel[i, 2])*dt
        ion_veloc[i, 0] += 0.5*(ion_old_accel[i, 0] + ion_accel[i, 0])*dt
        ion_veloc[i, 1] += 0.5*(ion_old_accel[i, 1] + ion_accel[i, 1])*dt
        ion_veloc[i, 2] += 0.5*(ion_old_accel[i, 2] + ion_accel[i, 2])*dt
    
    for i in range(0, n_atoms): #Hard Wall BC
        if electron_position[i, 0] >= (n_x_points-(wall_width+1))*x_spacing:
            electron_veloc[i, 0] = -electron_veloc[i, 0]
        
        if electron_position[i, 0] <= (wall_width+1)*x_spacing:
            electron_veloc[i, 0] = -electron_veloc[i, 0]
        
        if electron_position[i, 1] >= (n_y_points-(wall_width+1))*y_spacing:
            electron_veloc[i, 1] = -electron_veloc[i, 1]
        
        if electron_position[i, 1] <= (wall_width+1)*y_spacing:
            electron_veloc[i, 1] = -electron_veloc[i, 1]

        if electron_position[i, 2] >= (n_z_points-(wall_width+1))*z_spacing:
            electron_veloc[i, 2] = -electron_veloc[i, 2]
        
        if electron_position[i, 2] <= (wall_width+1)*z_spacing:
            electron_veloc[i, 2] = -electron_veloc[i, 2]
        
        if ion_position[i, 0] >= (n_x_points-(wall_width+1))*x_spacing:
            ion_veloc[i, 0] = -ion_veloc[i, 0]
        
        if ion_position[i, 0] <= (wall_width+1)*x_spacing:
            ion_veloc[i, 0] = -ion_veloc[i, 0]
        
        if ion_position[i, 1] >= (n_y_points-(wall_width+1))*y_spacing:
            ion_veloc[i, 1] = -ion_veloc[i, 1]
        
        if ion_position[i, 1] <= (wall_width+1)*y_spacing:
            ion_veloc[i, 1] = -ion_veloc[i, 1]

        if ion_position[i, 2] >= (n_z_points-(wall_width+1))*z_spacing:
            ion_veloc[i, 2] = -ion_veloc[i, 2]
        
        if ion_position[i, 2] <= (wall_width+1)*z_spacing:
            ion_veloc[i, 2] = -ion_veloc[i, 2]
    return(electron_position, electron_veloc, electron_accel, ion_position, ion_veloc, ion_accel, new_field_x, new_field_y, new_field_z)


#The Main Simulation Run.
e_x_plot = numpy.array([])
e_y_plot = numpy.array([])
e_z_plot = numpy.array([])

e_x_temp = numpy.array([])
e_y_temp = numpy.array([])
e_z_temp = numpy.array([])

electron_x_position_plot = e_positions[:, 0]
electron_y_position_plot = e_positions[:, 1]
electron_z_position_plot = e_positions[:, 2]

electron_vx_position_plot = e_velocities[:, 0]
electron_vy_position_plot = e_velocities[:, 1]
electron_vz_position_plot = e_velocities[:, 2]

ion_x_position_plot = i_positions[:, 0]
ion_y_position_plot = i_positions[:, 1]
ion_z_position_plot = i_positions[:, 2]

ion_vx_position_plot = i_velocities[:, 0]
ion_vy_position_plot = i_velocities[:, 1]
ion_vz_position_plot = i_velocities[:, 2]

counter = 0
sample_rate = 1
print("Running the main simulation...")
print("side length " +str(n_x_points*x_spacing))
print(e_positions)
while t <= t_f:
    t += dt
    e_positions, e_velocities, e_accelerations, i_positions, i_velocities, i_accelerations, e_x_temp, e_y_temp, e_z_temp = Verlet_Step(e_positions, e_velocities, e_accelerations, i_positions, i_velocities, i_accelerations, dt)
    charge_vector = charge_vector_base
    if counter%sample_rate == 0:
        print("made it to timestep " + str(counter))
        
        electron_x_position_plot = numpy.vstack((electron_x_position_plot, [e_positions[:, 0]]))
        electron_y_position_plot = numpy.vstack((electron_y_position_plot, [e_positions[:, 1]]))
        electron_z_position_plot = numpy.vstack((electron_z_position_plot, [e_positions[:, 2]]))

        electron_vx_position_plot = numpy.vstack((electron_vx_position_plot, [e_velocities[:, 0]]))
        electron_vy_position_plot = numpy.vstack((electron_vy_position_plot, [e_velocities[:, 1]]))
        electron_vz_position_plot = numpy.vstack((electron_vz_position_plot, [e_velocities[:, 2]]))

        ion_x_position_plot = numpy.vstack((ion_x_position_plot, [i_positions[:, 0]]))
        ion_y_position_plot = numpy.vstack((ion_y_position_plot, [i_positions[:, 1]]))
        ion_z_position_plot = numpy.vstack((ion_z_position_plot, [i_positions[:, 2]]))

        ion_vx_position_plot = numpy.vstack((ion_vx_position_plot, [i_velocities[:, 0]]))
        ion_vy_position_plot = numpy.vstack((ion_vy_position_plot, [i_velocities[:, 1]]))
        ion_vz_position_plot = numpy.vstack((ion_vz_position_plot, [i_velocities[:, 2]]))


    counter += 1

print(electron_x_position_plot)

with open("PIC_POSITIONS", "w") as f: #writing position data in such a way that OVITO accepts it as a lammps dump file. 
    for timestep in range(0, int(n_steps/sample_rate)):
        f.write("ITEM: TIMESTEP")
        f.write("\n") 
        f.write(str(timestep*sample_rate))
        f.write("\n")
        f.write("ITEM: NUMBER OF ATOMS")
        f.write("\n")
        f.write(str(2*n_atoms))
        f.write("\n")
        f.write("ITEM: BOX BOUNDS pp pp pp")
        f.write("\n")
        f.write("0.0000000000000000e+00 " + str(x_spacing*n_x_points))
        f.write("\n")
        f.write("0.0000000000000000e+00 " + str(x_spacing*n_x_points))  
        f.write("\n")
        f.write("0.0000000000000000e+00 " + str(x_spacing*n_x_points))
        f.write("\n")
        f.write("ITEM: ATOMS id type xs ys zs")
        f.write("\n")
        for n in range(0, n_atoms):
            f.write(str(n+1) + " " + str(1) + " " + str(electron_x_position_plot[timestep, n]/(x_spacing*n_x_points)) + " " + str(electron_y_position_plot[timestep, n]/(x_spacing*n_x_points))+ " " + str(electron_z_position_plot[timestep, n]/(x_spacing*n_x_points))) 
            f.write("\n")   
            f.write(str(n+1 +n_atoms) + " " + str(2) + " " + str(ion_x_position_plot[timestep, n]/(x_spacing*n_x_points)) + " " + str(ion_y_position_plot[timestep, n]/(x_spacing*n_x_points))+ " " + str(ion_z_position_plot[timestep, n]/(x_spacing*n_x_points))) 
            f.write("\n")   



