from math import pi, sin
from unittest import result
import numpy
from matplotlib import pyplot, rc, rcParams
from numpy.lib.type_check import real
from numpy import arcsin, arctan, exp
from scipy.special import kn

rc("xtick", labelsize = 13)
rc("ytick", labelsize = 13)
rcParams.update({"font.size":13})



#The retativistic runs always look like they have frozen on the first few timesteps, but as soon as the atoms start interacting, the simulation starts chopping down
#the stored trajectory list it has to search through, and speeds up massively.  
""" 
Constant variables (Ignore)
"""
c = 2.998*(10**8)
c_2 = c*c
box_size = 28.982
mass = 39.95 #g/mol
mass_to_kg = (10**(-3.0))/(6.022*(10**23.0))
length_to_m = 10**(-10.0)
mass_kg = mass*mass_to_kg
rest_mass_energy = mass_kg*(c**2) #SI units
ev_to_J = 1.60218e-19
J_to_GeV = 1/(ev_to_J*(10**(9.0)))
k_to_GeV = 11604525006170.0
boltz = 1.380649*(10**(-23)) #SI


# Variables for the LJ Potential. 
epsilon = 0.01034 #ev
eps = epsilon*ev_to_J #in Joules. I have a small brain that still thinks in SI. 
sigma = 3.4*(10**(-10)) #m
r_c = sigma*2.25
r_c_2 = r_c**2.0


#function settings (To avoid the function definitions crashing. Ignore)
b_init = 0.01*r_c #vertical offset (impact parameter)
dist_init = 1.0*r_c #horizontal offset.
sep_init = ((b_init**2.0) + (dist_init**2.0))**0.5
vx1_init = 0.01*c #initial horizontal speed of the leftmost atom
vx2_init = -0.01*c #initial horizontal speed of the rightmost atom. 
v_init = 0
vy1_init = 0.0
vy2_init = 0.0
g_init = "0_00002c"
b_init_text = "0_01r_c"

dt = 1e-21
dt = (0.001*c)/(abs(vx1_init - vx2_init))*dt  #a rough rule of thumb for how you should scale the timestep. Feel free to change.
dt = 1e-22
n_timesteps = 4000000 #Number of timesteps that you want to run for.



#Functions. 
def gamma(vx, vy):
    return((1 - ((vx/c)**2.0 + (vy/c)**2.0))**(-0.5))

def momentum(atom1_vx, atom1_vy, atom2_vx, atom2_vy):
    atom1_gamma = gamma(atom1_vx, atom1_vy)
    atom2_gamma = gamma(atom2_vx, atom2_vy)
    atom1_px = mass_kg*atom1_gamma*atom1_vx
    atom1_py = mass_kg*atom1_gamma*atom1_vy
    atom2_px = mass_kg*atom2_gamma*atom2_vx
    atom2_py = mass_kg*atom2_gamma*atom2_vy
    return(atom1_px + atom2_px, atom1_py + atom2_py)

def lj_potential(x1, y1, x2, y2):
    r = ((((x2 - x1)**2.0) + ((y2 - y1)**2.0))**0.5)
    if r <= r_c:
        potential = (4*eps*((((sigma**12.0)/(r**12.0))) - (((sigma**6.0)/(r**6.0))))) #didn't think I needed a minus sign here, but apparently I was wrong. 
    else:
        potential = 0
    return(potential)    

def total_energy(x1, y1, x2, y2, vx1, vy1, vx2, vy2):
    potential = lj_potential(x1, y1, x2, y2)
    
    kinetic1 = rest_mass_energy*(gamma(vx1, vy1) - 1)
    kinetic2 = rest_mass_energy*(gamma(vx2, vy2) - 1)
    return((potential + kinetic1 + kinetic2)*J_to_GeV)


def total_energy_ret_interacting(x1, y1, x2_ret, y2_ret, vx1, vy1):
    potential1 = (lj_potential(x1, y1, x2_ret, y2_ret))
    kinetic1 = rest_mass_energy*(gamma(vx1, vy1) - 1)
    return((potential1 + kinetic1)*J_to_GeV)

def total_energy_ret_non_interacting(vx1, vy1):
    kinetic1 = rest_mass_energy*(gamma(vx1, vy1) - 1)
    return(kinetic1*J_to_GeV)



def lj_force(x1, y1, x2, y2):
    r = ((((x2 - x1)**2.0) + ((y2 - y1)**2.0))**0.5)
    if r <= r_c:
        force = -(24*eps*((2*((sigma**12.0)/(r**13.0))) - (((sigma**6.0)/(r**7.0))))) #didn't think I needed a minus sign here, but apparently I was wrong. 
        fx = force*(((x2 - x1)/r))
        fy = force*(((y2 - y1)/r))
    else:
        fx = 0
        fy = 0
    return(fx, fy)

def update_position(x, y, vx, vy):
    new_x = (x + dt*vx)
    new_y = (y + dt*vy)
    return(new_x, new_y)

def update_velocity(vx, vy, fx, fy):
    old_gamma = gamma(vx, vy)
    big_gamma_x = ((old_gamma*vx) + (dt/(2*mass_kg))*fx)
    big_gamma_y = ((old_gamma*vy) + (dt/(2*mass_kg))*fy)
    big_gamma_dot = (big_gamma_x**2.0) + (big_gamma_y**2.0)
    new_vx = big_gamma_x/((1 + (big_gamma_dot/c_2))**0.5)
    new_vy = big_gamma_y/((1 + (big_gamma_dot/c_2))**0.5)
    return(new_vx, new_vy)


def cla_leapfrog(x1, y1, x2, y2, vx1, vy1, vx2, vy2, fx1, fy1, fx2, fy2):
    #advance velocity a half step.
    new_vx1, new_vy1 = update_velocity(vx1, vy1, fx1, fy1)
    new_vx2, new_vy2 = update_velocity(vx2, vy2, fx2, fy2)

    #advance position a whole step:
    new_x1, new_y1 = update_position(x1, y1, new_vx1, new_vy1)
    new_x2, new_y2 = update_position(x2, y2, new_vx2, new_vy2)

    #Calculate the forces from the new position. This is where you will place the back scan in the retarded potential case. 
    new_fx1, new_fy1 = lj_force(new_x1, new_y1, new_x2, new_y2)
    new_fx2, new_fy2 = -new_fx1, -new_fy1

    #calculate the new velocity based on this new force.
    new_vx1, new_vy1 = update_velocity(new_vx1, new_vy1, new_fx1, new_fy1)
    new_vx2, new_vy2 = update_velocity(new_vx2, new_vy2, new_fx2, new_fy2)

    return(new_x1, new_y1, new_x2, new_y2, new_vx1, new_vy1, new_vx2, new_vy2, new_fx1, new_fy1, new_fx2, new_fy2)

def interval(x1, y1, t1, x2, y2, t2, sim_dt): #calculates the spacetime interval. 
    delta_t = c*(t1 - t2)*sim_dt
    delta_x = x2 - x1
    delta_y = y2 - y1
    return((delta_t**2.0) - (delta_x**2.0) - (delta_y**2.0))

def interval_tolerance(t1, t2, sim_dt): #how close to zero the spacetime interval should be to be considered basically zero, since we have discrete timesteps/positions.
    return((8**0.5)*c_2*sim_dt*sim_dt*(abs(t2 - t1)))
    #return(2*c_2*(t1 - t2)*sim_dt*sim_dt + c_2*(sim_dt*sim_dt)) #if you use +/- timestep tolerance in the interval tolerance.
    #return(c_2*(t1 - t2)*sim_dt*sim_dt + c_2*(sim_dt*sim_dt*0.25)) #if you use +/- timestep/2 in the interval tolerance.



def backscan(current_x, current_y, current_t, x_traj, y_traj, t_traj, sim_dt): #trying out pontus' sign change idea.
    result = False
    #interaction as the very first index. we resize the arrays so that is always true. 
    # Test if there will be any solution at all. 
    idx = numpy.size(x_traj) - 1
    v1 = interval(current_x, current_y, current_t, x_traj[0], y_traj[0], t_traj[0], sim_dt)
    v2 = interval(current_x, current_y, current_t, x_traj[idx], y_traj[idx], t_traj[idx], sim_dt)
    if (v1*v2 > 0.0):
        if abs(interval(current_x, current_y, current_t, x_traj[0], y_traj[0], t_traj[0], sim_dt)) < abs(interval_tolerance(current_t, t_traj[0], sim_dt)):
            return 0
        else:
            return False

    for point in range(0, numpy.size(x_traj)): #scanning forwards through the trajectory to find the zero interval. 
        #calculating the interval and seeing if it is within tolerance. 
        if interval(current_x, current_y, current_t, x_traj[point], y_traj[point], t_traj[point], sim_dt) <= 0:
            result = point-1
            break 
    return(result) 

def ret_leapfrog(t1, x1, y1, t2, x2, y2, vx1, vy1, vx2, vy2, fx1, fy1, fx2, fy2, x1_trajectory, y1_trajectory, t1_trajectory, x2_trajectory, y2_trajectory, t2_trajectory, sim_dt): 
    #advance velocity a half step.
    new_vx1, new_vy1 = update_velocity(vx1, vy1, fx1, fy1)
    new_vx2, new_vy2 = update_velocity(vx2, vy2, fx2, fy2)

    #advance position a whole step:
    new_x1, new_y1 = update_position(x1, y1, new_vx1, new_vy1)
    new_x2, new_y2 = update_position(x2, y2, new_vx2, new_vy2)

    #Calculate the forces from the new position. This is where you will place the back scan, which returns the index on the trajectory that we should use.  
    index_for_atom_1 = backscan(new_x1, new_y1, t1, x2_trajectory, y2_trajectory, t2_trajectory, sim_dt)
    index_for_atom_2 = backscan(new_x2, new_y2, t2, x1_trajectory, y1_trajectory, t1_trajectory, sim_dt)


    if index_for_atom_1 != False: #We found a lightlike separation that doesnt travel beyond r_c, so calculate force and truncate the array at the separation point.  
        new_fx1, new_fy1 = lj_force(new_x1, new_y1, x2_trajectory[index_for_atom_1], y2_trajectory[index_for_atom_1])
        x2_trajectory = x2_trajectory[index_for_atom_1 - 1:]
        y2_trajectory = y2_trajectory[index_for_atom_1 - 1:]
        t2_trajectory = t2_trajectory[index_for_atom_1 - 1:]
        x2_trajectory = numpy.append(x2_trajectory, new_x2) #just adding the new point to the trajectory array. 
        y2_trajectory = numpy.append(y2_trajectory, new_y2)
        t2_trajectory = numpy.append(t2_trajectory, t2)
        energy_1 = total_energy_ret_interacting(new_x1, new_y1, x2_trajectory[index_for_atom_1], y2_trajectory[index_for_atom_1], new_vx1, new_vy1)
    else: 
        new_fx1 = 0
        new_fy1 = 0
        x2_trajectory = numpy.append(x2_trajectory, new_x2) #just adding the new point to the trajectory array. 
        y2_trajectory = numpy.append(y2_trajectory, new_y2)
        t2_trajectory = numpy.append(t2_trajectory, t2)
        energy_1 = total_energy_ret_non_interacting(new_vx1, new_vy1)



    if index_for_atom_2 != False: #We found a lightlike separation that doesnt travel beyond r_c, so calculate force and truncate the array at the separation point.  
        new_fx2, new_fy2 = lj_force(new_x2, new_y2, x1_trajectory[index_for_atom_2], y1_trajectory[index_for_atom_2])
        x1_trajectory = x1_trajectory[index_for_atom_2 - 1:]
        y1_trajectory = y1_trajectory[index_for_atom_2 - 1:]
        t1_trajectory = t1_trajectory[index_for_atom_2 - 1:]
        x1_trajectory = numpy.append(x1_trajectory, new_x1) #just adding the new point to the trajectory array. 
        y1_trajectory = numpy.append(y1_trajectory, new_y1)
        t1_trajectory = numpy.append(t1_trajectory, t1)
        energy_2 = total_energy_ret_interacting(new_x2, new_y2, x1_trajectory[index_for_atom_2], y1_trajectory[index_for_atom_2], new_vx2, new_vy2)
    else:
        new_fx2 = 0
        new_fy2 = 0
        x1_trajectory = numpy.append(x1_trajectory, new_x1) #just adding the new point to the trajectory array. 
        y1_trajectory = numpy.append(y1_trajectory, new_y1)
        t1_trajectory = numpy.append(t1_trajectory, t1)
        energy_2 = total_energy_ret_non_interacting(new_vx2, new_vy2) 

    #calculate the new velocity based on these new forces.
    new_vx1, new_vy1 = update_velocity(new_vx1, new_vy1, new_fx1, new_fy1)
    new_vx2, new_vy2 = update_velocity(new_vx2, new_vy2, new_fx2, new_fy2)
    new_t1 = t1 + 1
    new_t2 = t2 + 1

   
    return(new_t1, new_x1, new_y1, new_t2, new_x2, new_y2, new_vx1, new_vy1, new_vx2, new_vy2, new_fx1, new_fy1, new_fx2, new_fy2, x1_trajectory, y1_trajectory, t1_trajectory, x2_trajectory, y2_trajectory, t2_trajectory, energy_1, energy_2)


def opening_angle(y, x):
    angle = arctan(y/x)*(180/pi)

    if angle < 0:
        result = 180 + angle
    else:
        result = angle

    return(result)

def kinetic(vx, vy):
    gam =  gamma(vx, vy)
    kin = rest_mass_energy*(gam - 1)
    return(kin)

def Theta(T):
    return(rest_mass_energy/(boltz*T))

def temp_from_theta(THETA):
    return(rest_mass_energy/(boltz*THETA))

def gamma_b(beta):
    return(1/((1-(beta**2.0))**0.5))

def Maxwell_Juttner_Beta(beta_x, beta_y, theta):
    gam = gamma(beta_x*c, beta_y*c)
    return(real((theta/kn(2, theta))*(gam**5.0)*((beta_x**2.0) + (beta_y**2.0))*exp(-theta*gam)))

cla_e_0 = numpy.array([])
ret_e_0 = numpy.array([])
cla_e_f = numpy.array([])
ret_e_f = numpy.array([])
cla_de = numpy.array([])
ret_de = numpy.array([])


cla_theta_b_theta = numpy.array([])
cla_theta_b_b = numpy.array([])
ret_theta_b_theta = numpy.array([])
ret_theta_b_b = numpy.array([])

ret_vx_final = numpy.array([])
ret_vy_final = numpy.array([])
vx_ratio = numpy.array([]) 
vy_ratio = numpy.array([])

kin_ratio = numpy.array([])

cla_vx_final = 0
cla_vy_final = 0


ret_temp_cross = numpy.array([]) 
cla_temp_cross = numpy.array([]) 

ret_cross_section = numpy.array([])
cla_cross_section = numpy.array([])

speed1_cross = numpy.array([])
speed2_cross = numpy.array([])

ret_sin = numpy.array([])
cla_sin = numpy.array([])

ret_b_times_db_dtheta_times_sin = numpy.array([])
cla_b_times_db_dtheta_times_sin = numpy.array([])

#Single run settings. Useless for the multi-run case, but I build this on top of the multi run and it's a pain to remove all of it's features. 
b_init = 0.01*r_c #vertical offset (impact parameter)
dist_init = 0.5*r_c #horizontal offset.
sep_init = ((b_init**2.0) + (dist_init**2.0))**0.5
vx1_init = 0.00001*c #initial horizontal speed of the leftmost atom
vx2_init = -0.00001*c #initial horizontal speed of the rightmost atom. 
vy1_init = 0.0
vy2_init = 0.0
g_init = "0_00002c"
b_init_text = "0_01r_c"

ret_temp_transport_cross = numpy.array([])
cla_temp_transport_cross = numpy.array([])
ret_transport_cross = numpy.array([])
cla_transport_cross = numpy.array([])

#Start of the cross section scan.
n_timesteps = 40000000000000000 #Number of timesteps that you want to run for, an upper bound to stop things from running forever. 
equal_veloc = numpy.array([])
run_counter = 1
n_b = 150
n_v = 21
n_runs = 15
dv = c/n_v
for i in range(1, n_v):
    ret_cross_section = numpy.append(ret_cross_section, 0)
    cla_cross_section = numpy.append(cla_cross_section, 0)
    ret_transport_cross = numpy.append(ret_transport_cross, 0)
    cla_transport_cross = numpy.append(cla_transport_cross, 0)
    equal_veloc = numpy.append(equal_veloc, dv*i)
for speed_1 in range(1, n_v):
    for speed_2 in range(1, n_v):
        dt = 1e-21
        if (speed_1*dv >= 0.8*c) or (speed_2*dv >= 0.8*c):
            dt = 1e-22
        cla_e_0 = numpy.array([])
        ret_e_0 = numpy.array([])
        cla_e_f = numpy.array([])
        ret_e_f = numpy.array([])
        cla_de = numpy.array([])
        ret_de = numpy.array([])


        cla_theta_b_theta = numpy.array([])
        cla_theta_b_b = numpy.array([])
        ret_theta_b_theta = numpy.array([])
        ret_theta_b_b = numpy.array([])

        ret_vx_final = numpy.array([])
        ret_vy_final = numpy.array([])
        vx_ratio = numpy.array([]) 
        vy_ratio = numpy.array([])

        kin_ratio = numpy.array([])

        cla_vx_final = 0
        cla_vy_final = 0
        for impact in range(0, n_runs): #retative velocity dependent timestep to speed up the computation. 
            b_init = (0.001*r_c) + ((r_c/n_b)*impact)
            print(b_init)
            cla_vx_final = 0
            cla_vy_final = 0
            ek_init = 0#useless here, so just set to zero rather than risk screwing somthing up later on
            cla_e_0 = numpy.append(cla_e_0, ek_init)
            cla_ek_final = 0 
            """   
            if b_init >= (5*(10**(-10))):
                break
            """
            #Classical Data

            cla_temp_x1 = 0
            cla_temp_y1 = b_init
            
            cla_temp_x2 = dist_init
            cla_temp_y2 = 0

            cla_temp_vx1 = speed_1*dv
            cla_temp_vy1 = 0
            
            cla_temp_vx2 = -speed_2*dv
            cla_temp_vy2 = 0

            cla_temp_fx1 = 0
            cla_temp_fy1 = 0
            
            cla_temp_fx2 = 0
            cla_temp_fy2 = 0

            cla_r_temp = 0

            cla_px_temp = 0
            cla_py_temp = 0



            #The classical simulation. 
            t = 0
            print("Classical Run: ")
            print("") 
            while t < n_timesteps:
                cla_temp_x1, cla_temp_y1, cla_temp_x2, cla_temp_y2, cla_temp_vx1, cla_temp_vy1, cla_temp_vx2, cla_temp_vy2, cla_temp_fx1, cla_temp_fy1, cla_temp_fx2, cla_temp_fy2 = cla_leapfrog(cla_temp_x1, cla_temp_y1, cla_temp_x2, cla_temp_y2, cla_temp_vx1, cla_temp_vy1, cla_temp_vx2, cla_temp_vy2, cla_temp_fx1, cla_temp_fy1, cla_temp_fx2, cla_temp_fy2)
                
                if t%10000 == 0: #sample rate for classical
                    print("Classical")
                    print(t)
                    print("dt = " + str(dt))
                    print(cla_r_temp)
                    print("Classical")
                    print("working on the " + str((impact+1)) + "th case out of " + str(n_runs))
                    print("working on speed value " + str(run_counter) + " out of " + str((n_v-1)*(n_v-1)))
                    print("impact parameter: " + str(b_init))
                    print("-_-_-_-_-_-_-_-_")
                    print("")
                    if cla_temp_x1 != 0:
                        print(opening_angle((cla_temp_y1 - cla_temp_y2), (cla_temp_x1 - cla_temp_x2)))
                    cla_r_temp = real((((cla_temp_x2 - cla_temp_x1)**2.0) + ((cla_temp_y2 - cla_temp_y1)**2.0))**0.5)

                    cla_px_temp, cla_py_temp, = momentum(cla_temp_vx1, cla_temp_vy1, cla_temp_vx2, cla_temp_vy2)

                if cla_r_temp > 0.6*r_c:
                    print(cla_r_temp)
                    cla_theta_b_theta = numpy.append(cla_theta_b_theta, (opening_angle((cla_temp_vy1), (cla_temp_vx1))))
                    cla_theta_b_b = numpy.append(cla_theta_b_b, b_init)
                    cla_vx_final = cla_temp_vx1
                    cla_vy_final = cla_temp_vy1
                    cla_ek_final = kinetic(cla_vx_final, cla_vy_final)
                    cla_e_f = numpy.append(cla_e_f, cla_ek_final)
                    cla_de = numpy.append(cla_de, (cla_ek_final - ek_init))


                    #dump the angles. 
                    print("Out of simulation range.")
                    break
            
                
                t += 1

            #The retativistic simulation. 


            #retativistic Arrays and data

            ret_temp_t1 = 0
            ret_temp_x1 = 0
            ret_temp_y1 = b_init

            ret_temp_t2 = 0 
            ret_temp_x2 = dist_init
            ret_temp_y2 = 0

            ret_temp_vx1 = speed_1*dv
            ret_temp_vy1 = 0
            
            ret_temp_vx2 = -speed_2*dv
            ret_temp_vy2 = 0

            ret_temp_fx1 = 0
            ret_temp_fy1 = 0
            
            ret_temp_fx2 = 0
            ret_temp_fy2 = 0

            ret_r_temp = 0

            ret_px_temp = 0
            ret_py_temp = 0
            
            x1_traj = numpy.array([0])
            y1_traj = numpy.array([b_init])
            t1_traj = numpy.array([0])

            x2_traj = numpy.array([dist_init])
            y2_traj = numpy.array([0])
            t2_traj = numpy.array([0])
            ret_e_0 = numpy.append(ret_e_0, kinetic(v_init, 0))


            t = 0
            print("")
            print("_____________________________________")
            print("retativistic Run:")
            print("")
            while t < n_timesteps:
                ret_temp_t1, ret_temp_x1, ret_temp_y1, ret_temp_t2, ret_temp_x2, ret_temp_y2, ret_temp_vx1, ret_temp_vy1, ret_temp_vx2, ret_temp_vy2, ret_temp_fx1, ret_temp_fy1, ret_temp_fx2, ret_temp_fy2, x1_traj, y1_traj, t1_traj, x2_traj, y2_traj, t2_traj, nrg1_temp, nrg2_temp  = ret_leapfrog(ret_temp_t1, ret_temp_x1, ret_temp_y1, ret_temp_t2, ret_temp_x2, ret_temp_y2, ret_temp_vx1, ret_temp_vy1, ret_temp_vx2, ret_temp_vy2, ret_temp_fx1, ret_temp_fy1, ret_temp_fx2, ret_temp_fy2, x1_traj, y1_traj, t1_traj, x2_traj, y2_traj, t2_traj, dt)
                
                if t%10000 == 0: #sample rate for retarded potential case. 
                    print(t)
                    
                    print(ret_r_temp)
                    print("Retarded:")
                    print("dt = " + str(dt))
                    print("working on the " + str((impact+1)) + "th case out of " + str(n_runs))
                    print("working on speed value " + str(run_counter) + " out of " + str((n_v-1)*(n_v-1)))
                    print("impact parameter: " + str(b_init))
                    print("-_-_-_-_-_-_-_-_")
                    print("")
                    ret_r_temp = real((((ret_temp_x2 - ret_temp_x1)**2.0) + ((ret_temp_y2 - ret_temp_y1)**2.0))**0.5)
                    if ret_temp_x1 != 0.0:
                        print(opening_angle((ret_temp_y1 - ret_temp_y2), (ret_temp_x1 - ret_temp_x2)))

                    ret_px_temp, ret_py_temp = momentum(ret_temp_vx1, ret_temp_vy1, ret_temp_vx2, ret_temp_vy2)

                if ret_r_temp > 0.6*r_c:
                    #this is where you dump the angle. 
                    ret_theta_b_theta = numpy.append(ret_theta_b_theta, opening_angle((ret_temp_vy1), (ret_temp_vx1)))
                    ret_theta_b_b = numpy.append(ret_theta_b_b, b_init)
                    ret_vx_final = numpy.append(ret_vx_final, ret_temp_vx1)
                    ret_vy_final = numpy.append(ret_vy_final, ret_temp_vy1)
                    vx_ratio = numpy.append(vx_ratio, ret_temp_vx1/cla_vx_final)
                    vy_ratio = numpy.append(vy_ratio, ret_temp_vy1/cla_vy_final)
                    kin_ratio = numpy.append(kin_ratio, (kinetic(ret_temp_vx1, ret_temp_vy1) - ek_init)/ek_init)
                    ret_e_f = numpy.append(ret_e_f, kinetic(ret_temp_vx1, ret_temp_vy1))
                    ret_de = numpy.append(ret_de, (kinetic(ret_temp_vx1, ret_temp_vy1) - kinetic(v_init, 0)))
                    


                    print(ret_r_temp)
                    print("Out of simulation range.")
                    break
            
                t += 1
        #we now have theta(b) for a specific v1 v2.
        run_counter += 1

        #theta goes from big to small. 
        #b goes from small to big. 
        #therefore for b as a function of theta, we need to reverse both lists. 
        ret_cross_b = numpy.flip(ret_theta_b_b) #temporary reversed lists for the cross section calculation. 
        ret_cross_theta = numpy.flip(ret_theta_b_theta)
        cla_cross_b = numpy.flip(cla_theta_b_b)
        cla_cross_theta = numpy.flip(cla_theta_b_theta)

        for point in range(0, len(ret_cross_theta)):
            ret_cross_theta[point] = ret_cross_theta[point]*(pi/180)
            cla_cross_theta[point] = cla_cross_theta[point]*(pi/180)
        
        ret_cross_cos = numpy.cos(ret_cross_theta)
        cla_cross_cos = numpy.cos(cla_cross_theta)

        for i in range(0, len(ret_cross_cos)):
            ret_cross_cos[i] = 1.0 - ret_cross_cos[i]
            cla_cross_cos[i] = 1.0 - cla_cross_cos[i]
        
        print("1 - cos")
        print(cla_cross_cos)


            
        ret_db_dtheta = abs(numpy.gradient(ret_cross_b, ret_cross_theta))
        cla_db_dtheta = abs(numpy.gradient(cla_cross_b, cla_cross_theta))
        print(ret_db_dtheta)
        ret_b_times_db_dtheta = (numpy.multiply(ret_db_dtheta, ret_cross_b))
        cla_b_times_db_dtheta = (numpy.multiply(cla_db_dtheta, cla_cross_b))

        ret_transport_integrand = numpy.multiply(ret_b_times_db_dtheta, ret_cross_cos)
        cla_transport_integrand = numpy.multiply(cla_b_times_db_dtheta, cla_cross_cos)
        print(cla_transport_integrand)
        print(ret_b_times_db_dtheta)
        print(ret_cross_theta)

        ret_temp_transport_cross = numpy.append(ret_temp_transport_cross, numpy.trapz(ret_transport_integrand, ret_cross_theta))
        cla_temp_transport_cross = numpy.append(cla_temp_transport_cross, numpy.trapz(cla_transport_integrand, cla_cross_theta))
        print("ahhhhh")
        print(cla_temp_transport_cross)
        print(ret_temp_transport_cross)
        print("ahhhhh")
        ret_temp_cross = numpy.append(ret_temp_cross, 2*pi*numpy.trapz(ret_b_times_db_dtheta, ret_cross_theta))
        cla_temp_cross = numpy.append(cla_temp_cross, 2*pi*numpy.trapz(cla_b_times_db_dtheta, cla_cross_theta))

    ret_transport_cross = numpy.vstack((ret_transport_cross, ret_temp_transport_cross))
    cla_transport_cross = numpy.vstack((cla_transport_cross, cla_temp_transport_cross))
    cla_cross_section = numpy.vstack((cla_cross_section, cla_temp_cross))
    ret_cross_section = numpy.vstack((ret_cross_section, ret_temp_cross))
    cla_cross_section = numpy.vstack((cla_cross_section, cla_temp_cross))
    print("For v1: " + str(speed_1*dv) + " and v2: " + str(speed_2*dv) + " we have ret cross section " + str(ret_temp_cross) + " cla cross " + str(cla_temp_cross))
    ret_temp_cross = numpy.array([]) 
    cla_temp_cross = numpy.array([])
    ret_temp_transport_cross = numpy.array([])
    cla_temp_transport_cross = numpy.array([]) 

ret_cross_section = numpy.delete(ret_cross_section, 0, 0) #getting rid of the empty row that I needed in order to do the stacking. 
cla_cross_section = numpy.delete(cla_cross_section, 0, 0)
ret_transport_cross = numpy.delete(ret_transport_cross, 0, 0)
cla_transport_cross = numpy.delete(cla_transport_cross, 0, 0)

print(ret_transport_cross)
print(cla_transport_cross)
print("-_-_-_-_-_--------------------_-_-_-_-_-")
print("-_-_-_-_-_--------------------_-_-_-_-_-")
print("-_-_-_-_-_--------------------_-_-_-_-_-")
print("classical_transport_cross_section_data")
for i in range(0, len(equal_veloc)):
    print("[")
    for point in cla_transport_cross[i]:
        print(str(point), end = ", ")
    print("], ")

print("-_-_-_-_-_--------------------_-_-_-_-_-")
print("-_-_-_-_-_--------------------_-_-_-_-_-")
print("-_-_-_-_-_--------------------_-_-_-_-_-")
print("retarded_transport_cross_section_data")
for i in range(0, len(equal_veloc)):
    print("[")
    for point in ret_transport_cross[i]:
        print(str(point), end = ", ")
    print("], ")


print("-_-_-_-_-_--------------------_-_-_-_-_-")
print("-_-_-_-_-_--------------------_-_-_-_-_-")
print("-_-_-_-_-_--------------------_-_-_-_-_-")
print("classical_cross_section_data")
for i in range(0, len(equal_veloc)):
    print("[")
    for point in cla_cross_section[i]:
        print(str(point), end = ", ")
    print("], ")


print("")
print("")
print("")
print("-_-_-_-_-_--------------------_-_-_-_-_-")
print("-_-_-_-_-_--------------------_-_-_-_-_-")
print("-_-_-_-_-_--------------------_-_-_-_-_-")

print("-_-_-_-_-_--------------------_-_-_-_-_-")
print("-_-_-_-_-_--------------------_-_-_-_-_-")
print("-_-_-_-_-_--------------------_-_-_-_-_-")
print("retarded_cross_section_data")
for i in range(0, len(equal_veloc)):
    print("[")
    for point in ret_cross_section[i]:
        print(str(point), end = ", ")
    print("], ")
print("")
print("")
print("")
print("-_-_-_-_-_--------------------_-_-_-_-_-")
print("-_-_-_-_-_--------------------_-_-_-_-_-")
print("-_-_-_-_-_--------------------_-_-_-_-_-")
print("-_-_-_-_-_--------------------_-_-_-_-_-")
print("-_-_-_-_-_--------------------_-_-_-_-_-")
print("-_-_-_-_-_--------------------_-_-_-_-_-")
print("speed_data")
for point in equal_veloc:
    print(str(point), end = ", ")
print("")
print("")
print("")
print("-_-_-_-_-_--------------------_-_-_-_-_-")
print("-_-_-_-_-_--------------------_-_-_-_-_-")
print("-_-_-_-_-_--------------------_-_-_-_-_-")


cla_angstrom_b = numpy.array([])
ret_angstrom_b = numpy.array([])
for par in ret_theta_b_b:
    cla_angstrom_b = numpy.append(cla_angstrom_b, par*(10**10.0))
    ret_angstrom_b = numpy.append(ret_angstrom_b, par*(10**10.0))



n_v_bins = 8
t_from_sim = [1563869692390.2937, 6201989860014.763, 14434813949070.03, 25921101025334.86, 41494145699442.36, 65373429679167.695, 96798914014484.52, 150657400984939.75, 262814104941084.75]

temperature = 1563869692390.2937

juttner_1 = numpy.array([])
juttner_2 = numpy.array([])

cla_integrated = numpy.array([])
ret_integrated = numpy.array([])

cla_integrated_over_v2 = numpy.array([])
ret_integrated_over_v2 = numpy.array([])

temp_cla_integrand = numpy.array([])
temp_ret_integrand = numpy.array([])
small_juttner_distribution = numpy.array([])
equal_beta = numpy.array([])
for bet in range(0, len(equal_veloc)):
    equal_beta = numpy.append(equal_beta, equal_veloc[bet]/c)

for v in range(0, len(equal_veloc)):
    small_juttner_distribution = numpy.append(small_juttner_distribution,  Maxwell_Juttner_Beta(equal_beta[v], 0,  Theta(temperature)))



distribution_normalisation = numpy.trapz(small_juttner_distribution, equal_beta)

for v1 in range(0, len(equal_veloc)):
    for v2 in range(0, len(equal_veloc)):
        temp_cla_integrand = numpy.append(temp_cla_integrand, Maxwell_Juttner_Beta(equal_beta[v2], 0,  Theta(temperature))*(1/distribution_normalisation)*cla_cross_section[v1, v2])
        temp_ret_integrand = numpy.append(temp_ret_integrand, Maxwell_Juttner_Beta(equal_beta[v2], 0, Theta(temperature))*(1/distribution_normalisation)*ret_cross_section[v1, v2])
    cla_integrated_over_v2 = numpy.append(cla_integrated_over_v2, numpy.trapz(temp_cla_integrand, equal_beta))
    ret_integrated_over_v2 = numpy.append(ret_integrated_over_v2, numpy.trapz(temp_ret_integrand, equal_beta))
    temp_cla_integrand = numpy.array([])
    temp_ret_integrand = numpy.array([])

cla_v1_integrand = numpy.array([])
ret_v1_integrand = numpy.array([])
for v1 in range(0, len(equal_veloc)):
    cla_v1_integrand = numpy.append(cla_v1_integrand, Maxwell_Juttner_Beta(equal_beta[v1], 0,  Theta(temperature))*(1/distribution_normalisation)*cla_integrated_over_v2[v1])
    ret_v1_integrand = numpy.append(ret_v1_integrand, Maxwell_Juttner_Beta(equal_beta[v1], 0,  Theta(temperature))*(1/distribution_normalisation)*ret_integrated_over_v2[v1])


cla_integrated = numpy.trapz(cla_v1_integrand, equal_beta)
ret_integrated = numpy.trapz(ret_v1_integrand, equal_beta)
    

print("")
print("")
print("")      
print("-_-_-_-_-_--------------------_-_-_-_-_-")    
print("-_-_-_-_-_--------------------_-_-_-_-_-")
print("ret cross section = " + str(ret_integrated))
print("cla cross section = " + str(cla_integrated))
print("-_-_-_-_-_--------------------_-_-_-_-_-")
print("-_-_-_-_-_--------------------_-_-_-_-_-")
print("")
print("")
print("")
print("")
print("")
print("")
print("___________---------------______________--------------______________-------------______________")
print("impact parameter:")
for point in ret_angstrom_b:
    print(str(point), end = ", ")

print("")
print("___________---------------______________--------------______________-------------______________")
print("classical angle:")
for point in cla_theta_b_theta:
    print(str(point), end = ", ")

print("")
print("___________---------------______________--------------______________-------------______________")
print("retarded angle:")
for point in ret_theta_b_theta:
    print(str(point), end = ", ")

print("")
print("___________---------------______________--------------______________-------------______________")
print("kinetic ratio:")
for point in kin_ratio:
    print(str(point), end = ", ")

print("")
print("___________---------------______________--------------______________-------------______________")
print("cla initial energy:")
for point in cla_e_0:
    print(str(point), end = ", ")

print("")
print("___________---------------______________--------------______________-------------______________")
print("cla final energy:")
for point in cla_e_f:
    print(str(point), end = ", ")

print("")
print("___________---------------______________--------------______________-------------______________")
print("cla difference energy:")
for point in cla_de:
    print(str(point), end = ", ")


print("")
print("___________---------------______________--------------______________-------------______________")
print("ret initial energy:")
for point in ret_e_0:
    print(str(point), end = ", ")

print("")
print("___________---------------______________--------------______________-------------______________")
print("ret final energy:")
for point in ret_e_f:
    print(str(point), end = ", ")

print("")
print("")
print("")
print("_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-")
print("")
print("___________---------------______________--------------______________-------------______________")
print("ret cross:")
for point in ret_cross_section:
    print(str(point), end = ", ")

print("")
print("___________---------------______________--------------______________-------------______________")
print("cla cross:")
for point in cla_cross_section:
    print(str(point), end = ", ")

print("")
print("___________---------------______________--------------______________-------------______________")
print("speed 1:")
for point in speed1_cross:
    print(str(point), end = ", ")


print("")
print("___________---------------______________--------------______________-------------______________")
print("speed 2:")
for point in speed2_cross:
    print(str(point), end = ", ")






print("")

#Plotting zone. -----------------------------------------------------------

pyplot.plot(cla_angstrom_b, cla_theta_b_theta, "s-b", label = "Classical Potential")
pyplot.plot(ret_angstrom_b, ret_theta_b_theta, "^-r", label = "Retarded Potential")
pyplot.xlabel("Impact Parameter ($\AA$)")
pyplot.ylabel("Deflection Angle (Degrees)")
pyplot.legend()
pyplot.figure()
pyplot.plot(ret_angstrom_b, kin_ratio, "s-b", label = "Y Component")
pyplot.yscale("log")
pyplot.xlabel("Impact Parameter ($\AA$)")
pyplot.ylabel("Velocity Ratio (no units)")
pyplot.legend()

pyplot.show()
