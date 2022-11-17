from cmath import pi
from unittest import result
import numpy
from matplotlib import pyplot, rc, rcParams
from numpy.lib.type_check import real
from numpy import arcsin, arctan

rc("xtick", labelsize = 13)
rc("ytick", labelsize = 13)
rcParams.update({"font.size":13})



#The relativistic runs always look like they have frozen on the first few timesteps, but as soon as the atoms start interacting, the simulation starts chopping down
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


# Variables for the LJ Potential. 
epsilon = 0.01034 #ev
eps = epsilon*ev_to_J #in Joules. I have a small brain that still thinks in SI. 
sigma = 3.4*(10**(-10.0)) #m
r_c = sigma*2.25
r_c_2 = r_c**2.0

#This round of settings is here just so that the function definitions don't crash.
#initial universal settings. <--- For Single Runs(Not Scancs for cross section) This is where you set the parameters for the collision. 
b_init = 0.01*r_c #vertical offset (impact parameter)
dist_init = 1.0*r_c #horizontal offset.
sep_init = ((b_init**2.0) + (dist_init**2.0))**0.5
vx1_init = 0.01*c #initial horizontal speed of the leftmost atom
vx2_init = -0.01*c #initial horizontal speed of the rightmost atom. 
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
        potential = -(4*eps*((((sigma**12.0)/(r**12.0))) - (((sigma**6.0)/(r**6.0))))) #didn't think I needed a minus sign here, but apparently I was wrong. 
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
    #return(c_2*sim_dt*sim_dt*(abs(t2 - t1)))
    #return(2*c_2*(t1 - t2)*sim_dt*sim_dt + c_2*(sim_dt*sim_dt)) #if you use +/- timestep tolerance in the interval tolerance.
    #return(c_2*(t1 - t2)*sim_dt*sim_dt + c_2*(sim_dt*sim_dt*0.25)) #if you use +/- timestep/2 in the interval tolerance.

"""
def backscan(current_x, current_y, current_t, x_traj, y_traj, t_traj, sim_dt): #trajectory arrays have the previous timestep as the last element, and the index of the most recent
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
        if abs(interval(current_x, current_y, current_t, x_traj[point], y_traj[point], t_traj[point], sim_dt)) < abs(interval_tolerance(current_t, t_traj[point], sim_dt)):
            result = point
            break
    return(result)
"""

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

def rel_leapfrog(t1, x1, y1, t2, x2, y2, vx1, vy1, vx2, vy2, fx1, fy1, fx2, fy2, x1_trajectory, y1_trajectory, t1_trajectory, x2_trajectory, y2_trajectory, t2_trajectory, sim_dt): 
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
    """
    if 180 - result <= 0.000001: #even if the particles just barely interact, the angle goes negative by the attractive bit of the LJ, then get's falsely set to 180. Just set to 0 if the difference is small enough.
        result = 0 
    """
    return(result)

def kinetic(vx, vy):
    gam =  gamma(vx, vy)
    kin = rest_mass_energy*(gam - 1)
    return(kin)
cla_e_0 = numpy.array([])
ret_e_0 = numpy.array([])
cla_e_f = numpy.array([])
ret_e_f = numpy.array([])
cla_de = numpy.array([])
ret_de = numpy.array([])


cla_theta_b_theta = numpy.array([])
cla_theta_b_b = numpy.array([])
rel_theta_b_theta = numpy.array([])
rel_theta_b_b = numpy.array([])

rel_vx_final = numpy.array([])
rel_vy_final = numpy.array([])
vx_ratio = numpy.array([]) 
vy_ratio = numpy.array([])

kin_ratio = numpy.array([])

cla_vx_final = 0
cla_vy_final = 0

#initial universal settings for single runs. <--- This is where you set the parameters for the collision !!!if you are not doing a cross section scan!!!
b_init = 0.01*r_c #vertical offset (impact parameter)
dist_init = sigma #horizontal offset.
sep_init = ((b_init**2.0) + (dist_init**2.0))**0.5
vx1_init = 0.00001*c #initial horizontal speed of the leftmost atom
vx2_init = -0.00001*c #initial horizontal speed of the rightmost atom. 
vy1_init = 0.0
vy2_init = 0.0
g_init = "0_00002c"
b_init_text = "0_01r_c"

dt = 1e-21
dt = (0.001*c)/(abs(vx1_init - vx2_init))*dt  #a rough rule of thumb for how you should scale the timestep. Feel free to change.
dt = 1e-24
n_timesteps = 40000000000000 #Number of timesteps that you want to run for.

n_b = 100
n_v = 2
n_runs = 20
for speeds in range(1, n_v):
    for impact in range(0, n_runs):
        b_init = (r_c*0.0000000000000000001)+ ((r_c/n_b)*impact)
        b_init = 0.1*(10**(-10))
        v_init = 0.9*c
        dt = (1e-23)
        print(v_init)
        print(b_init)
        cla_vx_final = 0
        cla_vy_final = 0
        ek_init = kinetic(v_init, 0)
        cla_e_0 = numpy.append(cla_e_0, ek_init)
        cla_ek_final = 0    
        if b_init >= (5*(10**(-10))):
            break
        #Classical Data
        
        cla_x1 = numpy.array([])
        cla_y1 = numpy.array([])
        
        cla_x2 = numpy.array([])
        cla_y2 = numpy.array([])
        """
        cla_vx1 = numpy.array([])
        cla_vy1 = numpy.array([])
        
        cla_vx2 = numpy.array([])
        cla_vy2 = numpy.array([])

        cla_fx1 = numpy.array([])
        cla_fy1 = numpy.array([])
        
        cla_fx2 = numpy.array([])
        cla_fy2 = numpy.array([])

        cla_separation = numpy.array([])

        cla_time = numpy.array([])
        """
        cla_temp_x1 = 0
        cla_temp_y1 = b_init
        
        cla_temp_x2 = dist_init
        cla_temp_y2 = 0

        cla_temp_vx1 = v_init
        cla_temp_vy1 = 0
        
        cla_temp_vx2 = -v_init
        cla_temp_vy2 = 0

        cla_temp_fx1 = 0
        cla_temp_fy1 = 0
        
        cla_temp_fx2 = 0
        cla_temp_fy2 = 0

        cla_r_temp = 0

        cla_px_temp = 0
        cla_py_temp = 0

        #cla_px = numpy.array([])
        #cla_py = numpy.array([])

        #cla_energy = numpy.array([])

        #classical_seperation_data = numpy.array([])

        #The classical simulation. 
        t = 0

        print("Classical Run: ")
        print("") 
        cla_separation = numpy.array([])
        while t < n_timesteps:
            cla_temp_x1, cla_temp_y1, cla_temp_x2, cla_temp_y2, cla_temp_vx1, cla_temp_vy1, cla_temp_vx2, cla_temp_vy2, cla_temp_fx1, cla_temp_fy1, cla_temp_fx2, cla_temp_fy2 = cla_leapfrog(cla_temp_x1, cla_temp_y1, cla_temp_x2, cla_temp_y2, cla_temp_vx1, cla_temp_vy1, cla_temp_vx2, cla_temp_vy2, cla_temp_fx1, cla_temp_fy1, cla_temp_fx2, cla_temp_fy2)
            
            if t%100 == 0: #sample rate for classical
                print("Classical Run")
                print(t)
                print("dt " + str(dt))
                print("working on the " + str(speeds*(impact+1)) + "th case out of " + str(n_runs))
                print("impact parameter: " + str(b_init))
                print("-_-_-_-_-_-_-_-_")
                print("")
                if cla_temp_x1 != 0:
                    print(opening_angle((cla_temp_y1 - cla_temp_y2), (cla_temp_x1 - cla_temp_x2)))
                cla_r_temp = real((((cla_temp_x2 - cla_temp_x1)**2.0) + ((cla_temp_y2 - cla_temp_y1)**2.0))**0.5)

                print("seperation: " + str(cla_r_temp) + " m")
                print("x force: " + str(cla_temp_fx1) + " N")
                print("y force: " + str(cla_temp_fy1) + " N")
                print("atom 1 x velocity: " + str(cla_temp_vx1) + " m/s")
                print("atom 1 y velocity: " + str(cla_temp_vy1) + " m/s")
                print("atom 2 x velocity: " + str(cla_temp_vx2) + " m/s")
                print("atom 2 y velocity: " + str(cla_temp_vy2) + " m/s")
                print("atom 1 x position: " + str(cla_temp_x1) + " m")
                print("atom 1 y position: " + str(cla_temp_y1) + " m")
                print("atom 2 x position: " + str(cla_temp_x2) + " m")
                print("atom 2 y position: " + str(cla_temp_y2) + " m")
                print("_________")
                print("")

                cla_px_temp, cla_py_temp, = momentum(cla_temp_vx1, cla_temp_vy1, cla_temp_vx2, cla_temp_vy2)
                """
                cla_px = numpy.append(cla_px, cla_px_temp)
                cla_py = numpy.append(cla_py, cla_py_temp)
                cla_energy = numpy.append(cla_energy, total_energy(cla_temp_x1, cla_temp_y1, cla_temp_x2, cla_temp_y2, cla_temp_vx1, cla_temp_vy1, cla_temp_vx2, cla_temp_vy2))
                """
                cla_x1 = numpy.append(cla_x1, cla_temp_x1)
                cla_y1 = numpy.append(cla_y1, cla_temp_y1) 
                cla_x2 = numpy.append(cla_x2, cla_temp_x2)
                cla_y2 = numpy.append(cla_y2, cla_temp_y2)
                """
                cla_vx1 = numpy.append(cla_vx1, cla_temp_vx1) 
                cla_vy1 = numpy.append(cla_vy1, cla_temp_vy1)
                cla_vx2 = numpy.append(cla_vx2, cla_temp_vx2)
                cla_vy2 = numpy.append(cla_vy2, cla_temp_vy2)
                cla_fx1 = numpy.append(cla_fx1, cla_temp_fx1) 
                cla_fy1 = numpy.append(cla_fy1, cla_temp_fy1) 
                cla_fx2 = numpy.append(cla_fx2, cla_temp_fx2)
                cla_fy2 = numpy.append(cla_fy2, cla_temp_fy2)
                cla_time = numpy.append(cla_time, dt*t)
                """
                cla_separation = numpy.append(cla_separation, real((((cla_temp_x2 - cla_temp_x1)**2.0) + ((cla_temp_y2 - cla_temp_y1)**2.0))**0.5))
                
            if cla_r_temp > 0.6*r_c:
                print("Classical Run")
                print(cla_r_temp)
                cla_theta_b_theta = numpy.append(cla_theta_b_theta, (opening_angle((cla_temp_vy1), (cla_temp_vx1))))
                cla_theta_b_b = numpy.append(cla_theta_b_b, b_init)
                cla_vx_final = cla_temp_vx1
                cla_vy_final = cla_temp_vy1
                cla_ek_final = kinetic(cla_vx_final, cla_vy_final)
                cla_e_f = numpy.append(cla_e_f, cla_ek_final)
                cla_de = numpy.append(cla_de, (cla_ek_final - ek_init))
                pyplot.plot(cla_separation)
                print(min(cla_separation))
                pyplot.show()


                #dump the angles. 
                print("Out of simulation range.")
                break
            
            t += 1


        #The relativistic simulation. 


        #Relativistic Arrays and data
        
        #rel_t1 = numpy.array([])
        rel_x1 = numpy.array([])
        rel_y1 = numpy.array([])

        #rel_t1 = numpy.array([]) 
        rel_x2 = numpy.array([])
        rel_y2 = numpy.array([])
        """
        rel_vx1 = numpy.array([])
        rel_vy1 = numpy.array([])
        
        rel_vx2 = numpy.array([])
        rel_vy2 = numpy.array([])

        rel_fx1 = numpy.array([])
        rel_fy1 = numpy.array([])
        
        rel_fx2 = numpy.array([])
        rel_fy2 = numpy.array([])

        rel_separation = numpy.array([])

        rel_time = numpy.array([])
        """
        rel_temp_t1 = 0
        rel_temp_x1 = 0
        rel_temp_y1 = b_init

        rel_temp_t2 = 0 
        rel_temp_x2 = dist_init
        rel_temp_y2 = 0

        rel_temp_vx1 = v_init
        rel_temp_vy1 = 0
        
        rel_temp_vx2 = -v_init
        rel_temp_vy2 = 0

        rel_temp_fx1 = 0
        rel_temp_fy1 = 0
        
        rel_temp_fx2 = 0
        rel_temp_fy2 = 0

        rel_r_temp = 0

        rel_px_temp = 0
        rel_py_temp = 0
        
        """
        rel_px = numpy.array([])
        rel_py = numpy.array([])

        rel_energy = numpy.array([])
        """
        x1_traj = numpy.array([0])
        y1_traj = numpy.array([b_init])
        t1_traj = numpy.array([0])

        x2_traj = numpy.array([dist_init])
        y2_traj = numpy.array([0])
        t2_traj = numpy.array([0])
        ret_e_0 = numpy.append(ret_e_0, kinetic(v_init, 0))
        #nrg1_temp = 0
        #nrg2_temp = 0

        t = 0
        print("")
        print("_____________________________________")
        print("Relativistic Run:")
        print("")
        rel_separation = numpy.array([])
        while t < n_timesteps:
            rel_temp_t1, rel_temp_x1, rel_temp_y1, rel_temp_t2, rel_temp_x2, rel_temp_y2, rel_temp_vx1, rel_temp_vy1, rel_temp_vx2, rel_temp_vy2, rel_temp_fx1, rel_temp_fy1, rel_temp_fx2, rel_temp_fy2, x1_traj, y1_traj, t1_traj, x2_traj, y2_traj, t2_traj, nrg1_temp, nrg2_temp  = rel_leapfrog(rel_temp_t1, rel_temp_x1, rel_temp_y1, rel_temp_t2, rel_temp_x2, rel_temp_y2, rel_temp_vx1, rel_temp_vy1, rel_temp_vx2, rel_temp_vy2, rel_temp_fx1, rel_temp_fy1, rel_temp_fx2, rel_temp_fy2, x1_traj, y1_traj, t1_traj, x2_traj, y2_traj, t2_traj, dt)
            if t%10000 == 0: #sample rate for retarded potential case. 
                print("Retarded Run:")
                print(t)
                print("dt " + str(dt))
                print(rel_r_temp)
                print("searching a list of length:")
                print(len(t1_traj))
                print("working on the " + str(speeds*(impact+1)) + "th case out of " + str(n_runs))
                print("impact parameter: " + str(b_init))
                print("-_-_-_-_-_-_-_-_")
                print("")
                rel_r_temp = real((((rel_temp_x2 - rel_temp_x1)**2.0) + ((rel_temp_y2 - rel_temp_y1)**2.0))**0.5)
                if rel_temp_x1 != 0.0:
                    print(opening_angle((rel_temp_y1 - rel_temp_y2), (rel_temp_x1 - rel_temp_x2)))
                print("seperation: " + str(rel_r_temp) + " m")
                print("x force: " + str(rel_temp_fx1) + " N")
                print("y force: " + str(rel_temp_fy1) + " N")
                print("atom 1 x velocity: " + str(rel_temp_vx1) + " m/s")
                print("atom 1 y velocity: " + str(rel_temp_vy1) + " m/s")
                print("atom 2 x velocity: " + str(rel_temp_vx2) + " m/s")
                print("atom 2 y velocity: " + str(rel_temp_vy2) + " m/s")
                print("atom 1 x position: " + str(rel_temp_x1) + " m")
                print("atom 1 y position: " + str(rel_temp_y1) + " m")
                print("atom 2 x position: " + str(rel_temp_x2) + " m")
                print("atom 2 y position: " + str(rel_temp_y2) + " m")
                rel_px_temp, rel_py_temp = momentum(rel_temp_vx1, rel_temp_vy1, rel_temp_vx2, rel_temp_vy2)
                """
                rel_px = numpy.append(rel_px, rel_px_temp)
                rel_py = numpy.append(rel_py, rel_py_temp)
                rel_energy = numpy.append(rel_energy, nrg1_temp + nrg2_temp)
                """
                
                rel_x1 = numpy.append(rel_x1, rel_temp_x1)
                rel_y1 = numpy.append(rel_y1, rel_temp_y1) 
                rel_x2 = numpy.append(rel_x2, rel_temp_x2)
                rel_y2 = numpy.append(rel_y2, rel_temp_y2)
                """
                rel_vx1 = numpy.append(rel_vx1, rel_temp_vx1) 
                rel_vy1 = numpy.append(rel_vy1, rel_temp_vy1)
                rel_vx2 = numpy.append(rel_vx2, rel_temp_vx2)
                rel_vy2 = numpy.append(rel_vy2, rel_temp_vy2)
                rel_fx1 = numpy.append(rel_fx1, rel_temp_fx1) 
                rel_fy1 = numpy.append(rel_fy1, rel_temp_fy1) 
                rel_fx2 = numpy.append(rel_fx2, rel_temp_fx2)
                rel_fy2 = numpy.append(rel_fy2, rel_temp_fy2)
                rel_time = numpy.append(rel_time, dt*t)
                """
                rel_separation = numpy.append(rel_separation, real((((rel_temp_x2 - rel_temp_x1)**2.0) + ((rel_temp_y2 - rel_temp_y1)**2.0))**0.5))
                
            if rel_r_temp > 0.6*r_c:
                #this is where you dump the angle.
                print("Retarded Run") 
                rel_theta_b_theta = numpy.append(rel_theta_b_theta, opening_angle((rel_temp_vy1), (rel_temp_vx1)))
                rel_theta_b_b = numpy.append(rel_theta_b_b, b_init)
                rel_vx_final = numpy.append(rel_vx_final, rel_temp_vx1)
                rel_vy_final = numpy.append(rel_vy_final, rel_temp_vy1)
                vx_ratio = numpy.append(vx_ratio, rel_temp_vx1/cla_vx_final)
                vy_ratio = numpy.append(vy_ratio, rel_temp_vy1/cla_vy_final)
                kin_ratio = numpy.append(kin_ratio, (kinetic(rel_temp_vx1, rel_temp_vy1) - ek_init)/ek_init)
                ret_e_f = numpy.append(ret_e_f, kinetic(rel_temp_vx1, rel_temp_vy1))
                ret_de = numpy.append(ret_de, (kinetic(rel_temp_vx1, rel_temp_vy1) - kinetic(v_init, 0)))
                pyplot.figure()
                pyplot.plot(rel_x1, rel_y1, "r.-")
                pyplot.plot(cla_x1, cla_y1, "r-")
                pyplot.plot(rel_x2, rel_y2, "b.-")
                pyplot.plot(cla_x2, cla_y2, "b-")

                print(rel_theta_b_theta)
                print(cla_theta_b_theta)
                pyplot.show()


                print(rel_r_temp)
                print("Out of simulation range.")
                break
            t += 1
    print(rel_theta_b_theta)
    print("-_-_-_-_-_-_-_-_-_-_-_-_-")
    print(cla_theta_b_theta)
    print("-_-_-_-_-_-_-_-_-_-_-_-_-")
    print(cla_theta_b_b)
    print("-_-_-_-_-_-_-_-_-_-_-_-_-")
    print(vx_ratio)
    print("-_-_-_-_-_-_-_-_-_-_-_-_-")
    print(vy_ratio)
    print("-_-_-_-_-_-_-_-_-_-_-_-_-")
    print(rel_vx_final)
    print("-_-_-_-_-_-_-_-_-_-_-_-_-")
    print(rel_vy_final)
    print("-_-_-_-_-_-_-_-_-_-_-_-_-")

cla_angstrom_b = numpy.array([])
rel_angstrom_b = numpy.array([])
for par in rel_theta_b_b:
    cla_angstrom_b = numpy.append(cla_angstrom_b, par*(10**10.0))
    rel_angstrom_b = numpy.append(rel_angstrom_b, par*(10**10.0))

print("")
print("")
print("")
print("")
print("")
print("")
print("___________---------------______________--------------______________-------------______________")
print("impact parameter:")
for point in rel_angstrom_b:
    print(str(point), end = ", ")

print("")
print("___________---------------______________--------------______________-------------______________")
print("classical angle:")
for point in cla_theta_b_theta:
    print(str(point), end = ", ")

print("")
print("___________---------------______________--------------______________-------------______________")
print("retarded angle:")
for point in rel_theta_b_theta:
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

#Plotting zone. -----------------------------------------------------------

pyplot.plot(cla_angstrom_b, cla_theta_b_theta, "s-b", label = "Classical Potential")
pyplot.plot(rel_angstrom_b, rel_theta_b_theta, "^-r", label = "Retarded Potential")
pyplot.xlabel("Impact Parameter ($\AA$)")
pyplot.ylabel("Deflection Angle (Degrees)")
pyplot.legend()
pyplot.figure()
pyplot.plot(rel_angstrom_b, kin_ratio, "s-b", label = "Y Component")
pyplot.yscale("log")
pyplot.xlabel("Impact Parameter ($\AA$)")
pyplot.ylabel("Velocity Ratio (no units)")
pyplot.legend()

pyplot.show()


"""
pyplot.title("classical position space trajectory")
pyplot.plot(cla_x1, cla_y1, "r", label = "atom 1")
pyplot.plot(cla_x2, cla_y2, "b", label = "atom 2")
pyplot.legend()

pyplot.figure()
pyplot.title("classical atom 1 vel")
pyplot.plot(cla_time, cla_vx1, "r")
pyplot.plot(cla_time, cla_vy1, "b")

pyplot.figure()
pyplot.title("classical seperation")
pyplot.plot(cla_time, cla_separation, "r")

pyplot.figure()
pyplot.title("classical atom 1 pos")
pyplot.plot(cla_time, cla_x1, "r")
pyplot.plot(cla_time, cla_y1, "b")

pyplot.figure()
pyplot.title("classical atom 1 forces")
pyplot.plot(cla_time, cla_fx1, "r")
pyplot.plot(cla_time, cla_fy1, "b--")


pyplot.figure()
pyplot.title("classical atom 2 forces")
pyplot.plot(cla_time, cla_fx2, "r")
pyplot.plot(cla_time, cla_fy2, "b--")

pyplot.figure()
pyplot.title("relativistic position space trajectory")
pyplot.plot(rel_x1, rel_y1, "r")
pyplot.plot(rel_x2, rel_y2, "b")

pyplot.figure()
pyplot.title("relativistic atom 1 vel")
pyplot.plot(rel_time, rel_vx1, "r")
pyplot.plot(rel_time, rel_vy1, "b")

pyplot.figure()
pyplot.title("relativistic seperation")
pyplot.plot(rel_time, rel_separation, "r")

pyplot.figure()
pyplot.title("relativistic atom 1 pos")
pyplot.plot(rel_time, rel_x1, "r")
pyplot.plot(rel_time, rel_y1, "b")

pyplot.figure()
pyplot.title("relativistic atom 1 forces")
pyplot.plot(rel_time, rel_fx1, "r")
pyplot.plot(rel_time, rel_fy1, "b--")


pyplot.figure()
pyplot.title("relativistic atom 2 forces")
pyplot.plot(rel_time, rel_fx2, "r")
pyplot.plot(rel_time, rel_fy2, "b--")
pyplot.figure()



pyplot.plot(cla_time, cla_vx1, "r", label = "classical x velocity")
pyplot.plot(cla_time, cla_vy1, "b", label = "classical y velocity")
pyplot.plot(rel_time, rel_vx1, "g", label = "retarded x velocity")
pyplot.plot(rel_time, rel_vy1, "k", label = "retarded y velocity")
pyplot.xlabel("Time (s)")
pyplot.ylabel("Velocity ($ms^-1$)")
pyplot.legend()
pyplot.savefig("velo_for_" + b_init_text + "_g_ " + g_init + ".pdf")
pyplot.figure()


pyplot.plot(cla_x1, cla_y1, "r", label = "atom 1 cla")
pyplot.plot(cla_x2, cla_y2, "b", label = "atom 2 cla")
pyplot.plot(rel_x1, rel_y1, "g", label = "atom 1 rel")
pyplot.plot(rel_x2, rel_y2, "k", label = "atom 2 rel")
pyplot.xlabel("X Position (m)")
pyplot.ylabel("Y Position (m)")
pyplot.legend()
pyplot.savefig("traj_for_" + b_init_text + "_g_ " + g_init + ".pdf")
pyplot.figure()

pyplot.title("momentum for b = " + b_init_text + ", g = " + g_init + " ms^-1")
pyplot.plot(rel_time, rel_px, label = "retarded x momentum")
pyplot.plot(rel_time, rel_py, label = "retarded y momentum")
pyplot.plot(cla_time, cla_px, label = "classical x momentum")
pyplot.plot(cla_time, cla_py, label = "classical y momentum")
pyplot.xlabel("Time (s)")
pyplot.ylabel("Momentum (kg*m*s^-1)")
pyplot.legend()
pyplot.savefig("tot_mome_for_" + b_init_text + "_g_ " + g_init + ".pdf")
pyplot.figure()

pyplot.title("classical momentum for b = " + b_init_text + ", g = " + g_init + " ms^-1")
pyplot.plot(cla_time, cla_px, label = "classical x momentum")
pyplot.plot(cla_time, cla_py, label = "classical y momentum")
pyplot.xlabel("Time (s)")
pyplot.ylabel("Momentum (kg*m*s^-1)")
pyplot.legend()
pyplot.savefig("cla_mome_for_" + b_init_text + "_g_ " + g_init + ".pdf")
pyplot.figure()

pyplot.title("retarded momentum for b = " + b_init_text + ", g = " + g_init + " ms^-1")
pyplot.plot(rel_time, rel_px, label = "retarded x momentum")
pyplot.plot(rel_time, rel_py, label = "retarded y momentum")
pyplot.xlabel("Time (s)")
pyplot.ylabel("Momentum (kg*m*s^-1)")
pyplot.legend()
pyplot.savefig("ret_mome_for_" + b_init_text + "_g_ " + g_init + ".pdf")
pyplot.figure()




pyplot.plot(cla_time, cla_energy, label = "classical energy")
pyplot.plot(rel_time, rel_energy, label = "retarded energy")
pyplot.xlabel("Time (s)")
pyplot.ylabel("Energy (GeV)")
pyplot.legend(loc = "upper right")
pyplot.savefig("tot_energy_for_" + b_init_text + "_g_ " + g_init + ".pdf")
pyplot.figure()

pyplot.title("classical energy for b = " + b_init_text + ", g = " + g_init + " ms^-1")
pyplot.plot(cla_time, cla_energy)
pyplot.xlabel("Time (s)")
pyplot.ylabel("Energy (GeV)")
pyplot.savefig("cla_energy_for_" + b_init_text + "_g_ " + g_init + ".pdf")
pyplot.figure()

pyplot.title("retarded energy for b = " + b_init_text + ", g = " + g_init + " ms^-1")
pyplot.plot(rel_time, rel_energy)
pyplot.xlabel("Time (s)")
pyplot.ylabel("Energy (GeV)")
pyplot.savefig("ret_energy_for_" + b_init_text + "_g_ " + g_init + ".pdf")






pyplot.show()
"""
