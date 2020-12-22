import numpy as np
from scipy import integrate
from .utility import read_csv

"""
Simulates a spin-stabilized launch profile
"""
class Profile:
    def __init__(self, rocket, motor, init_spin, length=0, motor_pos=0, hangle=0, timesteps=50):
        self.rocket = read_csv(rocket)
        self.motor = motor
        self.init_spin = init_spin
        self.length = length
        # assume a length==0 implies simulation should end at end of motor burn
        if self.length == 0:
            self.length = self.motor.burn_time
        self.motor_pos = motor_pos

        # thrust is set to polynomial fit to get equally spaced timesteps for subsequent calcs
        t = np.linspace(0, self.motor.burn_time, len(self.motor.thrust))
        # TODO: double check with prop that polynomial fit is sufficient and ask abt degree
        thrust = np.polyfit(t, self.motor.thrust, 4)
        force = np.poly1d(thrust) # TODO: subtract the drag and gravity
        
        # simple integration and Newton's second
        self.tt = np.linspace(0, self.length, timesteps)
        self.accel = np.array([force(t) / (self.motor.mass(t) + self.rocket["Mass"]) \
            for t in self.tt])
        vel = np.array(integrate.cumtrapz(self.accel, x=self.tt, initial=0))
        self.vel = vel 
        self.altit = np.array(integrate.cumtrapz(vel * np.cos(hangle), x=self.tt, initial=0))

    def rho(self):
        """
        Returns an array with the air density at heights of [0:self.altitude]
        Source for air density at below 11km: https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html
        """
        rho = []

        # Lower troposphere (0 - 11km) --- returns result in kg/m^3
        # Source: https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html
        # Assumes that the pressure and temperature change only with altitude
        # ** Are the other pressure calculations also in kg/m^3?
        l_trop_x = np.argmax(self.altit > 0)
        if l_trop_x == 0:
            l_trop_x = len(self.altit)
        rho.extend([((101.29 * ((15.04 - 0.00649 * x) + 273.1) / 288.08) ** 5.256) / (0.2869 * ((15.04 - 0.00649 * x) + 273.1))]) \
            for x in self.altit[:l_trop_x])

        # Higher troposphere (11km - 20km)
        h_trop_x = np.argmax(self.altit > 11000)
        if h_trop_x == 0:
            h_trop_x = len(self.altit)
        rho.extend([1.225 * (288.15 / (288.15 + -0.0065 * x ** (1 + (9.8 * 0.02896) / (8.3145 * -0.0065)))) \
            for x in self.altit[l_trop_x:h_trop_x]])

        # Stratosphere (20km - 32 km)
        strat_x = np.argmax(self.altit > 20000)
        if strat_x == 0:
            strat_x = len(self.altit)
        rho.extend([0.364 * np.exp(-(9.8 * 0.02896 * x / (8.3145 * 216.65))) \
            for x in self.altit[h_trop_x:strat_x]])

        # assumed below the mesosphere (32km)
        rho.extend([0.088 * (216.65 / (216.65 + 0.001 * x ** (1 + (9.8 * 0.02896) / (8.3145 * 0.001)))) \
            for x in self.altit[h_trop_x:strat_x]])

        return np.array(rho)

    def iz(self):
        return np.array([self.rocket["I_z"] + self.motor.iz(time) + self.motor.mass(time) * self.motor_pos**2 \
            for time in self.tt])

    def ix(self):
        return np.array([self.rocket["I_x"] + self.motor.ix(time) + self.motor.mass(time) * self.motor_pos**2 \
            for time in self.tt])

    def gyro_stab_crit(self):
        # TODO: the number of calipers also changes as motor burns and CG changes, add fcn for this too
        return self.vel / self.ix() * np.sqrt(2 * self.rho() * self.iz() * self.rocket['Surface Area'] * \
            self.rocket['Calipers'] * self.rocket['Diameter']) 

    def dynamic_stab_crit(self):
        #TODO: McCoy dynamics stability criterion
        return np.ones(len(self.tt)) * self.init_spin

    def spin(self):
        # TODO: incorporate spin damping moment
        return np.ones(len(self.tt)) * self.init_spin

    def is_stable(self):
        return self.stab_crit() < self.spin()

    def min_spin(self):
        # TODO: incorporate skin drag despin
        return np.max(self.stab_crit())