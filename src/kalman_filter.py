import matplotlib.pyplot as plt
from filterpy.kalman import ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy as np
from tqdm import tqdm

rx_camera = 0.05        #
ry_camera = 0.05        #
rtheta_camera = 0.0348   #   noise from sensors -> To change from tests (variance)
r_motor_left = 0.321      #
r_motor_right = 0.207     #

class ThymioEKF(ExtendedKalmanFilter):
    def __init__(self):
        super().__init__(dim_x=3, dim_z=3, dim_u=2)  # [x, y, theta]

        self.wheel_base = 9.5 #Wheel separation in cm for angular velocity

        self.x = np.array([0., 0., 0.])

         #maps the 3 states to 3 measurements
        self.H = np.eye(3)

        self.P = np.diag([
            0.01, #Placement error in x
            0.01, #Placement error in y
            0.09, #Placement error in theta   }-> Check all that
        ])

        self.S_input=np.diag([r_motor_left,r_motor_right])

        self.Q_noise = np.diag([0.1, 0.1, 0.1])  # Process noise due to unaccounted effects (slip/friction)
        self.Q = np.diag([0.001, 0.001, 0.01])
        self.R = np.diag([
            rx_camera,  # noise from camera
            ry_camera,  # noise from camera
            rtheta_camera,  # noise from camera
        ])

    def initialize_kalman_pos(self,current_pos):
        self.x=current_pos

    def fx(self, dt, u):
        theta = self.x[2]
        v_l, v_r = u[0], u[1]
        wheel_base = self.wheel_base
        omega = (v_r - v_l) / wheel_base

        if abs(omega) < 1e-6:
            # Very small turn radius - approximate as straight
            dx = dt * (v_l + v_r) / 2 * np.cos(theta)
            dy = dt * (v_l + v_r) / 2 * np.sin(theta)
            dtheta = 0
        else:
            R = (v_l + v_r) / 2   # Turning radius
            dx = R * (np.cos(theta)*dt)
            dy = R * (np.sin(theta)*dt)
            dtheta = omega * dt

        # Normalize theta
        new_theta = (self.x[2] + dtheta + np.pi) % (2 * np.pi) - np.pi
        return np.array([self.x[0] + dx, self.x[1] + dy, new_theta])

    def F_jacobian(self,dt, u):

        #returns jacobian of state transition

        theta = self.x[2]

        F = np.eye(3)
        F[0, 2] = -dt * (u[0] + u[1])/2 * np.sin(theta)
        F[1, 2] = dt * (u[0] + u[1])/2 * np.cos(theta)

        return F

    def B_jacobian(self,dt):
        theta = self.x[2]
        B = np.array([
            [0.5*np.cos(theta)*dt,0.5*np.cos(theta)*dt],
            [0.5*np.sin(theta)*dt,0.5*np.sin(theta)*dt],
            [-dt/self.wheel_base,-dt/self.wheel_base],
        ])
        return B

    def H_jacobian(self, x):
        # Update method needs a function
        return self.H

    def predict_Jacob(self, dt, u):
        # Recompute self.F
        F = self.F_jacobian(dt, u)
        B = self.B_jacobian(dt)
        #self.predict(u)
        self.x = self.fx(dt,u)
        BS=np.dot(B,self.S_input)
        Q=self.Q_noise+np.dot(BS,B.T)
        self.Q=Q
        self.F=F
        self.B=B
        FP=np.dot(self.F, self.P)
        self.P_prior = np.dot(FP,self.F.T) + Q
        self.P=np.copy(self.P_prior)

    def hx(self, x):
        #measurement step
        return np.dot(self.H, x)

    def update_modified(self, z, HJacobian, Hx, R=None, args=(), hx_args=(),
            residual=np.subtract):

        if z is None:
            self.z = np.array([[None]*self.dim_z]).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            return

        if not isinstance(args, tuple):
            args = (args,)

        if not isinstance(hx_args, tuple):
            hx_args = (hx_args,)

        if R is None:
            R = self.R
        elif np.isscalar(R):
            R = np.eye(self.dim_z) * R

        if np.isscalar(z) and self.dim_z == 1:
            z = np.asarray([z], float)

        H = HJacobian(self.x, *args)

        PHT = np.dot(self.P, H.T)
        self.S = np.dot(H, PHT) + R
        self.K = PHT.dot(np.linalg.inv(self.S))

        hx = Hx(self.x, *hx_args)
        self.y = residual(z, hx)
        self.x = self.x + np.dot(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK' is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.
        I_KH = self._I - np.dot(self.K, H)
        self.P = np.dot(I_KH, self.P).dot(I_KH.T) + np.dot(self.K, R).dot(self.K.T)

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        # save measurement and posterior state
        self.z = z
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
