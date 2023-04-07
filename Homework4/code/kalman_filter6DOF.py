import numpy as np

class KalmanFilter6DOF():
    def __init__(self, x=0, y=0, yaw=0, dx=0, dy=0, dyaw=0):
        # State [x, y, yaw]
        self.x = np.array([x, y, yaw, dx, dy, dyaw])
        # Transition matrix
        self.A = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0],
                           [-1, 0, 0, 0, 0, 0],
                           [0, -1, 0, 0, 0, 0],
                           [0, 0, -1, 0, 0, 0]])
        self.B = np.array([[0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1],
                           [1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0]])
        # self.A = np.array([[1, 0, 0, 0, 0, 0],
        #                    [0, 1, 0, 0, 0, 0],
        #                    [0, 0, 1, 0, 0, 0],
        #                    [0, 0, 0, 1, 0, 0],
        #                    [0, 0, 0, 0, 1, 0],
        #                    [0, 0, 0, 0, 0, 1]])
        # self.B = np.array([[0, 0, 0, 1, 0, 0],
        #                    [0, 0, 0, 0, 1, 0],
        #                    [0, 0, 0, 0, 0, 1],
        #                    [0, 0, 0, 1, 0, 0],
        #                    [0, 0, 0, 0, 1, 0],
        #                    [0, 0, 0, 0, 0, 1]])
        # Error matrix
        self.P = np.identity(6) * 1
        # Observation matrix
        self.H = np.identity(3) * 1
        # State transition error covariance
        self.R = np.array([[0.1, 0, 0],
                           [0, 0.1, 0],
                           [0, 0, 0.1]])
        # Measurement error
        self.Q = np.array([[0.75, 0, 0],
                           [0, 0.75, 0 ],
                           [0, 0, 0.75]])

        ## For testing
        print('Setting KF over !')

    def predict(self, u):
        # raise NotImplementedError
        # print('u :\r\n')
        # print(u)
        self.x = np.matmul(self.A,self.x) + np.matmul(self.B,u)
        self.P = np.matmul(np.matmul(self.A,self.P),np.transpose(self.A)) + self.R
        # print('-------------')
        ## Cause we don't update yaw, we need to prevent
        ## covariance of yaw become inf
        # self.P[2,2] = 1


    def update(self, z):
        # raise NotImplementedError
        im_z = np.transpose(np.append(z, 0))
        ## Kalman Filter update part
        temp_sigma = np.matmul(np.matmul(self.H, self.P[:3,:3]), np.transpose(self.H)) + self.Q
        Kt = np.matmul(np.matmul(self.P[:3,:3], np.transpose(self.H)), np.linalg.inv(temp_sigma))
        self.x[:3] = self.x[:3] + np.matmul(Kt,(im_z - np.matmul(self.H, self.x[:3])))
        self.P[:3,:3] = np.matmul((np.identity(3) - np.matmul(Kt, self.H)), self.P[:3,:3])

        if np.isnan(np.sum(self.x)) == True :
            raise ValueError
        
        return self.x, self.P
