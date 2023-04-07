import numpy as np

class KalmanFilter():
    def __init__(self, x=0, y=0, yaw=0):
        # State [x, y, yaw]
        self.x = np.array([x, y, yaw])
        # Transition matrix
        self.A = np.identity(3)
        self.B = np.identity(3)
        # Error matrix
        self.P = np.identity(3) * 1
        # Observation matrix
        self.H = np.array([[1, 0, 0],
                           [0, 1, 0]])
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
        self.x = np.matmul(self.A,self.x) + np.matmul(self.B,u)
        self.P = np.matmul(np.matmul(self.A,self.P),np.transpose(self.A)) + self.R
        ## Cause we don't update yaw, we need to prevent
        ## covariance of yaw become inf
        self.P[2,2] = 1


    def update(self, z):
        # raise NotImplementedError
        ## Make z become 3*1 and H become 3*3 matrix
        im_z = np.transpose(np.append(z, 0))
        im_H = np.vstack([self.H, np.array([0, 0, 0])])
        ## Kalman Filter update part
        temp_sigma = np.matmul(np.matmul(im_H, self.P), np.transpose(im_H)) + self.Q
        Kt = np.matmul(np.matmul(self.P, np.transpose(im_H)), np.linalg.inv(temp_sigma))
        self.x = self.x + np.matmul(Kt,(im_z - np.matmul(im_H, self.x)))
        self.P = np.matmul((np.identity(3) - np.matmul(Kt, im_H)), self.P)

        if np.isnan(np.sum(self.x)) == True :
            raise ValueError
        
        return self.x, self.P
