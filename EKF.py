#!/usr/bin/env python

# ROS python api with lots of handy ROS functions
# coding=utf-8
import rospy
import time
import math
import tf
import numpy as np
import copy
from numpy import linalg as LA
from laser_geometry import LaserProjection
import matplotlib.pyplot as plt
from icp2 import *
import pickle

# to be able to subcribe to laser scanner data
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2 as pc2
import sensor_msgs.point_cloud2 as pcmapping
from operator import itemgetter

# to be able to publish Twist data (and move the robot)
from geometry_msgs.msg import Twist, Pose, PoseWithCovarianceStamped

# to be able to subcribe to odometry data
from nav_msgs.msg import Odometry, OccupancyGrid


class mbotSimpleBehavior(object):
    '''
    Exposes a behavior for the pioneer robot so that moves forward until
    it has an obstacle at 1.0 m then stops rotates for some time to the
    right and resumes motion.
    '''

    def __init__(self):
        '''
        Class constructor: will get executed at the moment
        of object creation
        '''
        self.odom_msg_received = False
        self.laser_msg_received = False
        self.map_msg_received = False
        # register node in ROS network
        rospy.init_node('pioneer_smart_behavior', anonymous=False)
        self.INITIALPOSE_x = -1.39379392735
        self.INITIALPOSE_y=-1.33716776885
        self.INITIALPOSE_teta=0


        # send map as point cloud
        self.publish_map = rospy.Publisher("/wall_map",pc2, queue_size=1 )
        # subscribe our laser point cloud
        self.laserProjection = LaserProjection()
        # puclish pointcloudlaser
        self.pcPub = rospy.Publisher("/laserPointCloud", pc2, queue_size=1)
        #
        self.matchinginfo = rospy.Publisher("/laserPointCloud", pc2, queue_size=1)
        # print message in terminal
        rospy.loginfo('Pioneer simple behavior started !')
        # subscribe to pioneer laser scanner topic
        rospy.Subscriber("/scan_combined", LaserScan, self.laserCallback, queue_size=1)
        # subscribe to pioneer odometry topic
        rospy.Subscriber("/odom", Odometry, self.odomCallback, queue_size=1)
        # subscribe to mbot map topic
        rospy.Subscriber("/map", OccupancyGrid, self.mapCallback, queue_size=1)
        rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.amclposeCallback, queue_size = 1 )

        # setup publisher to later on move the pioneer base
        #self.pub_cmd_vel = rospy.Publisher('robot_0/cmd_vel', Twist, queue_size=1)
        # define member variable and initialize with a big value
        # it will store the distance from the robot to the walls
        #self.distance = 10.0
        # Last two odometry readings
        self.initpose_icp = np.array([[np.cos(self.INITIALPOSE_teta),-np.sin(self.INITIALPOSE_teta),self.INITIALPOSE_x],[np.sin(self.INITIALPOSE_teta),np.cos(self.INITIALPOSE_teta),self.INITIALPOSE_y],[0,0,1]])
        self.varicp = 0
        self.amcl = np.array([[self.INITIALPOSE_x],[self.INITIALPOSE_y]])



        self.currentmsgodom = [self.INITIALPOSE_x,self.INITIALPOSE_y,self.INITIALPOSE_teta]
        self.currentmsgamcl = [self.INITIALPOSE_x,self.INITIALPOSE_y,self.INITIALPOSE_teta]
        self.odom = {'x': self.INITIALPOSE_x , 'y': self.INITIALPOSE_y, 'teta': self.INITIALPOSE_teta
            , 'x_1': self.INITIALPOSE_x,'y_1': self.INITIALPOSE_y, 'teta_1': self.INITIALPOSE_teta}
        self.odom_read =  {'x': self.INITIALPOSE_x , 'y': self.INITIALPOSE_y, 'teta': self.INITIALPOSE_teta
            , 'x_1': self.INITIALPOSE_x, 'y_1': self.INITIALPOSE_y, 'teta_1': self.INITIALPOSE_teta}
        # EKF pose predcition
        self.pose_prediction = {'x': self.INITIALPOSE_x, 'y': self.INITIALPOSE_y, 'teta': self.INITIALPOSE_teta}
        self.init_pose_prediction = {'x': self.INITIALPOSE_x, 'y': self.INITIALPOSE_y, 'teta': self.INITIALPOSE_teta}
        self.pose_estimation={'x': self.INITIALPOSE_x, 'y':self.INITIALPOSE_y, 'teta': self.INITIALPOSE_teta}
        # self.pose_prediction = {'x': 0,'y': 0, 'teta': 0}
        # Jacobian matrix g
        self.matrix_G = np.identity(3)
        self.matrix_H = np.identity(3)
        self.matrix_R = np.array([[0.9,0,0],[0,1.1,0],[0,0,100]])
        # covariance of noise observation matrix
        self.matrix_Q = np.array([[1,0,0],[0,1,0],[0,0,1]])
        # Covariance of instance t-1
        #self.cov_matrix = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])
        # covariance of instance t (actual)
        self.prev_cov =np.identity(3)
        self.act_cov = np.identity(3)
        # predicted covariance
        self.pred_cov = np.identity(3)
        self.pred_update = np.array([0,0,0], dtype=float)
        self.z_measurements = np.array([0,0,0],dtype=float)

        self.map = []
        self.width = 0
        self.height = 0
        self.resolution = 0
        self.cloudpointmap = []
        self.currentlaserdata = []
        self.usingpointlaser= []

        # defines the range threshold bellow which the robot should stop moving foward and rotate insteadnp.array([[np.cos(0),-np.sin(0),-1.37660027342],[np.sin(0),np.cos(0.003139031004),4.01820473614],[0,0,1]])
        # if rospy.has_param('distance_threshold'):
        #     # retrieves the threshold from the parameter server in the case where the parameter exists
        #     self.distance_threshold = rospy.get_param('distance_threshold')
        # else:
        #     self.distance_threshold = 1.0

    def odomCallback(self, msg):
        '''
        This function gets executed everytime a odomotry reading msg is received 		on the
        topic: /robot_0/odometry
        '''
        self.odom_msg_received = True
        # print("ANGULO", self.quaternion_to_euler(msg))
        # print("x", msg.pose.pose.position.x, "y", msg.pose.pose.position.y)
        self.currentmsgodom[0] = msg.pose.pose.position.x + self.init_pose_prediction['x'] - 1.3108866698979635
        self.currentmsgodom[1] = msg.pose.pose.position.y +  self.init_pose_prediction['y'] - 0.19804321874743439

        self.currentmsgodom[2] = self.quaternion_to_euler(msg)+ self.init_pose_prediction['teta'] + 2.401482407814359




    def amclposeCallback(self, msg):
        #print('amcl pose x= ', msg.pose.pose.position.x, 'amcl pose y= ', msg.pose.pose.position.y, "amcl teta = ", self.quaternion_to_euler(msg))
        self.amcl = np.hstack((self.amcl, np.array([[ msg.pose.pose.position.x],[msg.pose.pose.position.y]])))
        self.currentmsgamcl[0] = msg.pose.pose.position.x
        self.currentmsgamcl[1] = msg.pose.pose.position.y
        self.currentmsgamcl[2] = np.remainder(self.quaternion_to_euler(msg), 2*np.pi)
        #print("amcl angulo", self.currentmsgamcl[2])
    def quaternion_to_euler(self, msg):

        quaternion = (msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                      msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)

        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(quaternion)
        return yaw

    def laserCallback(self, msg):
        '''
        This function gets executed everytime a laser scanner msg is received on the
        topic: /scan_combined
        '''
        self.laser_msg_received = True
        temporaria =  []
        cloud_out = self.laserProjection.projectLaser(msg)
        for p in pcmapping.read_points(cloud_out, field_names=("x", "y", "z"), skip_nans=True):
            temporaria.append([p[0], p[1]])
        self.currentlaserdata = temporaria
        #print("laser len", len(self.currentlaserdata))
        #print("point _ point_cloud_laser=", self.currentlaserdata)

    def mapCallback(self, msg):

        self.map = msg.data
        self.width = msg.info.width
        self.height = msg.info.height
        self.resolution = msg.info.resolution
        i = 0
        j = 0
        for i in range(self.width):
            for j in range(self.height):
                if (self.map[i + j * self.width] > 0.0):
                    self.cloudpointmap.append([self.resolution * i + msg.info.origin.position.x,
                                               self.resolution * j + msg.info.origin.position.y])
        self.map_msg_received = True

    def ekf_predict(self):
        '''
        perform one EKF predict
        '''

        self.odom['x'] = self.currentmsgodom[0]
        self.odom['y'] = self.currentmsgodom[1]
        self.odom['teta'] = self.currentmsgodom[2]

        dx = self.odom['x'] - self.odom['x_1']
        dy = self.odom['y'] - self.odom['y_1']
        dtheta = self.odom['teta'] - self.odom['teta_1']

        delta_trans = math.sqrt(dx ** 2 + dy ** 2)
        delta_rot1 = math.atan2(dy, dx) - self.odom['teta_1']
        delta_rot2 = dtheta - delta_rot1

        #self.prev_cov = self.act_cov

        self.pose_prediction['x'] =  self.pose_estimation['x'] + delta_trans * math.cos(
            self.pose_estimation['teta'] + delta_rot1)

        self.pose_prediction['y'] = self.pose_estimation['y'] + delta_trans * math.sin(
            self.pose_estimation['teta'] + delta_rot1)

        self.pose_prediction['teta'] = self.pose_estimation['teta'] + delta_rot1 + delta_rot2

        self.pose_prediction['teta'] = np.remainder(self.pose_estimation['teta'], 2 * np.pi)
        #print(self.pose_prediction)

        #print("pose_prediction x =" ,self.pose_prediction['x'],"pose_prediction y =" ,self.pose_prediction['y'],"pose_prediction theta=" ,self.pose_prediction['teta'], "\n",  "odom x =", self.odom['x'], "odom y =", self.odom['y'], "odom theta =", self.odom['teta'])

        self.matrix_G = np.array([[1, 0, -delta_trans * math.sin(self.pose_estimation['teta'] + delta_rot1)],
                                  [0, 1, delta_trans * math.cos(self.pose_estimation['teta'] + delta_rot1)], [0, 0, 1]])

        self.pred_cov = np.dot(self.matrix_G, np.dot(self.act_cov, self.matrix_G.transpose())) + self.matrix_R

        self.odom['x_1'] = self.odom['x']
        self.odom['y_1'] = self.odom['y']
        self.odom['teta_1'] =self.odom['teta']

    def ekf_update(self):

        pose_estimation_update = np.array([0,0,0], dtype=float)

        self.pred_update[0]=self.pose_prediction['x']
        self.pred_update[1]=self.pose_prediction['y']
        self.pred_update[2]=self.pose_prediction['teta']

        #print(self.pred_update)

        #print("pose_prediction x =" ,self.pose_prediction['x'],"pose_prediction y =" ,self.pose_prediction['y'],"pose_prediction theta=" ,self.pose_prediction['teta'], "\n",  "odom x =", self.odom['x'], "odom y =", self.odom['y'], "odom theta =", self.odom['teta'])


        #Este Z deveria vir do ICP
        z=np.array([[self.currentmsgamcl[0]], [self.currentmsgamcl[1]],[self.currentmsgamcl[2]]])

        #kalman gain
        k_gain = (self.pred_cov.dot(self.matrix_H.transpose())).dot(LA.inv((self.matrix_H.dot(self.pred_cov)).dot(self.matrix_H.transpose()) + self.matrix_Q))


        self.updatemeasurements_icp()
        pose_estimation_update = self.pred_update.reshape(3,1) + k_gain.dot((self.z_measurements.reshape(3,1) - self.pred_update.reshape(3,1)))

        #print(k_gain)
        #print(pose_estimation_update)
        #print(pose_estimation_update)
        self.pose_estimation['x'] = pose_estimation_update[0][0]
        self.pose_estimation['y'] = pose_estimation_update[1][0]
        self.pose_estimation['teta'] =pose_estimation_update[2][0]

        self.act_cov = (np.identity(3) - k_gain.dot(self.matrix_H)).dot(self.pred_cov)

    def updatemeasurements_icp(self):



        #print("pointcloudlaser" , self.point_cloud_laser)
        # x_max_pointcloud_laser = max(self.point_cloud_laser,key=itemgetter(1))[0]
        # y_max_pointcloud_laser = max(self.point_cloud_laser,key=itemgetter(1))[1]
        # x_min_pointcloud_laser = min(self.point_cloud_laser,key=itemgetter(1))[0]
        # y_min_pointcloud_laser = min(self.point_cloud_laser,key=itemgetter(1))[1]
        #
        #
        # resizedmap = [x for x in self.cloudpointmap if  (x_min_pointcloud_laser - 0.5 <= x[0] <= x_max_pointcloud_laser +0.5 and y_min_pointcloud_laser - 0.5 <= x[1] <= y_max_pointcloud_laser +0.5) ]

        self.usingpointlaser = self.currentlaserdata

        # x_max_pointcloud_laser = max(self.usingpointlaser,key=itemgetter(1))[0]
        # y_max_pointcloud_laser = max(self.usingpointlaser,key=itemgetter(1))[1]
        # x_min_pointcloud_laser = min(self.usingpointlaser,key=itemgetter(1))[0]
        # y_min_pointcloud_laser = min(self.usingpointlaser,key=itemgetter(1))[1]



        if(self.varicp == 0):
            # resizedmap = [x for x in self.cloudpointmap if  (-1.43967161853 + x_min_pointcloud_laser -1.2 <= x[0] <=  -1.43967161853 + x_max_pointcloud_laser+1.2  or  4.02608497565 + y_min_pointcloud_laser - 1.2 <= x[1] <= 4.02608497565 + y_max_pointcloud_laser +1.2 ) ]
            T , distances, i= icp(np.array(self.usingpointlaser), np.array(self.cloudpointmap), np.array([[np.cos(self.INITIALPOSE_teta),-np.sin(self.INITIALPOSE_teta),self.INITIALPOSE_x],[np.sin(self.INITIALPOSE_teta),np.cos(self.INITIALPOSE_teta),self.INITIALPOSE_y],[0,0,1]]))
            self.z_measurements = np.array([[T[0][2]],[T[1][2]], [np.arctan2(T[1][0], T[0][0])]])
            #print(T)

            #print("foi aqui")
            #print("mapa ",np.shape( np.array(self.cloudpointmap)))
            #print("laser", np.shape( np.array(self.usingpointlaser)))
            self.varicp=1

        else:
            #resizedmap = [x for x in self.cloudpointmap if  (self.pred_update[0] + x_min_pointcloud_laser <= x[0] <=  self.pred_update[0] + x_max_pointcloud_laser  or  self.pred_update[1] + y_min_pointcloud_laser <= x[1] <= self.pred_update[1] + y_max_pointcloud_laser ) ]
            #print("mapa ",np.shape( np.array(self.cloudpointmap)))
            #print("laser", np.shape( np.array(self.usingpointlaser)))
            T , distances, i= icp(np.array(self.usingpointlaser), np.array(self.cloudpointmap), np.array([[np.cos(self.pred_update[2]), -np.sin(self.pred_update[2]),self.pred_update[0]],[np.sin(self.pred_update[2]),np.cos(self.pred_update[2]),self.pred_update[1]],[0,0,1]]))
            #print(T)
            self.z_measurements[0]= T[0][2]
            self.z_measurements[1]= T[1][2]
            self.z_measurements[2]= np.arctan2(T[1][0], T[0][0])

        #print(T)



    def run_behavior(self):
        pred=np.array([[self.pose_prediction['x']],[self.pose_prediction['y']], [self.pose_prediction['teta']]])
        odometry = np.array([[self.odom_read['x_1']],[ self.odom_read['y_1']], [self.odom_read['teta_1']]])
        error = np.array([[0],[0],[0]])

        while not rospy.is_shutdown():
            # base needs this msg to be published constantly for the robot to keep moving so we publish in a loop

            # while the distance from the robot to the walls is bigger than the defined threshold keep moving forward
            ##if self.distance > self.distance_threshold:
            ##    self.move_forward()
            ##else:
            # rotate for a certain angle
            ##    self.rotate_right()

            # EKF - Predict
            r=rospy.Rate(5)
            while len(self.currentlaserdata) == 0 or self.map_msg_received == False:
                continue
            #print("MAIN LASER", self.point_cloud_laser)
            if self.odom_msg_received == True:
                #rospy.loginfo('odom msg received!')
                self.odom_msg_received = False
            if self.laser_msg_received == True:
                #rospy.loginfo('laser msg received!')
                self.laser_msg_received == True
            total_time = 0


            self.ekf_predict()
            start = time.time()
            self.ekf_update()
            end = time.time()
            #print(end-start)
            arr = np.array([[self.pose_estimation['x']], [self.pose_estimation['y']], [self.pose_estimation['teta']]])
            #print("PRED=", pred)
            #print ("ARR = ", arr)
            pred = np.hstack((pred, arr))
            error_array = np.array([[self.pose_estimation['x'] - self.currentmsgamcl[0]], [self.pose_estimation['y'] - self.currentmsgamcl[1]], [self.pose_prediction['teta']- self.currentmsgamcl[2]]])

            error = np.hstack((error, error_array))

            with open('pred.pkl', 'wb') as f:
                pickle.dump(pred, f)
            with open('amcl.pkl', 'wb') as f2:
                pickle.dump(self.amcl, f2)

            arr2 = np.array([[self.odom['x_1']],[ self.odom['y_1']], [self.odom['teta_1']]])
            odometry = np.hstack((odometry, arr2))
            #plt.plot(odometry[0, :].flatten(),
            #         odometry[1, :].flatten(), "-r")
            with open('rosbagbom.pkl', 'wb') as f:
                pickle.dump((error[0,:], error[1,:], error[2,:]), f)
            plt.plot(pred[0, :].flatten(),
                    pred[1, :].flatten(), "-b")
            plt.plot(self.amcl[0, :].flatten(),
                    self.amcl[1, :].flatten(), "-g")

            # plt.plot(error[0, :].flatten(), "-r")
            # plt.plot(error[1, :].flatten(), '-b')
            # sleep for a small amount of time
            plt.grid(True)
            plt.pause(0.00001)
            r.sleep()


def main():
    # create object of the class mbotSimpleBehavior (constructor will get executed!)
    my_object = mbotSimpleBehavior()
    # call run_behavior method of class pioneerSimpleBehavior
    my_object.run_behavior()


if __name__ == '__main__':
    # create object of the class pioneerSimpleBehavior (constructor will get executed!)
    my_object = mbotSimpleBehavior()
    # call run_behavior method of class pioneerSimpleBehavior
    my_object.run_behavior()
