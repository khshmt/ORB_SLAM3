/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <opencv2/opencv.hpp>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include "nav_msgs/Odometry.h"
#include <Eigen/Dense>
#include "opencv2/core/eigen.hpp"
#include <opencv2/core/core.hpp>

#include"../../../include/System.h"

#ifdef USE_BACKWARD
#define BACKWARD_HAS_DW 1
#include "backward.hpp"
namespace backward
{
    backward::SignalHandling sh;
}
#endif

using namespace std;

ros::Publisher pub_pose;
ros::Publisher pub_path;
nav_msgs::Path orb_path;

bool is_stop_dz = false;
void command()
{
    while(1)
    {
        char c = getchar();
        if (c == 'q')
        {
            is_stop_dz = true;
        }

        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
    }
}

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM3::System* pSLAM):mpSLAM(pSLAM){}

    void GrabStereo(const sensor_msgs::ImageConstPtr& msgLeft,const sensor_msgs::ImageConstPtr& msgRight);

    ORB_SLAM3::System* mpSLAM;
    bool do_rectify;
    cv::Mat M1l,M2l,M1r,M2r;
};

void convertOrbSlamPoseToOdom(const cv::Mat &cv_data, nav_msgs::Odometry &Twb) {

    assert(orb_data.rows == 7);
    Eigen::MatrixXf eig_data;
    cv::cv2eigen(cv_data, eig_data);
    Eigen::MatrixXd eig_data_d = eig_data.cast<double>();
    Eigen::Quaterniond q_wb(eig_data_d.block<3, 3>(0, 0));
    q_wb.normalize();
    Twb.pose.pose.orientation.w = q_wb.w();
    Twb.pose.pose.orientation.x = q_wb.x();
    Twb.pose.pose.orientation.y = q_wb.y();
    Twb.pose.pose.orientation.z = q_wb.z();
    Twb.pose.pose.position.x = eig_data_d(0, 3);
    Twb.pose.pose.position.y = eig_data_d(1, 3);
    Twb.pose.pose.position.z = eig_data_d(2, 3);
    Twb.twist.twist.linear.x = eig_data_d(4, 0);
    Twb.twist.twist.linear.y = eig_data_d(4, 1);
    Twb.twist.twist.linear.z = eig_data_d(4, 2);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "RGBD");
    ros::start();

    if(argc != 4)
    {
        cerr << endl << "Usage: rosrun ORB_SLAM3 Stereo path_to_vocabulary path_to_settings do_rectify" << endl;
        ros::shutdown();
        return 1;
    }

    std::thread keyboard_command_process;
    keyboard_command_process = std::thread(command);

    cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
    std::string cam1_topic, cam2_topic;
    if (fsSettings["cam1_topic"].empty() || fsSettings["cam2_topic"].empty())
    {
        std::cerr << " plese provide cam and imu topics' name!!!!" << std::endl;
        return -1;
    }
    else
    {
        fsSettings["cam1_topic"] >> cam1_topic;
        fsSettings["cam2_topic"] >> cam2_topic;
        std::cout << "cam1_topic is : " << cam1_topic << std::endl;
        std::cout << "cam2_topic is : " << cam2_topic << std::endl;
    }
    fsSettings.release();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::STEREO,true);

    ImageGrabber igb(&SLAM);

    stringstream ss(argv[3]);
	ss >> boolalpha >> igb.do_rectify;

    if(igb.do_rectify)
    {      
        // Load settings related to stereo calibration
        cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
        if(!fsSettings.isOpened())
        {
            cerr << "ERROR: Wrong path to settings" << endl;
            return -1;
        }

        cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
        fsSettings["LEFT.K"] >> K_l;
        fsSettings["RIGHT.K"] >> K_r;

        fsSettings["LEFT.P"] >> P_l;
        fsSettings["RIGHT.P"] >> P_r;

        fsSettings["LEFT.R"] >> R_l;
        fsSettings["RIGHT.R"] >> R_r;

        fsSettings["LEFT.D"] >> D_l;
        fsSettings["RIGHT.D"] >> D_r;

        int rows_l = fsSettings["LEFT.height"];
        int cols_l = fsSettings["LEFT.width"];
        int rows_r = fsSettings["RIGHT.height"];
        int cols_r = fsSettings["RIGHT.width"];

        if(K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
                rows_l==0 || rows_r==0 || cols_l==0 || cols_r==0)
        {
            cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
            return -1;
        }

        cv::initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F,igb.M1l,igb.M2l);
        cv::initUndistortRectifyMap(K_r,D_r,R_r,P_r.rowRange(0,3).colRange(0,3),cv::Size(cols_r,rows_r),CV_32F,igb.M1r,igb.M2r);
    }

    ros::NodeHandle nh;

    message_filters::Subscriber<sensor_msgs::Image> left_sub(nh, cam1_topic.c_str(), 1);
    message_filters::Subscriber<sensor_msgs::Image> right_sub(nh, cam2_topic.c_str(), 1);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), left_sub,right_sub);
    sync.registerCallback(boost::bind(&ImageGrabber::GrabStereo,&igb,_1,_2));

    pub_pose = nh.advertise<nav_msgs::Odometry>("orb_pose", 1);
    pub_path = nh.advertise<nav_msgs::Path>("orb_path", 1, true);

    ros::spin();

    // Stop all threads
    SLAM.Shutdown();
    const string kf_file =  "kf_traj.txt";
    const string f_file =  "f_traj.txt";
    SLAM.SaveTrajectoryTUM(f_file);
    SLAM.SaveKeyFrameTrajectoryTUM(kf_file);

    ros::shutdown();

    return 0;
}

void ImageGrabber::GrabStereo(const sensor_msgs::ImageConstPtr& msgLeft,const sensor_msgs::ImageConstPtr& msgRight)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptrLeft;
    try
    {
        cv_ptrLeft = cv_bridge::toCvShare(msgLeft);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_bridge::CvImageConstPtr cv_ptrRight;
    try
    {
        cv_ptrRight = cv_bridge::toCvShare(msgRight);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    if(do_rectify)
    {
        cv::Mat imLeft, imRight;
        cv::remap(cv_ptrLeft->image,imLeft,M1l,M2l,cv::INTER_LINEAR);
        cv::remap(cv_ptrRight->image,imRight,M1r,M2r,cv::INTER_LINEAR);
        cv::Mat Data_TVag = mpSLAM->TrackStereo(imLeft,imRight,cv_ptrLeft->header.stamp.toSec());

        if (!Data_TVag.empty())
        {
            nav_msgs::Odometry Twb;
            Twb.header.frame_id = "world";
            Twb.header.stamp = cv_ptrLeft->header.stamp;
            convertOrbSlamPoseToOdom(Data_TVag, Twb);
            pub_pose.publish(Twb);
            orb_path.header.frame_id = Twb.header.frame_id;
            orb_path.header.stamp = Twb.header.stamp;
            geometry_msgs::PoseStamped tmp_pose;
            tmp_pose.header.frame_id = Twb.header.frame_id;
            tmp_pose.header.stamp = Twb.header.stamp;
            tmp_pose.pose.position = Twb.pose.pose.position;
            tmp_pose.pose.orientation = Twb.pose.pose.orientation;
            orb_path.poses.push_back(tmp_pose);
            pub_path.publish(orb_path);
        }
    }
    else
    {
        cv::Mat Data_TVag = mpSLAM->TrackStereo(cv_ptrLeft->image,cv_ptrRight->image,cv_ptrLeft->header.stamp.toSec());

        if (!Data_TVag.empty())
        {
            nav_msgs::Odometry Twb;
            Twb.header.frame_id = "world";
            Twb.header.stamp = cv_ptrLeft->header.stamp;
            convertOrbSlamPoseToOdom(Data_TVag, Twb);
            pub_pose.publish(Twb);
            orb_path.header.frame_id = Twb.header.frame_id;
            orb_path.header.stamp = Twb.header.stamp;
            geometry_msgs::PoseStamped tmp_pose;
            tmp_pose.header.frame_id = Twb.header.frame_id;
            tmp_pose.header.stamp = Twb.header.stamp;
            tmp_pose.pose.position = Twb.pose.pose.position;
            tmp_pose.pose.orientation = Twb.pose.pose.orientation;
            orb_path.poses.push_back(tmp_pose);
            pub_path.publish(orb_path);
        }
    }

    if (is_stop_dz)
    {
        std::cout << "current tast is ended !!! " << std::endl;
        mpSLAM->Shutdown();
        const string kf_file =  "kf_traj.txt";
        const string f_file =  "f_traj.txt";
        mpSLAM->SaveTrajectoryTUM(f_file);
        mpSLAM->SaveKeyFrameTrajectoryTUM(kf_file);
        exit(0);
    }

}


