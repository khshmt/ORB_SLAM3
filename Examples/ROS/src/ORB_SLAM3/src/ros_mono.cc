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

    void GrabImage(const sensor_msgs::ImageConstPtr& msg);

    ORB_SLAM3::System* mpSLAM;
};

void convertOrbSlamPoseToOdom(const cv::Mat &cv_data, nav_msgs::Odometry &Twb) {

    assert(cv_data.rows == 7);
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
    ros::init(argc, argv, "Mono");
    ros::start();

    if(argc != 3)
    {
        cerr << endl << "Usage: rosrun ORB_SLAM3 Mono path_to_vocabulary path_to_settings" << endl;        
        ros::shutdown();
        return 1;
    }

    std::thread keyboard_command_process;
    keyboard_command_process = std::thread(command);

    cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
    std::string cam_topic;
    if (fsSettings["cam_topic"].empty())
    {
        std::cerr << " plese provide cam and imu topics' name!!!!" << std::endl;
        return -1;
    }
    else
    {
        fsSettings["cam_topic"] >> cam_topic;
        std::cout << "cam_topic is : " << cam_topic << std::endl;
    }
    fsSettings.release();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::MONOCULAR,true);

    ImageGrabber igb(&SLAM);

    ros::NodeHandle nodeHandler;
    ros::Subscriber sub = nodeHandler.subscribe(cam_topic.c_str(), 1, &ImageGrabber::GrabImage,&igb);

    pub_pose = nodeHandler.advertise<nav_msgs::Odometry>("orb_pose", 1);
    pub_path = nodeHandler.advertise<nav_msgs::Path>("orb_path", 1, true);

    ros::spin();

    // Stop all threads
/*    SLAM.Shutdown();
    const string kf_file =  "kf_traj.txt";
    const string f_file =  "f_traj.txt";
    SLAM.SaveTrajectoryTUM(f_file);
    SLAM.SaveKeyFrameTrajectoryTUM(kf_file);*/

    ros::shutdown();

    return 0;
}

void ImageGrabber::GrabImage(const sensor_msgs::ImageConstPtr& msg)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvShare(msg);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv::Mat Data_TVag = mpSLAM->TrackMonocular(cv_ptr->image,cv_ptr->header.stamp.toSec());

    if (!Data_TVag.empty())
    {
        nav_msgs::Odometry Twb;
        Twb.header.frame_id = "world";
        Twb.header.stamp = cv_ptr->header.stamp;
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


