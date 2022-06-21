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
#include<vector>
#include<queue>
#include<thread>
#include<mutex>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>
#include "nav_msgs/Odometry.h"
#include <Eigen/Dense>

#include "opencv2/core/eigen.hpp"
#include <opencv2/core/core.hpp>

#include "../../../include/System.h"
#include "../include/ImuTypes.h"

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
Eigen::Vector3d ba, bg;
float shift = 0;

bool is_stop_dz = false;

void command() {
    while (1)
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

class ImuGrabber
{
public:
    ImuGrabber() {};

    void GrabImu(const sensor_msgs::ImuConstPtr &imu_msg);

    queue<sensor_msgs::ImuConstPtr> imuBuf;
    std::mutex mBufMutex;
};

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM3::System *pSLAM, ImuGrabber *pImuGb, const bool bClahe) : mpSLAM(pSLAM), mpImuGb(pImuGb),
                                                                                    mbClahe(bClahe) {}

    void GrabImage(const sensor_msgs::ImageConstPtr &msg);

    cv::Mat GetImage(const sensor_msgs::ImageConstPtr &img_msg);

    void SyncWithImu();

    queue<sensor_msgs::ImageConstPtr> img0Buf;
    std::mutex mBufMutex;

    ORB_SLAM3::System *mpSLAM;
    ImuGrabber *mpImuGb;

    const bool mbClahe;
    cv::Ptr<cv::CLAHE> mClahe = cv::createCLAHE(3.0, cv::Size(8, 8));
};

void convertOrbSlamPoseToOdom(const cv::Mat &cv_data, nav_msgs::Odometry &Twb, Eigen::Vector3d ang_vel,
                              Eigen::Vector3d acc_) {

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
    Twb.twist.twist.angular.x = ang_vel.x();
    Twb.twist.twist.angular.y = ang_vel.y();
    Twb.twist.twist.angular.z = ang_vel.z();

    Twb.twist.covariance[0] = acc_.x();
    Twb.twist.covariance[1] = acc_.y();
    Twb.twist.covariance[2] = acc_.z();

    ba = eig_data_d.block<1, 3>(5, 0);
    bg = eig_data_d.block<1, 3>(6, 0);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "Mono_Inertial");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    bool bEqual = false;
    if (argc < 3 || argc > 4)
    {
        cerr << endl << "Usage: rosrun ORB_SLAM3 Mono_Inertial path_to_vocabulary path_to_settings [do_equalize]"
             << endl;
        ros::shutdown();
        return 1;
    }


    if (argc == 4)
    {
        std::string sbEqual(argv[3]);
        if (sbEqual == "true")
            bEqual = true;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::IMU_MONOCULAR, true);

    std::thread keyboard_command_process;
    keyboard_command_process = std::thread(command);

    cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
    shift = fsSettings["time_shift"];
    std::cout << "time_shift is : " << shift << std::endl;
    std::string imu_topic, cam_topic;
    if (fsSettings["imu_topic"].empty() || fsSettings["cam_topic"].empty())
    {
        std::cerr << " plese provide cam and imu topics' name!!!!" << std::endl;
        return -1;
    } else
    {
        fsSettings["imu_topic"] >> imu_topic;
        fsSettings["cam_topic"] >> cam_topic;
        std::cout << "imu_topic is : " << imu_topic << std::endl;
        std::cout << "cam_topic is : " << cam_topic << std::endl;
    }
    fsSettings.release();

    ImuGrabber imugb;
    ImageGrabber igb(&SLAM, &imugb, bEqual); // TODO

    // Maximum delay, 5 seconds
    ros::Subscriber sub_imu = n.subscribe(imu_topic.c_str(), 1000, &ImuGrabber::GrabImu, &imugb);
    ros::Subscriber sub_img0 = n.subscribe(cam_topic.c_str(), 100, &ImageGrabber::GrabImage, &igb);

    pub_pose = n.advertise<nav_msgs::Odometry>("orb_pose", 1);
    pub_path = n.advertise<nav_msgs::Path>("orb_path", 1, true);

    std::thread sync_thread(&ImageGrabber::SyncWithImu, &igb);

    ros::spin();

    return 0;
}

void ImageGrabber::GrabImage(const sensor_msgs::ImageConstPtr &img_msg) {
    mBufMutex.lock();
    if (!img0Buf.empty())
        img0Buf.pop();
    img0Buf.push(img_msg);
    mBufMutex.unlock();
}

cv::Mat ImageGrabber::GetImage(const sensor_msgs::ImageConstPtr &img_msg) {
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvShare(img_msg, sensor_msgs::image_encodings::MONO8);
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }

    if (cv_ptr->image.type() == 0)
    {
        return cv_ptr->image.clone();
    } else
    {
        std::cout << "Error type" << std::endl;
        return cv_ptr->image.clone();
    }
}

void ImageGrabber::SyncWithImu() {
    while (1)
    {
        cv::Mat im;
        double tIm = 0;
        if (!img0Buf.empty() && !mpImuGb->imuBuf.empty())
        {
            tIm = img0Buf.front()->header.stamp.toSec();
            tIm = tIm + shift;
            if (tIm > mpImuGb->imuBuf.back()->header.stamp.toSec())
                continue;
            {
                this->mBufMutex.lock();
                im = GetImage(img0Buf.front());
                img0Buf.pop();
                this->mBufMutex.unlock();
            }

//            tIm = tIm + shift;

            vector<ORB_SLAM3::IMU::Point> vImuMeas;
            mpImuGb->mBufMutex.lock();
            Eigen::Vector3d angel_vel = Eigen::Vector3d::Zero();
            Eigen::Vector3d acc_dz = Eigen::Vector3d::Zero();
            if (!mpImuGb->imuBuf.empty())
            {
                // Load imu measurements from buffer
                vImuMeas.clear();
                while (!mpImuGb->imuBuf.empty() && mpImuGb->imuBuf.front()->header.stamp.toSec() <= tIm)
                {
                    double t = mpImuGb->imuBuf.front()->header.stamp.toSec();
                    cv::Point3f acc(mpImuGb->imuBuf.front()->linear_acceleration.x,
                                    mpImuGb->imuBuf.front()->linear_acceleration.y,
                                    mpImuGb->imuBuf.front()->linear_acceleration.z);
                    cv::Point3f gyr(mpImuGb->imuBuf.front()->angular_velocity.x,
                                    mpImuGb->imuBuf.front()->angular_velocity.y,
                                    mpImuGb->imuBuf.front()->angular_velocity.z);
                    vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc, gyr, t));
                    angel_vel.x() = gyr.x; angel_vel.y() = gyr.y; angel_vel.z() = gyr.z;
                    acc_dz.x() = acc.x; acc_dz.y() = acc.y; acc_dz.z() = acc.z;
                    mpImuGb->imuBuf.pop();
                }
            }
            mpImuGb->mBufMutex.unlock();
            if (mbClahe)
                mClahe->apply(im, im);

            cv::Mat Data_TVag = mpSLAM->TrackMonocular(im, tIm, vImuMeas);
            if (!Data_TVag.empty())
            {
                nav_msgs::Odometry Twb;
                Twb.header.frame_id = "world";
                Twb.header.stamp.fromSec(tIm);
                convertOrbSlamPoseToOdom(Data_TVag, Twb, angel_vel, acc_dz);
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

        std::chrono::milliseconds tSleep(1);
        std::this_thread::sleep_for(tSleep);

        if (is_stop_dz)
            break;
    }

    mpSLAM->Shutdown();
    const string kf_file =  "kf_traj.txt";
    const string f_file =  "f_traj.txt";
    mpSLAM->SaveTrajectoryTUM(f_file);
    mpSLAM->SaveKeyFrameTrajectoryTUM(kf_file);
    exit(0);
}

void ImuGrabber::GrabImu(const sensor_msgs::ImuConstPtr &imu_msg) {
    mBufMutex.lock();
    imuBuf.push(imu_msg);
    mBufMutex.unlock();
    return;
}


