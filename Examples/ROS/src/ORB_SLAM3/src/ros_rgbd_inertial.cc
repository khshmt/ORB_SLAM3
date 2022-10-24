/**
* This file is part of ORB-SLAM3
*
* Copydepth (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copydepth (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
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
float shift = 0.;

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

class ImuGrabber
{
public:
    ImuGrabber(){};
    void GrabImu(const sensor_msgs::ImuConstPtr &imu_msg);

    queue<sensor_msgs::ImuConstPtr> imuBuf;
    std::mutex mBufMutex;
};

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM3::System* pSLAM, ImuGrabber *pImuGb, const bool bRect, const bool bClahe): mpSLAM(pSLAM), mpImuGb(pImuGb), do_rectify(bRect), mbClahe(bClahe){}

    void GrabImageRgb(const sensor_msgs::ImageConstPtr& msg);
    void GrabImageDepth(const sensor_msgs::ImageConstPtr& msg);
    cv::Mat GetImage(const sensor_msgs::ImageConstPtr &img_msg);
    void SyncWithImu();

    queue<sensor_msgs::ImageConstPtr> imgRgbBuf, imgDepthBuf;
    std::mutex mBufMutexRgb,mBufMutexDepth;

    ORB_SLAM3::System* mpSLAM;
    ImuGrabber *mpImuGb;

    const bool do_rectify;
    cv::Mat M1l,M2l,M1r,M2r;

    const bool mbClahe;
    cv::Ptr<cv::CLAHE> mClahe = cv::createCLAHE(3.0, cv::Size(8, 8));
};



int main(int argc, char **argv)
{
    ros::init(argc, argv, "RGBD_Inertial");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    bool bEqual = false;
    bool bView = false;
    if(argc < 4 || argc > 6)
    {
        cerr << endl << "Usage: rosrun ORB_SLAM3 RGBD_inertial path_to_vocabulary path_to_settings [do_equalize(false)] [enable_view(false)]" << endl;
        ros::shutdown();
        return 1;
    }

    std::string sbRect(argv[3]);
    if(argc==5)
    {
        std::string sbEqual(argv[4]);
        if(sbEqual == "true")
            bEqual = true;
    }

    if (argc == 6)
    {
        std::string sbView(argv[5]);
        if (sbView == "true")
            bView = true;
    }

    std::thread keyboard_command_process;
    keyboard_command_process = std::thread(command);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::IMU_RGBD,bView);

    ImuGrabber imugb;
    ImageGrabber igb(&SLAM,&imugb,sbRect == "true",bEqual);

    cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        cerr << "ERROR: Wrong path to settings" << endl;
        return -1;
    }

    shift = fsSettings["time_shift"];
    std::cout << "time_shift is : " << shift << std::endl;
    std::string imu_topic, color_topic, depth_topic;

    if (fsSettings["imu_topic"].empty() || fsSettings["color_topic"].empty() || fsSettings["depth_topic"].empty())
    {
        std::cerr << " plese provide cam and imu topics' name!!!!" << std::endl;
        return -1;
    }
    else
    {
        fsSettings["imu_topic"] >> imu_topic;
        fsSettings["color_topic"] >> color_topic;
        fsSettings["depth_topic"] >> depth_topic;
        std::cout << "imu_topic is : " << imu_topic << std::endl;
        std::cout << "color_topic is : " << color_topic << std::endl;
        std::cout << "depth_topic is : " << depth_topic << std::endl;
    }
    fsSettings.release();

    // Maximum delay, 5 seconds
    ros::Subscriber sub_imu = n.subscribe(imu_topic.c_str(), 1000, &ImuGrabber::GrabImu, &imugb);
    ros::Subscriber sub_img_rgb = n.subscribe(color_topic.c_str(), 100, &ImageGrabber::GrabImageRgb,&igb);
    ros::Subscriber sub_img_depth = n.subscribe(depth_topic.c_str(), 100, &ImageGrabber::GrabImageDepth,&igb);

    pub_pose = n.advertise<nav_msgs::Odometry>("orb_pose", 1);
    pub_path = n.advertise<nav_msgs::Path>("orb_path", 1, true);

    std::thread sync_thread(&ImageGrabber::SyncWithImu,&igb);

    ros::spin();

    return 0;
}



void ImageGrabber::GrabImageRgb(const sensor_msgs::ImageConstPtr &img_msg)
{
    mBufMutexRgb.lock();
    if (!imgRgbBuf.empty())
        imgRgbBuf.pop();
    imgRgbBuf.push(img_msg);
    mBufMutexRgb.unlock();
}

void ImageGrabber::GrabImageDepth(const sensor_msgs::ImageConstPtr &img_msg)
{
    mBufMutexDepth.lock();
    if (!imgDepthBuf.empty())
        imgDepthBuf.pop();
    imgDepthBuf.push(img_msg);
    mBufMutexDepth.unlock();
}

cv::Mat ImageGrabber::GetImage(const sensor_msgs::ImageConstPtr &img_msg)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvShare(img_msg);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }

    return cv_ptr->image.clone();
}

void ImageGrabber::SyncWithImu()
{
    const double maxTimeDiff = 0.01;
    while(1)
    {
        cv::Mat imRgb, imDepth;
        double tImRgb = 0, tImDepth = 0;
        if (!imgRgbBuf.empty()&&!imgDepthBuf.empty()&&!mpImuGb->imuBuf.empty())
        {
            tImRgb = imgRgbBuf.front()->header.stamp.toSec();
            tImDepth = imgDepthBuf.front()->header.stamp.toSec();

            this->mBufMutexDepth.lock();
            while((tImRgb-tImDepth)>maxTimeDiff && imgDepthBuf.size()>1)
            {
                imgDepthBuf.pop();
                tImDepth = imgDepthBuf.front()->header.stamp.toSec();
            }
            this->mBufMutexDepth.unlock();

            this->mBufMutexRgb.lock();
            while((tImDepth-tImRgb)>maxTimeDiff && imgRgbBuf.size()>1)
            {
                imgRgbBuf.pop();
                tImRgb = imgRgbBuf.front()->header.stamp.toSec();
            }
            this->mBufMutexRgb.unlock();

            if((tImRgb-tImDepth)>maxTimeDiff || (tImDepth-tImRgb)>maxTimeDiff)
            {
                // std::cout << "big time difference" << std::endl;
                std::chrono::milliseconds tSleep(2);
                std::this_thread::sleep_for(tSleep);
                continue;
            }

            //更新正确的图像时间，将图像时间对齐到IMU时间戳上.
            tImRgb = tImRgb + shift;
            if(tImRgb>mpImuGb->imuBuf.back()->header.stamp.toSec())
                continue;

            this->mBufMutexRgb.lock();
            imRgb = GetImage(imgRgbBuf.front());
            imgRgbBuf.pop();
            this->mBufMutexRgb.unlock();

            this->mBufMutexDepth.lock();
            imDepth = GetImage(imgDepthBuf.front());
            imgDepthBuf.pop();
            this->mBufMutexDepth.unlock();

            vector<ORB_SLAM3::IMU::Point> vImuMeas;
            mpImuGb->mBufMutex.lock();
            Eigen::Vector3d angel_vel = Eigen::Vector3d::Zero();
            Eigen::Vector3d acc_dz = Eigen::Vector3d::Zero();
            if(!mpImuGb->imuBuf.empty())
            {
                // Load imu measurements from buffer
                vImuMeas.clear();
                while(!mpImuGb->imuBuf.empty() && mpImuGb->imuBuf.front()->header.stamp.toSec()<=tImRgb)
                {
                    double t = mpImuGb->imuBuf.front()->header.stamp.toSec();
                    cv::Point3f acc(mpImuGb->imuBuf.front()->linear_acceleration.x, mpImuGb->imuBuf.front()->linear_acceleration.y, mpImuGb->imuBuf.front()->linear_acceleration.z);
                    cv::Point3f gyr(mpImuGb->imuBuf.front()->angular_velocity.x, mpImuGb->imuBuf.front()->angular_velocity.y, mpImuGb->imuBuf.front()->angular_velocity.z);
                    vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc,gyr,t));
                    angel_vel.x() = gyr.x; angel_vel.y() = gyr.y; angel_vel.z() = gyr.z;
                    acc_dz.x() = acc.x; acc_dz.y() = acc.y; acc_dz.z() = acc.z;
                    mpImuGb->imuBuf.pop();
                }
            }
            mpImuGb->mBufMutex.unlock();
            if(mbClahe)
            {
                mClahe->apply(imRgb,imRgb);
                mClahe->apply(imDepth,imDepth);
            }

            // if(do_rectify)
            // {
            //   cv::remap(imRgb,imRgb,M1l,M2l,cv::INTER_LINEAR);
            //   cv::remap(imDepth,imDepth,M1r,M2r,cv::INTER_LINEAR);
            // }

            cv::Mat Data_TVag = mpSLAM->TrackRGBD(imRgb,imDepth,tImRgb,vImuMeas);
            if (!Data_TVag.empty())
            {
                nav_msgs::Odometry Twb;
                Twb.header.frame_id = "world";
                Twb.header.stamp.fromSec(tImRgb);
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

//            std::chrono::milliseconds tSleep(1);
//            std::this_thread::sleep_for(tSleep);
        } else
        {
            std::chrono::milliseconds tSleep(2);
            std::this_thread::sleep_for(tSleep);
        }

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

void ImuGrabber::GrabImu(const sensor_msgs::ImuConstPtr &imu_msg)
{
    mBufMutex.lock();
    imuBuf.push(imu_msg);
    mBufMutex.unlock();
    return;
}
