/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
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
    ImageGrabber(ORB_SLAM3::System *pSLAM, ImuGrabber *pImuGb, const bool bRect, const bool bClahe) : mpSLAM(pSLAM),
                                                                                                      mpImuGb(pImuGb),
                                                                                                      do_rectify(bRect),
                                                                                                      mbClahe(bClahe) {}

    void GrabImageLeft(const sensor_msgs::ImageConstPtr &msg);

    void GrabImageRight(const sensor_msgs::ImageConstPtr &msg);

    cv::Mat GetImage(const sensor_msgs::ImageConstPtr &img_msg);

    void SyncWithImu();

    queue<sensor_msgs::ImageConstPtr> imgLeftBuf, imgRightBuf;
    std::mutex mBufMutexLeft, mBufMutexRight;

    ORB_SLAM3::System *mpSLAM;
    ImuGrabber *mpImuGb;

    const bool do_rectify;
    cv::Mat M1l, M2l, M1r, M2r;

    const bool mbClahe;
    cv::Ptr<cv::CLAHE> mClahe = cv::createCLAHE(3.0, cv::Size(8, 8));
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

    ba = eig_data_d.block<1, 3>(5, 0);
    bg = eig_data_d.block<1, 3>(6, 0);
}

void convertOrbSlamPoseToOdom2(const cv::Mat &T_cv, geometry_msgs::PoseStamped &Twc) {
    assert(T_cv.rows == 4);
    Eigen::Matrix4f Tcw_f;
    cv::cv2eigen(T_cv, Tcw_f);
    Eigen::Matrix4d Tcw_d = Tcw_f.cast<double>();
    Eigen::Quaterniond q_cw(Tcw_d.block<3, 3>(0, 0));
    q_cw.normalize();
    Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d Twc_ = Eigen::Isometry3d::Identity();
    Tcw.rotate(q_cw);
    Tcw.pretranslate(Tcw_d.block<3, 1>(0, 3));
    Twc_ = Tcw.inverse();
    Eigen::Quaterniond q_wc(Twc_.rotation());
    Eigen::Translation3d t_wc(Twc_.translation());
    Twc.pose.orientation.w = q_wc.w();
    Twc.pose.orientation.x = q_wc.x();
    Twc.pose.orientation.y = q_wc.y();
    Twc.pose.orientation.z = q_wc.z();
    Twc.pose.position.x = t_wc.x();
    Twc.pose.position.y = t_wc.y();
    Twc.pose.position.z = t_wc.z();
}


int main(int argc, char **argv) {
    ros::init(argc, argv, "Stereo_Inertial");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    bool bEqual = false;
    if (argc < 4 || argc > 5)
    {
        cerr << endl
             << "Usage: rosrun ORB_SLAM3 Stereo_Inertial path_to_vocabulary path_to_settings do_rectify [do_equalize]"
             << endl;
        ros::shutdown();
        return 1;
    }

    std::string sbRect(argv[3]);
    if (argc == 5)
    {
        std::string sbEqual(argv[4]);
        if (sbEqual == "true")
            bEqual = true;
    }

    std::thread keyboard_command_process;
    keyboard_command_process = std::thread(command);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::IMU_STEREO, false);

    ImuGrabber imugb;
    ImageGrabber igb(&SLAM, &imugb, sbRect == "true", bEqual);
    if (igb.do_rectify)
    {
        // Load settings related to stereo calibration
        cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
        if (!fsSettings.isOpened())
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

        if (K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() ||
            D_r.empty() ||
            rows_l == 0 || rows_r == 0 || cols_l == 0 || cols_r == 0)
        {
            cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
            return -1;
        }

        cv::initUndistortRectifyMap(K_l, D_l, R_l, P_l.rowRange(0, 3).colRange(0, 3), cv::Size(cols_l, rows_l), CV_32F,
                                    igb.M1l, igb.M2l);
        cv::initUndistortRectifyMap(K_r, D_r, R_r, P_r.rowRange(0, 3).colRange(0, 3), cv::Size(cols_r, rows_r), CV_32F,
                                    igb.M1r, igb.M2r);
    }
    cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
    shift = fsSettings["time_shift"];
    std::cout << "time_shift is : " << shift << std::endl;
    std::string imu_topic, cam1_topic, cam2_topic;
    if (fsSettings["imu_topic"].empty() || fsSettings["cam1_topic"].empty() || fsSettings["cam2_topic"].empty())
    {
        std::cerr << " plese provide cam and imu topics' name!!!!" << std::endl;
        return -1;
    }
    else
    {
        fsSettings["imu_topic"] >> imu_topic;
        fsSettings["cam1_topic"] >> cam1_topic;
        fsSettings["cam2_topic"] >> cam2_topic;
        std::cout << "imu_topic is : " << imu_topic << std::endl;
        std::cout << "cam1_topic is : " << cam1_topic << std::endl;
        std::cout << "cam2_topic is : " << cam2_topic << std::endl;
    }
    fsSettings.release();
    // Maximum delay, 5 seconds
    ros::Subscriber sub_imu = n.subscribe(imu_topic.c_str(), 1000, &ImuGrabber::GrabImu, &imugb);
    ros::Subscriber sub_img_left = n.subscribe(cam1_topic.c_str(), 100, &ImageGrabber::GrabImageLeft,
                                               &igb);
    ros::Subscriber sub_img_right = n.subscribe(cam2_topic.c_str(), 100, &ImageGrabber::GrabImageRight,
                                                &igb);

    pub_pose = n.advertise<nav_msgs::Odometry>("orb_pose", 1);
    pub_path = n.advertise<nav_msgs::Path>("orb_path", 1, true);

    std::thread sync_thread(&ImageGrabber::SyncWithImu, &igb);

    ros::spin();

    return 0;
}


void ImageGrabber::GrabImageLeft(const sensor_msgs::ImageConstPtr &img_msg) {
    mBufMutexLeft.lock();
    if (!imgLeftBuf.empty())
        imgLeftBuf.pop();
    imgLeftBuf.push(img_msg);
    mBufMutexLeft.unlock();
}

void ImageGrabber::GrabImageRight(const sensor_msgs::ImageConstPtr &img_msg) {
    mBufMutexRight.lock();
    if (!imgRightBuf.empty())
        imgRightBuf.pop();
    imgRightBuf.push(img_msg);
    mBufMutexRight.unlock();
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
    const double maxTimeDiff = 0.01;
    while (1)
    {
        nav_msgs::Odometry Twb;
        Twb.header.frame_id = "world";

        cv::Mat imLeft, imRight;
        double tImLeft = 0, tImRight = 0;
        if (!imgLeftBuf.empty() && !imgRightBuf.empty() && !mpImuGb->imuBuf.empty())
        {
            tImLeft = imgLeftBuf.front()->header.stamp.toSec();
            tImRight = imgRightBuf.front()->header.stamp.toSec();

            this->mBufMutexRight.lock();
            while ((tImLeft - tImRight) > maxTimeDiff && imgRightBuf.size() > 1)
            {
                imgRightBuf.pop();
                tImRight = imgRightBuf.front()->header.stamp.toSec();
            }
            this->mBufMutexRight.unlock();

            this->mBufMutexLeft.lock();
            while ((tImRight - tImLeft) > maxTimeDiff && imgLeftBuf.size() > 1)
            {
                imgLeftBuf.pop();
                tImLeft = imgLeftBuf.front()->header.stamp.toSec();
            }
            this->mBufMutexLeft.unlock();

            if ((tImLeft - tImRight) > maxTimeDiff || (tImRight - tImLeft) > maxTimeDiff)
            {
                // std::cout << "big time difference" << std::endl;
                continue;
            }
            if (tImLeft > mpImuGb->imuBuf.back()->header.stamp.toSec())
                continue;

//            Twb.header.stamp = imgLeftBuf.front()->header.stamp;
            tImLeft = tImLeft + shift;
            Twb.header.stamp.fromSec(tImLeft);

            this->mBufMutexLeft.lock();
            imLeft = GetImage(imgLeftBuf.front());
            imgLeftBuf.pop();
            this->mBufMutexLeft.unlock();

            this->mBufMutexRight.lock();
            imRight = GetImage(imgRightBuf.front());
            imgRightBuf.pop();
            this->mBufMutexRight.unlock();

            vector<ORB_SLAM3::IMU::Point> vImuMeas;
            mpImuGb->mBufMutex.lock();
            if (!mpImuGb->imuBuf.empty())
            {
                // Load imu measurements from buffer
                vImuMeas.clear();
                while (!mpImuGb->imuBuf.empty() && mpImuGb->imuBuf.front()->header.stamp.toSec() <= tImLeft)
                {
                    double t = mpImuGb->imuBuf.front()->header.stamp.toSec();
                    cv::Point3f acc(mpImuGb->imuBuf.front()->linear_acceleration.x,
                                    mpImuGb->imuBuf.front()->linear_acceleration.y,
                                    mpImuGb->imuBuf.front()->linear_acceleration.z);
                    cv::Point3f gyr(mpImuGb->imuBuf.front()->angular_velocity.x,
                                    mpImuGb->imuBuf.front()->angular_velocity.y,
                                    mpImuGb->imuBuf.front()->angular_velocity.z);
                    vImuMeas.push_back(ORB_SLAM3::IMU::Point(acc, gyr, t));
                    mpImuGb->imuBuf.pop();
                }
            }
            mpImuGb->mBufMutex.unlock();
            if (mbClahe)
            {
                mClahe->apply(imLeft, imLeft);
                mClahe->apply(imRight, imRight);
            }

            if (do_rectify)
            {
                cv::remap(imLeft, imLeft, M1l, M2l, cv::INTER_LINEAR);
                cv::remap(imRight, imRight, M1r, M2r, cv::INTER_LINEAR);
            }

            cv::Mat Data_TVag = mpSLAM->TrackStereo(imLeft, imRight, tImLeft, vImuMeas);
            if (!Data_TVag.empty())
            {
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

/*            orb_path.header.frame_id = Twb.header.frame_id;
            orb_path.header.stamp = Twb.header.stamp;
            geometry_msgs::PoseStamped tmp_pose;
            tmp_pose.header.frame_id = Twb.header.frame_id;
            tmp_pose.header.stamp = Twb.header.stamp;
            convertOrbSlamPoseToOdom2(Data_TVag, tmp_pose);
            orb_path.poses.push_back(tmp_pose);
            pub_path.publish(orb_path);*/
            }

            std::chrono::milliseconds tSleep(1);
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

void ImuGrabber::GrabImu(const sensor_msgs::ImuConstPtr &imu_msg) {
    mBufMutex.lock();
    imuBuf.push(imu_msg);
    mBufMutex.unlock();
    return;
}


