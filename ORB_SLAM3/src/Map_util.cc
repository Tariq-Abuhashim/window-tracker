
/**
* Tariq updated - Jun, 2022
* Tariq updated - Mar, 2023
**/

#include "Converter.h"
#include "Map.h"
#include <mutex>

namespace ORB_SLAM3
{

void Map::AddMapObject(MapObject *pMO)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapObjects.insert(pMO);
}

void Map::EraseMapObject(MapObject *pMO)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapObjects.erase(pMO);
}

vector<MapObject*> Map::GetAllMapObjects()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<MapObject*>(mspMapObjects.begin(), mspMapObjects.end());
}

MapObject* Map::GetMapObject(int object_id)
{
    unique_lock<mutex> lock(mMutexMap);
    for (auto mspMapObject : mspMapObjects)
    {
        if(mspMapObject->mnId != object_id)
            continue;
        return mspMapObject;
    }
    return NULL;
}

void Map::ApplyScaledRotationWithObjects(const Eigen::Matrix3f &R, 
										const float s, const bool bScaledVel, 
										const Eigen::Vector3f &t)
{
    unique_lock<mutex> lock(mMutexMap);

    // Body position (IMU) of first keyframe is fixed to (0,0,0)
    /*
    cv::Mat Txw = cv::Mat::eye(4,4,CV_32F);
    R.copyTo(Txw.rowRange(0,3).colRange(0,3));

    cv::Mat Tyx = cv::Mat::eye(4,4,CV_32F);

    cv::Mat Tyw = Tyx*Txw;
    Tyw.rowRange(0,3).col(3) = Tyw.rowRange(0,3).col(3)+t;
    cv::Mat Ryw = Tyw.rowRange(0,3).colRange(0,3);
    cv::Mat tyw = Tyw.rowRange(0,3).col(3);
    */
    
    Eigen::Matrix4f Tyw = Eigen::Matrix4f::Identity();
	Tyw.topLeftCorner<3,3>() = R;
	Tyw.topRightCorner<3,1>() = t;
	Eigen::Matrix3f Ryw = R;
	Eigen::Vector3f tyw = t;
	
	// KeyFrames
    for(set<KeyFrame*>::iterator sit=mspKeyFrames.begin(); sit!=mspKeyFrames.end(); sit++)
    {
        KeyFrame* pKF = *sit;
        Sophus::SE3f Twc = pKF->GetPoseInverse();  // was cv::Mat
        // Scale translation part of Twc
        Eigen::Matrix4f Twc_mat = Twc.matrix();
        Twc_mat.block<3,1>(0,3) *= s;
        Twc = Sophus::SE3f(Twc_mat.block<3,3>(0,0), Twc_mat.block<3,1>(0,3));
        // Apply Tyw
        Eigen::Matrix4f Tyc = Tyw * Twc.matrix();
        // Convert Tyc to Tcy = inverse of Tyc
        Eigen::Matrix3f Rcy = Tyc.block<3,3>(0,0).transpose();
        Eigen::Vector3f tcy = -Rcy * Tyc.block<3,1>(0,3);
        Sophus::SE3f Tcy(Rcy, tcy);
        
        pKF->SetPose(Tcy);
        
        Eigen::Vector3f Vw = pKF->GetVelocity();
        if(!bScaledVel)
            pKF->SetVelocity(Ryw*Vw);
        else
            pKF->SetVelocity(Ryw*Vw*s);

    }
    
	// MapPoints
    for(set<MapPoint*>::iterator sit=mspMapPoints.begin(); sit!=mspMapPoints.end(); sit++)
    {
        MapPoint* pMP = *sit;
        pMP->SetWorldPos(s*Ryw*pMP->GetWorldPos()+tyw);
        pMP->UpdateNormalAndDepth();
    }

/*  
	// MapObjects
    for(set<MapObject*>::iterator sit=mspMapObjects.begin(); sit!=mspMapObjects.end(); sit++)
    {
        MapObject* pMO = *sit;
		Eigen::Matrix4f SE3Two = pMO->GetPoseSE3();
		Two.block<3,1>(0,3) *= s;
		Eigen::Matrix4f Tyo = Tyw * Two;
		pMO->SetObjectPoseSE3(Tyo);
    }
*/
    mnMapChange++;
}

}

