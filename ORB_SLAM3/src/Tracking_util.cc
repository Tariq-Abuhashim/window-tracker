
/**
* updated - Jun, 2022
**/

#include "Tracking.h"
#include "ObjectDetection.h"
#include "ORBmatcher.h"
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

#include "Converter.h"

using namespace std;

namespace ORB_SLAM3 {

/*
 * Tracking utils for stereo+lidar on KITTI
 */
void Tracking::GetObjectDetectionsLiDAR(KeyFrame *pKF) {

    PyThreadStateLock PyThreadLock;
    int count = 0;
	py::list detections = mpSystem->pySequence.attr("get_frame_by_id")(pKF->mnFrameId);
	for (auto det : detections) {
		auto pts = det.attr("surface_points").cast<Eigen::MatrixXf>();
		auto Sim3Tco = det.attr("T_cam_obj").cast<Eigen::Matrix4f>();
		auto rays = det.attr("rays");
		auto box = det.attr("scale").cast<Eigen::Vector3f>(); // TODO new update, use point-pillar bounding box
		Eigen::MatrixXf rays_mat;
		Eigen::VectorXf depth;

		if (rays.is_none()) {
			//std::cout << "No 2D masks associated!" << std::endl;
			rays_mat = Eigen::Matrix<float, 0, 0>::Zero();
			depth = Eigen::VectorXf::Zero(0); // DSP changes, this was Eigen::Vector<float, 0>::Zero() 
		} else {
			rays_mat = rays.cast<Eigen::MatrixXf>();
			depth = det.attr("depth").cast<Eigen::VectorXf>();
		count++;
		}
        		
		// Create C++ detection instance
		auto o = new ObjectDetection(Sim3Tco, pts, rays_mat, depth, box);
		//auto o = std::make_shared<ObjectDetection>(Sim3Tco, pts, rays_mat, depth, box);
		pKF->mvpDetectedObjects.push_back(o);
	}
	pKF->nObj = pKF->mvpDetectedObjects.size();
	pKF->mvpMapObjects = vector<MapObject *>(pKF->nObj, static_cast<MapObject *>(NULL));
	//std::cout << "TRACK_UTIL: KF" << pKF->mnId << " detections: " << pKF->nObj
	//          << " 3D cuboids + " << count << " 2D masks." << std::endl;
}

void Tracking::ObjectDataAssociation(KeyFrame *pKF)
{
    vector<MapObject *> vpLocalMapObjects;
    vpLocalMapObjects.reserve(mvpLocalKeyFrames.size()); // Pre-reserve memory
    // Loop over all the local frames to find matches
    for (KeyFrame *plKF : mvpLocalKeyFrames)
    {
        std::vector<MapObject *> vpMOs = plKF->GetMapObjectMatches();
        for (MapObject *pMO : vpMOs)
        {
            if (pMO)
            {
                // Prevent multiple association to the same object
                if (pMO->mnAssoRefID != pKF->mnId)
                {
                    vpLocalMapObjects.push_back(pMO);
                    pMO->mnAssoRefID = pKF->mnId;
                }
            }
        }
    }
    if (vpLocalMapObjects.empty())
        return;

    Sophus::SE3<float> Tcw = mCurrentFrame.GetPose(); // return mTcw (mCurrentFrame.mTcw is private)
    Eigen::Matrix3f Rcw = Tcw.so3().matrix();
    Eigen::Vector3f tcw = Tcw.translation(); 
    auto vDetections = pKF->mvpDetectedObjects;
    // loop over all the detections.
    for (int i = 0; i < pKF->nObj; i++)
    {
        auto det = vDetections[i];
        const Eigen::Vector3f transDet = det->tco;
        
        std::vector<float> dist;
		dist.reserve(vpLocalMapObjects.size());
		
		auto computeDist = [&](const Eigen::Vector3f& objPos) {
    		Eigen::Vector3f d3 = Rcw * objPos + tcw - transDet;
    		return Eigen::Vector2f(d3[0], d3[2]).norm();
		};
		
        for (auto pObj : vpLocalMapObjects)
        {
            if (!pObj || pObj->isBad()) {
                dist.push_back(1000.0f);
                continue;
            }

            if (pObj->isDynamic()) {
				float deltaT = (float)(mCurrentFrame.mnId - mpLastKeyFrame->mnFrameId);
                dist.push_back(computeDist(pObj->two+pObj->velocity*deltaT));
            }
            else {
            	dist.push_back(computeDist(pObj->two));
            }
        }
        
        //float minDist = *min_element(dist.begin(), dist.end());
        auto it = std::min_element(dist.begin(), dist.end());
        float minDist = *it;

        // Start with a loose threshold
        if (minDist < 5.0f)
        {
            det->isNew = false;
			det->isGood = (det->nPts >= 25); // number of points in detection (default is 25)

			int idx = std::distance(dist.begin(), it);
            MapObject *pMO = vpLocalMapObjects[idx];
            if (!pKF->mdAssociatedObjects.count(pMO)) {
                pKF->mdAssociatedObjects[pMO] = minDist;
                pKF->AddMapObject(pMO, i);
                pMO->AddObservation(pKF, i);
            } else {// Another detection is associated with pMO, compare distance
                if (minDist < pKF->mdAssociatedObjects[pMO]) {
                    cout << "Associated to: " << pMO->mnId << ", Distance: " << minDist << endl;
                    pKF->mdAssociatedObjects[pMO] = minDist;
                    int detId = pMO->GetObservations()[pKF];
                    pKF->EraseMapObjectMatch(detId);
                    vDetections[detId]->isNew = true;
                    pKF->AddMapObject(pMO, i);
                    pMO->AddObservation(pKF, i);
                }
            }
        }
        else {
            det->isNew = true;
            det->isGood = (det->nPts >= 50); // number of points in detection (default is 50)
        }
    }
}

/*
 * Tracking utils for monocular input on Freiburg Cars and Redwood OS
 */
cv::Mat Tracking::GetCameraIntrinsics()
{
    return mK;
}

void Tracking::GetObjectDetectionsMono(KeyFrame *pKF)
{
    PyThreadStateLock PyThreadLock;

    py::list detections = mpSystem->pySequence.attr("get_frame_by_id")(pKF->mnFrameId);
    int num_dets = detections.size();
    // No detections, return immediately
    if (num_dets == 0) // FIXME tariq commented out to look like lidar detection code
        return;

    for (int detected_idx = 0; detected_idx < num_dets; detected_idx++)
    {
        auto det = new ObjectDetection();
        auto py_det = detections[detected_idx];
        det->background_rays = py_det.attr("background_rays").cast<Eigen::MatrixXf>();
        auto mask = py_det.attr("mask").cast<Eigen::MatrixXf>();
        cv::Mat mask_cv;
        cv::eigen2cv(mask, mask_cv);
        // cv::imwrite("mask.png", mask_cv);
        cv::Mat mask_erro = mask_cv.clone();
        cv::Mat kernel = getStructuringElement(cv::MORPH_ELLIPSE,
                                               cv::Size(2 * maskErrosion + 1, 2 * maskErrosion + 1),
                                               cv::Point(maskErrosion, maskErrosion));
        cv::erode(mask_cv, mask_erro, kernel);

        // get 2D feature points inside mask
        for (int i = 0; i < pKF->mvKeys.size(); i++)
        {
            int val = (int) mask_erro.at<float>(pKF->mvKeys[i].pt.y, pKF->mvKeys[i].pt.x);
            if (val > 0)  // inside the mask
            {
                det->AddFeaturePoint(i);
            }
        }

        // Reject the detection if too few keypoints are extracted
        if (det->NumberOfPoints() < 20) // FIXME (Default is 20)
        {
            det->isGood = false;
        }
        pKF->mvpDetectedObjects.push_back(det);
    }
    pKF->nObj = pKF->mvpDetectedObjects.size();
    pKF->mvpMapObjects = vector<MapObject *>(pKF->nObj, static_cast<MapObject *>(NULL));
    
    std::cout << "Mono Detections:";
    std::cout << " KF: " << pKF->mnId;
    std::cout << " 2D masks: " << pKF->mvpDetectedObjects.size();
    std::cout << std::endl;

}

void Tracking::AssociateObjectsByProjection(KeyFrame *pKF)
{
    auto mvpMapPoints = pKF->GetMapPointMatches();
    // Try to match and triangulate key-points with last key-frame
    auto detectionsKF1 = pKF->mvpDetectedObjects;
    for (int d_i = 0; d_i < detectionsKF1.size(); d_i++)
    {
        // cout << "Detection: " << d_i + 1 << endl;
        auto detKF1 = detectionsKF1[d_i];
        map<int, int> observed_object_id;
        int nOutliers = 0;
        for (int k_i : detKF1->GetFeaturePoints()) {
            auto pMP = mvpMapPoints[k_i];
            if (!pMP)
                continue;
            if (pMP->isOutlier())
            {
                nOutliers++;
                continue;
            }

            if (pMP->object_id < 0)
                continue;

            if (observed_object_id.count(pMP->object_id))
                observed_object_id[pMP->object_id] += 1;
            else
                observed_object_id[pMP->object_id] = 1;
        }

        // If associated with an object
        if (!observed_object_id.empty())
        {
            // Find object that has the most matches
            int object_id_max_matches = 0;  // global object id
            int max_matches = 0;
            for (auto it = observed_object_id.begin(); it != observed_object_id.end(); it++) {
                if (it->second > max_matches) {
                    max_matches = it->second;
                    object_id_max_matches = it->first;
                }
            }

            // associated object
            auto pMO = mpAtlas->GetMapObject(object_id_max_matches);
            pKF->AddMapObject(pMO, d_i);
            detKF1->isNew = false;

            // add newly detected feature points to object
            int newly_matched_points = 0;
            for (int k_i : detKF1->GetFeaturePoints()) {
                auto pMP = mvpMapPoints[k_i];
                if (pMP)
                {
                    if (pMP->isBad())
                        continue;
                    // new map points
                    if (pMP->object_id < 0)
                    {
                        pMP->in_any_object = true;
                        pMP->object_id = object_id_max_matches;
                        pMO->AddMapPoints(pMP);
                        newly_matched_points++;
                    }
                    else
                    {
                        // if pMP is already associate to a different object, set bad flag
                        if (pMP->object_id != object_id_max_matches)
                            pMP->SetBadFlag();
                    }
                }
            }
            /*cout <<  "Matches: " << max_matches << ", New points: " << newly_matched_points << ", Keypoints: " <<
                 detKF1->mvKeysIndices.size() << ", Associated to object by projection " << object_id_max_matches
                 << endl << endl;*/
        }

    }
}


}
