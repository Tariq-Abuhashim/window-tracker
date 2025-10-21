/**
* 
* Tariq Abuhashim.
* Adopted from ORBSLAM3 examples.
* Kitti Monocular-Inertial example.
* This reads Kitti images and inertial data using actual time stamps.
* Raw data can be used (previous examples only use odometry data).
* KITTI.yaml has been updated to capture data and Inertial changes.
*
* 09-Oct-2025
*
*/

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <thread>
#include <sstream>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <System.h>
#include "ImuTypes.h"

#include <filesystem>
namespace fs = std::filesystem;

using namespace std;

std::uint64_t to_microseconds(const boost::posix_time::ptime& t) {
    static boost::posix_time::ptime epoch(boost::gregorian::date(1970, 1, 1));
    return (t - epoch).total_microseconds();
}

void LoadImages(const string &pathSeq, vector<string> &vstrImageLeft,
				vector<double> &vTimeStamps);

void LoadIMU(const string &strImuPath, vector<double> &vTimeStamps, 
				vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro);

int main(int argc, char **argv)
{
	if(argc < 4)
	{
		std::cerr << std::endl << "Usage: " << std::endl;
		          
		std::cerr << "./Examples/Monocular-Inertial/mono_inertial_kitti "
		          << "Vocabulary/ORBvoc.txt "
		          << "<path_to_vocabulary> "
		          << "<path_to_settings> "
		          << "<path_to_seq1> "
		          << "[<path_to_seq2> ...]"
		          << std::endl;

		std::cerr << std::endl << "Examples:" << std::endl;

		std::cerr << "./Examples/Monocular-Inertial/mono_inertial_kitti "
		          << "Vocabulary/ORBvoc.txt "
		          << "/media/mrt/Whale/data/kitti/2011_09_30/KITTI.yaml "
		          << "/media/mrt/Whale/data/kitti/2011_09_30/"
		          << std::endl;

		std::cerr << std::endl << "Debug mode (then type <run>, and backtrace <bt>):" << std::endl;
		std::cerr << "cmake -DCMAKE_BUILD_TYPE=Debug .. \n";
		std::cerr << "make -j  \n";
		std::cerr << "gdb --args ./Examples/Monocular-Inertial/mono_inertial_kitti "
		          << "<path_to_vocabulary> "
		          << "<path_to_settings> "
		          << "<path_to_seq1> "
		          << std::endl;
		          
		std::cerr << std::endl << "Find memory leaks:" << std::endl;
		std::cerr << "valgrind ./Examples/Monocular-Inertial/mono_inertial_kitti "
		          << "<path_to_vocabulary> "
		          << "<path_to_settings> "
		          << "<path_to_seq1> "
		          << std::endl;

		return 1;
	}

	const string vocabFile = argv[1];
    const string settingsFile = argv[2];
    
    const int num_seq = argc - 3;
    cout << "[MONO_INERTIAL_KITTI] num_seq = " << num_seq << endl;
    bool bFileName= (((argc-3) % 1) == 1);
    string file_name;
    if (bFileName)
    {
        file_name = string(argv[argc-1]);
        cout << "[MONO_INERTIAL_KITTI] file name: " << file_name << endl;
    }

    /* Load all sequences: */
    int seq;
    // Images
    vector< vector<double> > vTimestampsCam; vTimestampsCam.resize(num_seq);
    vector< vector<string> > vstrImageLeft; vstrImageLeft.resize(num_seq);
    vector<int> nImages; nImages.resize(num_seq);
 	vector< vector<double> > vTimestampsImu; vTimestampsImu.resize(num_seq);
    vector< vector<cv::Point3f> > vAcc; vAcc.resize(num_seq);
    vector< vector<cv::Point3f> > vGyro; vGyro.resize(num_seq);
    vector<int> nImu; nImu.resize(num_seq);
    vector<int> first_imu(num_seq,0);
	cout << endl << "-------" << endl;
    cout.precision(17);
    int tot_images = 0;
    for (seq = 0; seq<num_seq; seq++)
    {
        string pathSeq(argv[(2*seq) + 3]);

        cout << "[MONO_INERTIAL_KITTI] Loading images for sequence " << seq << "...\n";
        LoadImages(pathSeq+"2011_09_30_drive_0018_sync/", vstrImageLeft[seq], vTimestampsCam[seq]);
        std::cout 	<< "[MONO_INERTIAL_KITTI] Sequence has " 
        			<< vstrImageLeft[seq].size() 
        			<< " Images ..." << std::endl;

        cout << "[MONO_INERTIAL_KITTI] Loading IMU for sequence " << seq << "...";
		LoadIMU(pathSeq+"2011_09_30_drive_0018_extract/", vTimestampsImu[seq], vAcc[seq], vGyro[seq]);
		std::cout 	<< "[MONO_INERTIAL_KITTI] Sequence has " 
        			<< vTimestampsImu[seq].size() 
        			<< " IMU data ..." << std::endl;

		//for (int i = 0; i<vTimestampsImu[seq].size(); i++)
		//for (int i = 0; i<10; i++)
		//	cout << vTimestampsImu[seq][i] << " " << vAcc[seq][i] << " " << vGyro[seq][i] << endl;

        nImages[seq] = vstrImageLeft[seq].size();
        tot_images += nImages[seq];
        nImu[seq] = vTimestampsImu[seq].size();

        if((nImages[seq]<=0)||(nImu[seq]<=0))
        {
            cerr << "[MONO_INERTIAL_KITTI] ERROR: Failed to load images or IMU for sequence" << seq << endl;
            return 1;
        }

        // Find first imu to be considered, supposing imu measurements start first
        while(vTimestampsImu[seq][first_imu[seq]]<=vTimestampsCam[seq][0])
            first_imu[seq]++;
        first_imu[seq]--; // first imu measurement to be considered
    }
	cout << endl << "-------" << endl;
	
	// Create SLAM system. It initializes all system threads and gets ready to process frames.
	ORB_SLAM3::System SLAM(vocabFile, 
						settingsFile, 
						ORB_SLAM3::System::IMU_MONOCULAR, 
						true, 
						0, 
						argv[3]); // FIXME if argv[3] is sequnce path, how to set for (num_seq>1)
	float imageScale = SLAM.GetImageScale();

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(tot_images);
    double t_track = 0;
    double ttrack_tot = 0;

    cv::Mat imLeft;
    for (seq = 0; seq<num_seq; seq++)
    {
    	std::cout << std::endl;
		std::cout << "[MONO_INERTIAL_KITTI] Starting for sequence " << seq << " ..." << std::endl;

		// Seq loop
        vector<ORB_SLAM3::IMU::Point> vImuMeas;
        int proccIm = 0;
        for(int ni=0; ni<nImages[seq]; ni++, proccIm++)
        {
            // Read image from file
            imLeft = cv::imread(vstrImageLeft[seq][ni],cv::IMREAD_UNCHANGED);
			double tframe = vTimestampsCam[seq][ni];
			
            if(imLeft.empty())
            {
                cerr << endl << "[MONO_INERTIAL_KITTI] Failed to load image at: "
                     << vstrImageLeft[seq][ni] << endl;
                return 1;
            }
            
            if(imageScale != 1.f)
            {
                int width = imLeft.cols * imageScale;
                int height = imLeft.rows * imageScale;
                cv::resize(imLeft, imLeft, cv::Size(width, height));
            }

            // Load imu measurements from previous frame
            vImuMeas.clear();
            if(ni>0) {
                while(vTimestampsImu[seq][first_imu[seq]]<=vTimestampsCam[seq][ni])
                {
                    vImuMeas.push_back(ORB_SLAM3::IMU::Point(
                    					vAcc[seq][first_imu[seq]].x,
										vAcc[seq][first_imu[seq]].y,
										vAcc[seq][first_imu[seq]].z,
										vGyro[seq][first_imu[seq]].x,
										vGyro[seq][first_imu[seq]].y,
										vGyro[seq][first_imu[seq]].z,
										vTimestampsImu[seq][first_imu[seq]]));
                    first_imu[seq]++;
                }
			}

            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
            
            // Pass the images to the SLAM system
            SLAM.TrackMonocular(imLeft,tframe,vImuMeas);
            
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
            t_track = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2-t1).count();
            ttrack_tot += t_track;
            vTimesTrack[ni] = t_track;
            
            std::cout << "fps = " << 1000/t_track << std::endl;
            
#ifdef REGISTER_TIMES
            //SLAM.InsertTrackTime(t_track);
#endif
          	usleep(1000.0);
          	
			// Wait to load the next frame (following actually timestamps of data)
            double T=0;
            if(ni<nImages[seq]-1)
                T = vTimestampsCam[seq][ni+1]-tframe;
            else if(ni>0)
                T = tframe-vTimestampsCam[seq][ni-1];
            if(t_track<T)
                usleep((T-t_track)*1e6); // 1e6
            
		}
		
		if(seq < num_seq - 1) {
			cout << "[MONO_INERTIAL_KITTI] Changing the dataset" << endl;
			//SLAM.ChangeDataset();
		}
    }
    
    // Wait viewer
    //cv::waitKey(0);
    
    // Stop all threads
    SLAM.Shutdown();
    
    // Save camera trajectory
    /*
    if (bFileName)
    {
        const string kf_file =  "kf_" + string(argv[argc-1]) + ".txt";
        const string f_file =  "f_" + string(argv[argc-1]) + ".txt";
        SLAM.SaveTrajectoryEuRoC(f_file);
        SLAM.SaveKeyFrameTrajectoryEuRoC(kf_file);
    }
    else
    {
        SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");
        SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");
    }
    */

    return 0;
}

void LoadImages(const string &pathSeq, vector<string> &vstrImageLeft,
				vector<double> &vTimeStamps)
{

	std::cout << "[LoadImages] pathSeq := " + pathSeq << std::endl;
	
	/* 
	for kitti/07 visual odometry dataset: 
		- has no IMU data
	
	for kitti/yyyy_mm_dd_drive_xxxx_sync semantic/object dataset, 
		- Tr in calibration file is between image_00 and Velodyne
		- SLAM takes /image_00/ and /image_01/
		- MaskRCNN takes /image_02/
		- SLAM/Object outputs are in /image_00/ frame
	*/
	
	/* monocular, only get left camera */
	string strPrefixLeft = pathSeq + "/image_00/data/"; // left
    
    /* open timestamps file */
    ifstream fTimes;
    //string strPathTimeFile = pathSeq + "/times.txt";
	string strPathTimeFile = pathSeq + "/oxts/timestamps.txt";
    fTimes.open(strPathTimeFile.c_str());
    if (!fTimes.is_open()) {
    	std::cerr << "[LoadImages] Couldn't open " << strPathTimeFile << std::endl;
    	exit(1);
	}

	vTimeStamps.reserve(5000);
    vstrImageLeft.reserve(5000);
    
    /* read timestamps from file
    	format = yyyy-mm-dd hh:mm:s */
    string str;
    while (getline(fTimes, str))
    {
		if(!str.empty())
        {
			int yyyy, mm, dd, h, m = 0;
			double s = 0;
			if (sscanf(str.c_str(), "%d-%d-%d %d:%d:%lf", &yyyy, &mm, &dd, &h, &m, &s) == 6)
			{
  				double t = h *3600 + m*60 + s;
				//cout << t << endl;
				vTimeStamps.push_back(t);
			}
        }
    }
    
    /* read timestamps from file
    	format = seconds */
/*
    string str;
    while (getline(fTimes, str))
    {
		if(!str.empty())
        {
			double s = 0;
			if (sscanf(str.c_str(), "%lf", &s) == 1)
			{
				vTimeStamps.push_back(s);
			}
        }
    } */
    
    /* fake timestamps using fps
    	this is not useful if IMU-Image synchronisation is needed
    */
    /*
    const float fps = 5.;
    const float dt = 1. / fps;
    float t = 0.;
    string s;
    while(getline(fTimes,s))
    {
        if(!s.empty())
        {
            vTimeStamps.push_back(t);
            t += dt;
        }
    }
    */
    
	const int nTimes = vTimeStamps.size();
	std::cout << "[LoadImages] nTimes " << nTimes << "\n";
	
	/* Assuming timestamps and images are synchronised */
    vstrImageLeft.resize(nTimes);
    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(10) << i;  // FIXME 6 or 10
        vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
    }
}

void LoadIMU(const string &pathSeq, vector<double> &vTimeStamps, vector<cv::Point3f> &vAcc, vector<cv::Point3f> &vGyro)
{

	ifstream fTimes;
    string strPathTimeFile = pathSeq + "/oxts/timestamps.txt";
	fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string str;
        getline(fTimes,str);
        if(!str.empty())
        {
			int yyyy, mm, dd, h, m = 0;
			double s = 0;
			if (sscanf(str.c_str(), "%d-%d-%d %d:%d:%lf", &yyyy, &mm, &dd, &h, &m, &s) == 6)
			{
  				double t = h *3600 + m*60 + s;
				//cout << t << endl;
				vTimeStamps.push_back(t);
			}
        }
    }
	const int nTimes = vTimeStamps.size();

    vAcc.reserve(50000);
    vGyro.reserve(50000);
	for(int i=0; i<nTimes; i++)
	{
		stringstream ss;
		ss << setfill('0') << setw(10) << i;
		string strImuPath = pathSeq + "/oxts/data/" + ss.str() + ".txt";
		//cout << strImuPath << endl;
		ifstream fImu;
		fImu.open(strImuPath.c_str());

		string s;
		getline(fImu,s);
		if(!s.empty())
        {
            string item;
            size_t pos = 0;
            double data[30];
            int count = 0;
            while ((pos = s.find(' ')) != string::npos) {
                item = s.substr(0, pos); // get sub string at starting location 0 with length pos
                data[count++] = stod(item); // Convert string to double
                s.erase(0, pos + 1); // string& erase (size_t pos = 0, size_t len = npos);
            }
            item = s.substr(0, pos);
            data[29] = stod(item);

            //vTimeStamps.push_back(data[0]/1e9);
            vAcc.push_back(cv::Point3f(data[14],data[15],data[16]));
            vGyro.push_back(cv::Point3f(data[20],data[21],data[22]));

			//if (i<10)
			//{
			//	cout << endl;
			//	cout << data[11] << " " << data[12] << " " << data[13] << " " << data[17] << " " << data[18] << " " << data[19] << endl;
			//	cout << cv::Point3d(data[11],data[12],data[13]) <<  " " << cv::Point3d(data[17],data[18],data[19]) << endl;
			//}
        } 
	}
}
