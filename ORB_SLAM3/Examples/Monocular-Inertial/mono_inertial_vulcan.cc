/**
* 
* Tariq Abuhashim.
*
* Usage:
* 
* Stream the data using (check file /media/mrt/Whale/data/mission_systems/which_data):
*
* $ cd /media/mrt/Whale/data/mission_systems/2024_05_30_03_auto_orbit
* $ format=t,3ui,s[7062528]
* $  cat cameras/alvium_1800_c507c/*.bin | csv-play --binary $format --slow 6 | io-publish tcp:4001 --size $( echo $format | csv-format size )
*
* $ cd /media/mrt/Whale/data/mission_systems/2024_05_30_03_auto_orbit
* $ format=$( ms-log-multitool data --include advanced-navigation --output-format )
* $ cat advanced-navigation/*.bin | ms-log-multitool data --include advanced-navigation | csv-play --binary $format --slow 6 | io-publish tcp:4002 --size $( echo $format | csv-format size )
*
* Run:
*
* $ cd ~/dev/mission-systems/window-tracker/ORB_SLAM3
* $ ./Examples/Monocular-Inertial/mono_inertial_vulcan Vocabulary/ORBvoc.txt Examples/Monocular-Inertial/vulcan.yaml
*
* Updates:
*
* 17-Jun-2024
*
*/

#include <iostream>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <sstream>

#include <opencv2/opencv.hpp>

#include<System.h>

// for make directory
#include <sys/stat.h>
#include <sys/types.h>

#include "tcp_interface.hpp" // mission-systems

using namespace std;

int main(int argc, char **argv)
{
    if(argc < 3)
    {
		cerr << endl << "Usage: \n"
                     "./mono_inertial_vulcan \\ \n"
                     "  path_to_vocabulary path_to_settings"<< endl;
		cerr << endl << "Example: \n" 
                     "./mono_inertial_vulcan \\ \n"
          			 "  Vocabulary/ORBvoc.txt \\ \n"
          			 "  Examples/Monocular/vulcan.yaml" << endl;
		cerr << endl;
        return 1;
    }
	cout << endl << "-------" << endl;
    cout.precision(17);

	//int fps = 3;
    //float dT = 1.f/fps;
    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::IMU_MONOCULAR, true);
    //double t_track = 0.f;
	//vector<float> vTimesTrack;
    //vTimesTrack.resize(10000);

	dataQueue imageQueue; // for ImageData
    ImageReader image_reader(imageQueue,"127.0.0.1", 4001, "t,3ui,s[7062528]"); /* ("127.0.0.1", 4001) */
	double image_time=-1;

	dataQueue imuQueue;  // for ImuData
	ImuReader imu_reader(imuQueue, "127.0.0.1", 4002, "t,3d,3f,2uw,33f,5ub"); /* ("127.0.0.1", 4002) */
	double imu_time=-1;
	vector<ORB_SLAM3::IMU::Point> vImuMeas;

    bool first_frame = true;

	// Start threads to read from the sensors
	std::cout << "\n Start data threads ..." << std::endl;
    std::thread t2(&ImageReader::readFromSocket, &image_reader); // fills imageQueue
	std::thread t3(&ImuReader::readFromSocket, &imu_reader); // fills imuQueue
	std::this_thread::sleep_for(std::chrono::seconds(1)); // ensure that threads have started before moving on
	if (image_reader.is_connected()) std::cout << "\n Connected ..." << std::endl;
    while (true) { //image_reader.is_connected()) {  // continue as long as both readers are alive
		while (!image_reader.safeEmpty()) {
			std::shared_ptr<SensorData> image_ptr = image_reader.safeFront();
			if (!image_ptr) {
            	std::cerr << "Received a null pointer from the image reader." << std::endl;
            	continue;
        	}
			std::shared_ptr<ImageData> actual_image_ptr = std::dynamic_pointer_cast<ImageData>(image_ptr);
			if (!actual_image_ptr) {
            	std::cerr << "Failed to cast SensorData to ImageData." << std::endl;
            	image_reader.safePop(); // Ensure the item is removed from the queue
            	continue;
        	}
			ImageData& image = *actual_image_ptr; // dereferencing
			Image I = image.getData();
			image_time = image.getTimestamp()/1e+6;

			// get the imu match
			//dataQueue relevant_imu_data;
			while (!imu_reader.safeEmpty()) {
				std::shared_ptr<SensorData> imu_ptr = imu_reader.safeFront();
				std::shared_ptr<ImuData> actual_imu_ptr = std::dynamic_pointer_cast<ImuData>(imu_ptr);
				ImuData& imu = *actual_imu_ptr; // dereferencing
				imu_time = imu.getTimestamp()/1e+6;
				if (imu_time <= image_time) {
					//relevant_imu_data.push_back(imu_ptr);
					Imu IMU = imu.getData();
					if (first_frame && abs(imu_time-image_time)<0.02) // discard all IMU data before first image frame, unless 0.02s difference for IMU at 50Hz (1/50), ie take last measurement
						vImuMeas.push_back(ORB_SLAM3::IMU::Point(IMU.accelerometer[0], IMU.accelerometer[1], IMU.accelerometer[2],
                                                             	 IMU.gyroscope[0],     IMU.gyroscope[1],     IMU.gyroscope[2],
                                                                 imu_time));
					else if (!first_frame) // if not first camera frame, take IMU measurement
						vImuMeas.push_back(ORB_SLAM3::IMU::Point(IMU.accelerometer[0], IMU.accelerometer[1], IMU.accelerometer[2],
                                                             	 IMU.gyroscope[0],     IMU.gyroscope[1],     IMU.gyroscope[2],
                                                                 imu_time));						
					std::cout 	<< imu_time << " ";
					std::cout 	<< IMU.accelerometer[0] << " " << IMU.accelerometer[1] << " " << IMU.accelerometer[2] << " ";
					std::cout 	<< IMU.gyroscope[0] << " " << IMU.gyroscope[1] << " " << IMU.gyroscope[2];
					std::cout 	<< std::endl;
					imu_reader.safePop();
				}	
				else break;
				first_frame = false;
			}
			

			// get the image
			cv::Mat im(I.height, I.width, CV_8UC3, I.data.data()); // does not copy the data. If vecData goes out of scope or is modified, the cv::Mat will be affected.
			cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
			cv::flip(im, im, 0);

#ifdef COMPILEDWITHC11
            //std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    #else
            //std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
    #endif
            // Pass the image to the SLAM system
            cout << image_time << " " << vImuMeas.size() << endl;
            SLAM.TrackMonocular(im, image_time, vImuMeas); // TODO change to monocular_inertial
			cout << "----------------------------------" << endl;

    #ifdef COMPILEDWITHC11
            //std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    #else
            //std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
    #endif

#ifdef REGISTER_TIMES
            //t_track = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
            //SLAM.InsertTrackTime(t_track);
#endif
            //double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
            //vTimesTrack[ni]=ttrack;

			image_reader.safePop();
			vImuMeas.clear();
			//relevant_imu_data.clear(); // Clear the relevant_imu_data for the next iteration
			//std::this_thread::sleep_for(std::chrono::milliseconds(100));  // Sleep to avoid busy waiting
		}
	}
	std::cout << "\n Joining camera thread ..." << std::endl;
	t2.join();

	// Stop all threads
    //SLAM.Shutdown();

    // Save camera trajectory
	// Creating a directory
	//if (mkdir("sparse", 0777) == -1)
	//	cerr << "Error :  " << strerror(errno) << endl;
	//else
	//	cout << "Directory created";
	//SLAM.WriteCamerasText("sparse/cameras.txt");
	//SLAM.WriteImagesText("sparse/images.txt");
	//SLAM.WritePoints3DText("sparse/points3D.txt");

    return 0;
}
