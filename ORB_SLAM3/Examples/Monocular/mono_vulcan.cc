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
* Run:
*
* $ cd ~/dev/mission-systems/window-tracker/ORB_SLAM3
* $ ./Examples/Monocular/mono_vulcan Vocabulary/ORBvoc.txt Examples/Monocular/vulcan.yaml
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
                     "./mono_vulcan \\ \n"
                     "  path_to_vocabulary path_to_settings"<< endl;
		cerr << endl << "Example: \n" 
                     "./mono_vulcan \\ \n"
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
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::MONOCULAR, true);
    //double t_track = 0.f;
	//vector<float> vTimesTrack;
    //vTimesTrack.resize(10000);

	dataQueue imageQueue; // for ImageData
    ImageReader image_reader(imageQueue,"127.0.0.1", 4001, "t,3ui,s[7062548]"); /* ("127.0.0.1", 4001) */
	double timestamp=-1;

	// Start threads to read from the sensors
	std::cout << "\n Start data threads ..." << std::endl;
    std::thread t2(&ImageReader::readFromSocket, &image_reader); // fills imageQueue
	std::this_thread::sleep_for(std::chrono::seconds(1)); // ensure that threads have started before moving on
	if (image_reader.is_connected()) std::cout << "\n Connected ..." << std::endl;
    while (image_reader.is_connected()) {  // continue as long as both readers are alive
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
			timestamp = image.getTimestamp()/1e+6;
			
			// Diagnostics
			std::cout << "[DEBUG] timestamp(ns)=" << timestamp
				      << " width=" << I.width << " height=" << I.height
				      << " data.size=" << I.data.size() << std::endl;

			// get the image
			cv::Mat im(I.height, I.width, CV_8UC3, I.data.data()); // does not copy the data. If vecData goes out of scope or is modified, the cv::Mat will be affected.
			cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
			//cv::flip(im, im, 0);

#ifdef COMPILEDWITHC11
            //std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    #else
            //std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
    #endif
            // Pass the image to the SLAM system
            cout << timestamp << endl;
            SLAM.TrackMonocular(im, timestamp);

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
			//std::this_thread::sleep_for(std::chrono::milliseconds(100));  // Sleep to avoid busy waiting
		}
	}
	std::cout << "\n Joining camera thread ..." << std::endl;
	t2.join();

	// Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
	// Creating a directory
	if (mkdir("sparse", 0777) == -1)
		cerr << "Error :  " << strerror(errno) << endl;
	else
		cout << "Directory created";
	SLAM.WriteCamerasText("sparse/cameras.txt");
	SLAM.WriteImagesText("sparse/images.txt");
	SLAM.WritePoints3DText("sparse/points3D.txt");

    return 0;
}
