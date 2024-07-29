/**
* 
* Tariq Abuhashim.
*
* This is mono_vulcan.cpp but synced to Advanced Navigation.
* Seperately, it generates SLAM solution in format similar to COLMAP, and at the same time it saves
* Image and GPS Coordinates.
*
* Usage:
*
* ./run_camera.h  (part of csv-to-euroc)
* ./run_nav.h  (part of csv-to-euroc)
* ./mono_geo_vulcan Vocabulary/ORBvoc.txt Examples/Monocular/vulcan.yaml $WORKSPACE
*
* Updates:
*
* 16-Jul-2024
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

void writeToFile(const std::string& filename, const std::string& content) {
    std::ofstream file;
    file.open(filename, std::ios::out | std::ios::app); // Open in append mode
    if (file.is_open()) {
        file << content;
        file.close();
    } else {
        std::cerr << "Unable to open " << filename << " for writing." << std::endl;
    }
}

bool doesDirExist(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        return false; // Cannot access directory
    } else if (info.st_mode & S_IFDIR) {
        return true; // It's a directory
    }
    return false; // It exists but is no directory
}

bool createDir(const std::string& dirPath) {
    return (mkdir(dirPath.c_str(), 0777) == 0);
}

int main(int argc, char **argv)
{
    if(argc < 3)
    {
		cerr << endl << "Usage: \n"
                     "./mono_vulcan \\ \n"
                     "  path_to_vocabulary path_to_settings output_dir"<< endl;
		cerr << endl << "Example: \n" 
                     "./mono_vulcan \\ \n"
          			 "  Vocabulary/ORBvoc.txt \\ \n"
          			 "  Examples/Monocular/vulcan.yaml \\ \n"
          			 "  $WORKSPACE" << endl;
		cerr << endl;
        return 1;
    }
    std::string output_dir = argv[3];
    cout.precision(17);
    cout << endl << "-------" << endl;

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::MONOCULAR, true);

	dataQueue imageQueue;
	dataQueue navQueue;
    ImageReader image_reader(imageQueue,"127.0.0.1", 4001, "t,3ui,s[7062528]"); /* ("127.0.0.1", 4001) */
	NavReader nav_reader(navQueue, "127.0.0.1", 4002, "t,3d,3f,2uw,33f,5ub"); /* ("127.0.0.1", 4002) */
    double image_time = -1, nav_time = -1;

	std::string outputFilename, directoryPath;

	// Start threads to read from the sensors
	std::cout << "\n Start data threads ..." << std::endl;
    std::thread t2(&ImageReader::readFromSocket, &image_reader); // fills imageQueue
	std::thread t3(&NavReader::readFromSocket, &nav_reader);
	std::this_thread::sleep_for(std::chrono::seconds(1)); // ensure that threads have started before moving on
	if (image_reader.is_connected() && nav_reader.is_connected()) std::cout << "\n Connected ..." << std::endl;
    while (image_reader.is_connected() && nav_reader.is_connected()) {
		while (!image_reader.safeEmpty() && !nav_reader.safeEmpty()) {

			std::shared_ptr<SensorData> image_ptr = image_reader.safeFront();
            std::shared_ptr<ImageData> actual_image_ptr = std::dynamic_pointer_cast<ImageData>(image_ptr);
            if (!actual_image_ptr) {
                std::cerr << "Failed to cast SensorData to ImageData." << std::endl;
                image_reader.safePop();
                continue;
            }
			ImageData &image = *actual_image_ptr;
			image_time = image.getTimestamp() / 1e+6;

            std::shared_ptr<SensorData> nav_ptr = nav_reader.safeFront();
            std::shared_ptr<NavData> actual_nav_ptr = std::dynamic_pointer_cast<NavData>(nav_ptr);
            if (!actual_nav_ptr) {
                std::cerr << "Failed to cast SensorData to NavData." << std::endl;
                nav_reader.safePop();
                continue;
            }
            NavData &nav = *actual_nav_ptr;
            nav_time = nav.getTimestamp() / 1e+6;

			if (std::abs(image_time - nav_time) < 0.01) {

                std::cout << image_time << " " << nav_time << std::endl;

				Image I = image.getData();
				cv::Mat Cvimage(I.height, I.width, CV_8UC3, I.data.data());
				SLAM.TrackMonocular(Cvimage, image_time);
        		std::string timestamp = std::to_string(image_time);
        		size_t pos = timestamp.find('.');
        		if (pos != std::string::npos) {timestamp.erase(pos, 1);}
        		outputFilename = timestamp + ".png";
        		directoryPath = output_dir + "/images/";
        		if (!doesDirExist(directoryPath)) {
            		if (!createDir(directoryPath)) {
                		std::cerr << "Failed to create directory." << std::endl;
                		continue;
            		}
        		}
        		cv::imwrite(directoryPath + outputFilename, Cvimage);

				Nav N = nav.getData();
        		std::stringstream navContent;
        		navContent << std::fixed << outputFilename << " ";
				navContent << std::fixed << std::setprecision(9) << N.coordinates[0] << " " << N.coordinates[1] << " " << N.coordinates[2] << "\n";
				outputFilename = "gps.txt";
        		directoryPath = output_dir + "/nav/";
				if (!doesDirExist(directoryPath)) {
            		if (!createDir(directoryPath)) {
                		std::cerr << "Failed to create directory." << std::endl;
                		continue;
            		}
        		}
        		writeToFile(directoryPath + outputFilename, navContent.str());

                image_reader.safePop();
                nav_reader.safePop();

            } else if ((image_time - nav_time) > 0) {
                nav_reader.safePop();
            } else if ((image_time - nav_time) < 0) {
                image_reader.safePop();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
		}
	}
	std::cout << "\n Joining camera thread ..." << std::endl;
	t2.join();
	t3.join();

    // Save camera trajectory
	// Creating a directory
	if (mkdir("sparse", 0777) == -1)
		cerr << "Error :  " << strerror(errno) << endl;
	else
		cout << "Directory created";
	SLAM.WriteCamerasText("sparse/cameras.txt");
	SLAM.WriteImagesText("sparse/images.txt");
	SLAM.WritePoints3DText("sparse/points3D.txt");

	// Stop all threads
	std::cout << "\n SLAM Shutdown ..." << std::endl;
	SLAM.Shutdown();

    return 0;
}
