#include <fstream>
#include <vector>
#include <string>

struct Camera {
    int id;
    std::string model;
    int width;
    int height;
    std::vector<double> params;
};

struct Image {
    int id;
    double qw, qx, qy, qz; // Quaternion components
    double tx, ty, tz; // Translation components
    int camera_id;
    std::string image_name;
};

struct Point3D {
    int id;
    double x, y, z;
    int r, g, b;
    std::vector<std::pair<int, int>> observations; // pair of image_id and keypoint_index
};

void writeCameras(const std::vector<Camera>& cameras, const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) throw std::runtime_error("Unable to open file");

    for (const auto& camera : cameras) {
        file << camera.id << " " << camera.model << " " << camera.width << " " << camera.height;
        for (const auto& param : camera.params) {
            file << " " << param;
        }
        file << "\n";
    }
}

void writeImages(const std::vector<Image>& images, const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) throw std::runtime_error("Unable to open file");

    for (const auto& image : images) {
        file << image.id << " " << image.qw << " " << image.qx << " " << image.qy << " " << image.qz << " "
             << image.tx << " " << image.ty << " " << image.tz << " " << image.camera_id << " " << image.image_name << "\n";
    }
}

void writePoints3D(const std::vector<Point3D>& points, const std::string& filepath) {
    std::ofstream file(filepath);
    if (!file.is_open()) throw std::runtime_error("Unable to open file");

    for (const auto& point : points) {
        file << point.id << " " << point.x << " " << point.y << " " << point.z << " "
             << point.r << " " << point.g << " " << point.b;
        for (const auto& obs : point.observations) {
            file << " " << obs.first << " " << obs.second;
        }
        file << "\n";
    }
}

std::vector<Point3D> ConvertORBMapPointsToColmap(const std::vector<MapPoint*>& orbMapPoints) {
    std::vector<Point3D> points3D;
    int pointId = 1;

    for (const auto& orbPoint : orbMapPoints) {
        Point3D point;
        cv::Point3f pos = orbPoint->GetWorldPos();
        point.x = pos.x;
        point.y = pos.y;
        point.z = pos.z;

        // Default color, replace if ORB-SLAM provides color
        point.r = 255;  
        point.g = 255;
        point.b = 255;

        // Assuming ORB-SLAM has a method to get observations
        point.observations = orbPoint->GetObservations();

        point.id = pointId++;
        points3D.push_back(point);
    }

    return points3D;
}

int main() {
    try {
        // Example data for Cameras
        std::vector<Camera> cameras = {
            {1, "PINHOLE", 1920, 1080, {1600.0, 1200.0, 960.0, 540.0}},
            // Add more cameras as needed
        };

        // Example data for Images
        std::vector<Image> images = {
            {1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1, "image1.jpg"},
            // Add more images as needed
        };

        // Example data for Points3D
        std::vector<Point3D> points3D = {
            {1, 1.0, 2.0, 3.0, 255, 0, 0, {{1, 1}, {2, 2}}},
            // Add more points as needed
        };

        // Call function to write cameras data to file
        writeCameras(cameras, "cameras.txt");

        // Call function to write images data to file
        writeImages(images, "images.txt");

        // Call function to write points3D data to file
        writePoints3D(points3D, "points3D.txt");

    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Data written successfully." << std::endl;
    return 0;
}