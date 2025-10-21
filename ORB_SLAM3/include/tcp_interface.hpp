// Copyright (c) 2023 Mission Systems Pty Ltd
// Tariq Abuhashim, May 2023

/*
TCP-Interface for Object-SLAM
to compile: g++ -o tcp_interface tcp_interface.cpp -lboost_system -lpthread
*/

/**
* Tariq updated - Jul, 2023
**/

#ifndef TCP_INTERFACE_HPP
#define TCP_INTERFACE_HPP

#include <iostream>
#include <deque>
#include <string>
#include <thread>
#include <mutex>
#include <boost/asio.hpp>

#include <snark/imaging/cv_mat/serialization.h>
#include <snark/sensors/lidars/ouster/types.h>
#include <snark/sensors/lidars/ouster/config.h>
#include <snark/sensors/lidars/ouster/traits.h>

//const std::size_t IMAGE_SIZE = 7062548; // 20 header + 7062528 data
const double EPSILON = 0.001; // in seconds
const std::size_t QUEUE_MAX_SIZE = 50;
// std::mutex mtxImageQueue, mtxCloudQueue;

// this data strcture is used to read lidar data from tcp stream
struct LidarPoint {
	// t,2uw,2ui,uw,3d,3uw,3d
	// ouster-to-csv lidar --output-format | csv-format collapse
	// ouster-to-csv lidar --output-fields
	// echo t,3ui | csv-format size
    std::uint64_t timestamp; // 8-bytes (64-bits)
	std::uint16_t measurement_id; // 2-bytes
	std::uint16_t frame_id; // 2-bytes
	std::uint32_t encoder_count; // 4-bytes
	std::uint32_t block; // 4-bytes
	std::uint16_t channel; // 2-bytes
	double range; // 8-bytes (64-bits)
	double bearing; // 8-bytes (64-bits)
	double elevation; // 8-bytes (64-bits)
	std::uint16_t signal; // 2-bytes
	std::uint16_t reflectivity; // 2-bytes
	std::uint16_t ambient; // 2-bytes
	double point[3]; // 24-bytes (x,y,z)
};

struct Header {
    uint32_t seq;
    std::uint64_t stamp; // ROS uses a specific time format, but we can use a double for simplicity.
    std::string frame_id;
};

struct Fields {
    std::uint8_t INT8    = 1;
	std::uint8_t UINT8   = 2;
	std::uint8_t INT16   = 3;
	std::uint8_t UINT16  = 4;
	std::uint8_t INT32   = 5;
	std::uint8_t UINT32  = 6;
	std::uint8_t FLOAT32 = 7;
	std::uint8_t FLOAT64 = 8;

	std::string name;      // Name of field
	std::uint32_t offset;    // Offset from start of point struct
	std::uint8_t datatype;  // Datatype enumeration, see above
	std::uint32_t count;    // How many elements in the field

	void clear() {
		name.clear();// std::string has a clear function that resets it to an empty string
        offset = 0;
		datatype = 0;
		count = 0;
    }
};

struct Point {
    double x, y, z, range, bearing, elevation; // range, bearing, elevation
	std::uint16_t intensity; // reflectivity
	double offset;  // time offset in seconds from a base time (since the first point in the current scan/frame was captured)
};

// this is the data structure used to process the lidar data
struct Cloud {
	Header header;
	std::vector<Point> points;
	uint32_t height;
	uint32_t width;
	//Fields fields;
	bool is_bigendian;
	//uint32_t point_step;
	//uint32_t row_step;
	//std::uint16_t* data;
	bool is_dense;

	Cloud(): header(), points(), height(0), width(0), is_bigendian(false), is_dense(true) {}

	void clear() {
        header.stamp = 0;
		header.frame_id.clear();// std::string has a clear function that resets it to an empty string
		points.clear(); points.resize(0);
        height = 0;
		width = 0;
		//fields.clear();
		is_bigendian = false;
		//point_step = 0;
		//row_step = 0;
		//data = nullptr;
		is_dense = true;
    }

	bool isempty() {
		return width==0;
	}

	std::size_t size() {
		return width;
	}
};

// this is the data structure used to process the image data
// echo t,3ui | csv-format size
struct Image {
	Header header;
 	uint32_t height;
    uint32_t width;
    std::string encoding;
    bool is_bigendian;
    uint32_t step;
    std::vector<uint8_t> data; // Use a vector to store the image data because the size can vary.
	Image() : 
        header(),
        height(0),
        width(0),
        encoding(),
        is_bigendian(false),
        step(0),
        data()
    {}
	//FIXME 
	//Please note that the default constructors for std::string and std::vector will construct empty instances of these classes. 
	//std::string() creates an empty string, and std::vector<uint8_t>() creates an empty vector.
};

// t,21f
struct Imu { 
	Header header; // time
	std::array<float, 3> accelerometer;
	std::array<float, 3> gyroscope;
	std::array<float, 3> magnetometer;
	std::array<float, 3> orientation;
	std::array<float, 3> orientation_stddev;
	Imu() :
        header(),
        accelerometer{0, 0, 0},
        gyroscope{0, 0, 0},
        magnetometer{0, 0, 0},
        orientation{0, 0, 0},
        orientation_stddev{0, 0, 0}
    {}
};

// t,3d,3f,2uw,33f,5ub
// t, 
// coordinates/latitude,coordinates/longitude,height,
// orientation/x,orientation/y,orientation/z,
// system_status,filter_status,
// position_stddev/x,position_stddev/y,position_stddev/z,orientation_stddev/x,orientation_stddev/y,orientation_stddev/z, velocity_stddev/x,velocity_stddev/y,velocity_stddev/z,
// velocity/x,velocity/y,velocity/z,
// body_acceleration/x,body_acceleration/y,body_acceleration/z,g_force, 
// angular_velocity/x,angular_velocity/y,angular_velocity/z,
// accelerometer/x,accelerometer/y,accelerometer/z,gyroscope/x,gyroscope/y,gyroscope/z,magnetometer/x,magnetometer/y,magnetometer/z,
// imu_temperature,pressure,pressure_temperature,
// hdop,vdop,gps_satellites,
// glonass_satellites,beidou_satellites,galileo_satellites,sbas_satellites
struct Nav { 
	Header header; // time
	std::array<double, 3> coordinates; // coordinates/latitude,coordinates/longitude,height
	std::array<float, 3> orientation; // orientation/x,orientation/y,orientation/z
	std::array<int, 2> status; // 2uw // system_status,filter_status
	std::array<float, 9> stddev; // 3xposition + 3xorientation + 3xvelocity
	std::array<float, 3> velocity; // velocity/x,velocity/y,velocity/z
	std::array<float, 3> body_acceleration; // body_acceleration/x,body_acceleration/y,body_acceleration/z
	float g_force;
	std::array<float, 3> angular_velocity; // angular_velocity/x,angular_velocity/y,angular_velocity/z
	std::array<float, 9> imu; // 3xaccelerometer + 3xgyroscope + 3xmagnetometer
	float imu_temperature;
	float pressure;
	float pressure_temperature;
	std::array<float, 2> dop; //hdop,vdop
	std::array<int, 5> satellites; // 5ub
	Nav() :
        header(),
        coordinates{0, 0, 0},
        orientation{0, 0, 0},
        status{0, 0},
        stddev{0, 0, 0, 0, 0, 0, 0, 0, 0},
        velocity{0, 0, 0},
        body_acceleration{0, 0, 0},
        g_force(0),
        angular_velocity{0, 0, 0},
        imu{0, 0, 0, 0, 0, 0, 0, 0, 0},
		imu_temperature(0),
		pressure(0),
		pressure_temperature(0),
		dop{0,0},
		satellites{0,0,0,0,0}
    {}
};

/* base data class */
class SensorData {
public:
    SensorData(double timestamp) : timestamp_(timestamp) {}
    virtual ~SensorData() {}

    double getTimestamp() const { return timestamp_; }

    // Make it pure virtual
    virtual std::size_t getDataSize() const = 0;

private:
    double timestamp_;
};

/* derived class (Image) */
class ImageData : public SensorData {
public:
    ImageData(double timestamp, const Image& data)
        : SensorData(timestamp), data_(data) {}

	std::size_t getDataSize() const override {
        return data_.data.size(); // total size of image data
    }
	std::uint32_t getHeight() const {
        return data_.height;
    }
	std::uint32_t getWidth() const {
        return data_.width;
    }
	const Image& getData() const { return data_; }

private:
    Image data_;
};

/* derived class (Cloud) */
class CloudData : public SensorData {
public:
    CloudData(double timestamp, const Cloud& data)
        : SensorData(timestamp), data_(data) {}

	std::size_t getDataSize() const override {
		return data_.points.size();
	}

	const Cloud& getData() const { return data_; }

private:
    Cloud data_;
};

/* derived class (Imu) */
class ImuData : public SensorData {
public:
    ImuData(double timestamp, const Imu& data)
        : SensorData(timestamp), data_(data) {}

	std::size_t getDataSize() const override {
        return 1; // TODO or whatever makes sense for your use case
    }

	const Imu& getData() const { return data_; }

private:
    Imu data_;
};

/* derived class (Nav) */
class NavData : public SensorData {
public:
    NavData(double timestamp, const Nav& data)
        : SensorData(timestamp), data_(data) {}

	std::size_t getDataSize() const override {
        return 1; // TODO or whatever makes sense for your use case
    }

	const Nav& getData() const { return data_; }

private:
    Nav data_;
};

/* data queue */
using dataQueue = std::deque<std::shared_ptr<SensorData>>;

/* base data class */
class SensorReader {
protected:
    dataQueue& deque_;
    std::string ip_;
    short port_;
	boost::asio::ip::tcp::socket* socket_;
	const std::string format_;
	Cloud cloud;
	std::uint16_t currentFrameId = 0;
	std::atomic<bool> is_connected_;

public:
    SensorReader(dataQueue& deque, const std::string& ip, const short& port, const std::string& format) 
        : deque_(deque), ip_(ip), port_(port), format_(format), is_connected_(false), socket_(nullptr) {}

    virtual ~SensorReader() {
		if(socket_){
            socket_->close();
            delete socket_;
            socket_ = nullptr;
        }
	}

	void readFromSocket() {
		try {
		    boost::asio::io_service io_service;
			socket_ =  new boost::asio::ip::tcp::socket(io_service);
            boost::asio::ip::tcp::endpoint endpoint(boost::asio::ip::address::from_string(ip_), port_);
            socket_->connect(endpoint);
			is_connected_.store(true);
			std::cout << "SensorReader of type " << typeid(*this).name() << " is set to read data on " <<ip_<<":"<<port_<< std::endl;
		    while (is_connected_) { handleData(); }
		}
		catch (const boost::system::system_error& ex) {
			std::cerr << ex.what() << '\n';
            std::cerr << ex.code() << '\n';
			is_connected_.store(false);
		}
    }

	bool is_connected() const {
        return is_connected_.load(); // ensures that this variable is updated and read atomically to avoid race conditions. 
    }

    virtual void processImpl() = 0; // pure virtual function
	//virtual void addData() = 0; // pure virtual function

	void safePush(std::shared_ptr<SensorData> data) {
        std::lock_guard<std::mutex> lock(mtx);
        deque_.push_back(data);
    }

    std::shared_ptr<SensorData> safeFront() {
        std::lock_guard<std::mutex> lock(mtx);
        return deque_.empty() ? nullptr : deque_.front();
    }

    void safePop() {
        std::lock_guard<std::mutex> lock(mtx);
        if (!deque_.empty()) {
            deque_.pop_front();
        }
    }

    bool safeEmpty() {
        std::lock_guard<std::mutex> lock(mtx);
        return deque_.empty();
    }

	std::mutex mtx;

protected:
	void handleData() { processImpl(); }

	/* ptime to microseconds */
	std::uint64_t to_microseconds(const boost::posix_time::ptime& t) {
    	static boost::posix_time::ptime epoch(boost::gregorian::date(1970, 1, 1));
    	return (t - epoch).total_microseconds();
	}

};

/* derived image class */
class ImageReader : public SensorReader {
public:
    ImageReader(dataQueue& deque, const std::string& ip, const short& port, const std::string& format)
        : SensorReader(deque, ip, port, format) {}

private:

/*	void processImpl() override {
		comma::csv::format format(format_);
		boost::asio::streambuf buffer;
		auto mutable_buffers = buffer.prepare(format.size());
		boost::asio::read(*socket_, buffer, boost::asio::transfer_exactly(format.size()));

		// Process sensor data based on its type
		Image reading = processImage(buffer);
	    addData(reading.header.stamp, reading);
	} */
	
	void processImpl() override {
		try {
			// Read fixed 20-byte header
			boost::asio::streambuf headerBuf;
			boost::asio::read(*socket_, headerBuf, boost::asio::transfer_exactly(20));
			const char* headerData = boost::asio::buffer_cast<const char*>(headerBuf.data());
			
			// Parse header
			uint64_t ptime;
			std::memcpy(&ptime, headerData, sizeof(uint64_t)); 
			uint32_t height = *reinterpret_cast<const int*>(headerData + 8);
			uint32_t width = *reinterpret_cast<const int*>(headerData + 12);
			uint32_t depth = *reinterpret_cast<const int*>(headerData + 16);
			
			if (width == 0 || height == 0 || width > 10000 || height > 10000) {
				std::ostringstream oss;
				oss << "Invalid image dimensions read from stream: "
				    << width << "x" << height;
				throw std::runtime_error(oss.str());
			}
			
			// Compute payload size
			int channels = 3;
			size_t expectedBytes = static_cast<size_t>(width) * height * channels;
			
			// Read image payload
			boost::asio::streambuf imgBuf;
			boost::asio::read(*socket_, imgBuf, boost::asio::transfer_exactly(expectedBytes));
			const char* imgData = boost::asio::buffer_cast<const char*>(imgBuf.data());
			
			// Construct Image
			Image reading = processImage(ptime, width, height, depth, imgData, expectedBytes);
			addData(reading.header.stamp, reading);
		} catch (std::exception& e) {
			std::cerr << "[ImageReader] Error: " << e.what() << std::endl;
			throw;
		}
		
	}
	
	/* Image specific function */
	Image processImage(uint64_t ptime, uint32_t width, uint32_t height,
					uint32_t depth, const char* imgData, size_t dataSize) {
					
		int channels = 3;
    	int cvDepth = (depth == 8) ? CV_8U : CV_8U;
    	
    	cv::Mat image(height, width, CV_MAKETYPE(cvDepth, channels));
    	std::memcpy(image.data, imgData, dataSize);
    	
    	Header header;
		header.seq = 1;
		header.stamp = ptime;
		header.frame_id = "camera_frame";

		Image img;
		img.header = header;
		img.height = image.rows;
		img.width = image.cols;
		img.encoding = "bgr8";
		img.is_bigendian = false;
		img.step = image.step[0];
		img.data.assign(image.datastart, image.dataend);

		return img;
	}

	/* Image specific function */
	Image processImage(boost::asio::streambuf& buffer) {
		if(buffer.size() < 20) { // Make sure there's enough data for the header
			//throw std::runtime_error("handleImageData: Invalid buffer size, for image header.");
		}
		const char* headerData = boost::asio::buffer_cast<const char*>(buffer.data()); // address of first data

		// Extract the values from the header based on their positions
		// Assuming the 8-byte representation follows a specific format, e.g., a Unix timestamp
		// Using reinterpret_cast to reinterpret bytes as a different type is considered unsafe and can lead to 
		// 		undefined behavior if the actual bytes do not represent a valid uint64_t value. 
		// Perform explicit byte-level deserialization by copying the bytes from headerData into a uint64_t variable 
		uint64_t ptime;
		std::memcpy(&ptime, headerData, sizeof(uint64_t)); 
		uint32_t height = *reinterpret_cast<const int*>(headerData + 8);
		uint32_t width = *reinterpret_cast<const int*>(headerData + 12);
		uint32_t depth = *reinterpret_cast<const int*>(headerData + 16);
		//std::cout << ptime << "-" << columns << "-" << rows << "-" << dataType << std::endl;
		
		// Sanity checks
		if (width == 0 || height == 0 || width > 10000 || height > 10000) {
		    std::ostringstream oss;
		    oss << "Invalid image dimensions read from stream: "
		        << width << "x" << height;
		    throw std::runtime_error(oss.str());
		}

		int channels = 3;  // Assuming 3 channels
		int cvDepth = CV_8U; // Hardcode to 8-bit if your stream is raw RGB data
    	if (depth == 0 || depth > 8) cvDepth = CV_8U; // ignore bogus field
				
		const char* imageData = headerData + 20;  // Skip the 20-byte header
		size_t expectedBytes = static_cast<size_t>(width) * height * channels;
		
		if(buffer.size() < 20 + expectedBytes) { // Make sure there's enough data for the image
			std::ostringstream oss;
			oss << "processImage(): buffer too small for image payload. "
            	<< "Expected " << expectedBytes << " bytes, got " << buffer.size() - 20;
        	throw std::runtime_error(oss.str());
		}
		
		cv::Mat image(height, width, CV_MAKETYPE(cvDepth, channels));
		std::memcpy(image.data, imageData, expectedBytes);
		//cv::rotate(image, image, 0); //cv::ROTATE_90_COUNTERCLOCKWISE, cv::ROTATE_180
		//std::cout << image.rows << "x" << image.cols << "x" << image.channels() << std::endl;
		//cv::imwrite("output.jpg", image);

		// Create a header
		Header header;
		header.seq = 1;
		header.stamp = ptime; // This should be a meaningful timestamp in your application.
		header.frame_id = "camera_frame";

		// Initialize the Image
		Image img;
		img.header = header;
		img.height = image.rows; //time_image.second.rows;
		img.width = image.cols; //time_image.second.cols;
		img.encoding = "bgr8";
		img.is_bigendian = false;
		img.step = image.step[0]; //time_image.second.step[0];  // Assuming 3 bytes per pixel for BGR8 encoding.
		img.data.assign(image.datastart, image.dataend);
    	//std::cout << "Received Image Data: Timestamp: " << header.stamp
       //       	<< ", Dimensions: " << img.width << "x" << img.height
        //      	<< ", Type: " << time_image.second.type()
        //      	<< ", Pixel Count: " << img.width*img.height*time_image.second.channels() << std::endl;
		return img;
	}

	/* add data to the deque */
	void addData(double timestamp, const Image& data) {
		std::lock_guard<std::mutex> guard(mtx); 
		while (!deque_.empty() && deque_.back()->getTimestamp() > timestamp) {
			deque_.pop_front();
		}
		deque_.push_back(std::make_shared<ImageData>(timestamp, data));
	}
    
};

/* derived cloud class */
class CloudReader : public SensorReader {
public:
    CloudReader(dataQueue& deque, const std::string& ip, const short& port, const std::string& format)
        : SensorReader(deque, ip, port, format) {}

private:

	void processImpl() override {
		comma::csv::format format(format_);
		boost::asio::streambuf buffer;
		auto mutable_buffers = buffer.prepare(format.size());
		boost::asio::read(*socket_, buffer, boost::asio::transfer_exactly(format.size()));

		// Process sensor data based on its type
		processCloud(buffer); // this one needs to continue accomulating points
	}

	/* Lidar specific function */
	void processCloud(boost::asio::streambuf& buffer) {

		const char* data = boost::asio::buffer_cast<const char*>(buffer.data());
		
		size_t ptr = 0;
		LidarPoint lidar_point;

		lidar_point.timestamp = *reinterpret_cast<const std::uint64_t*>(data + ptr);
		ptr += sizeof(uint64_t);

		lidar_point.measurement_id = *reinterpret_cast<const std::uint16_t*>(data + ptr);
		ptr += sizeof(std::uint16_t);

		lidar_point.frame_id = *reinterpret_cast<const std::uint16_t*>(data + ptr);
		ptr += sizeof(std::uint16_t);

		lidar_point.encoder_count = *reinterpret_cast<const std::uint32_t*>(data + ptr);
		ptr += sizeof(std::uint32_t);

		lidar_point.block = *reinterpret_cast<const std::uint32_t*>(data + ptr);
		ptr += sizeof(std::uint32_t);

		lidar_point.channel = *reinterpret_cast<const std::uint16_t*>(data + ptr);
		ptr += sizeof(std::uint16_t);

		lidar_point.range = *reinterpret_cast<const double*>(data + ptr);
		ptr += sizeof(double);

		lidar_point.bearing = *reinterpret_cast<const double*>(data + ptr);
		ptr += sizeof(double);

		lidar_point.elevation = *reinterpret_cast<const double*>(data + ptr);
		ptr += sizeof(double);

		lidar_point.signal = *reinterpret_cast<const std::uint16_t*>(data + ptr);
		ptr += sizeof(std::uint16_t);

		lidar_point.reflectivity = *reinterpret_cast<const std::uint16_t*>(data + ptr);
		ptr += sizeof(std::uint16_t);

		lidar_point.ambient = *reinterpret_cast<const std::uint16_t*>(data + ptr);
		ptr += sizeof(std::uint16_t);

		lidar_point.point[0] = *reinterpret_cast<const double*>(data + ptr);
		ptr += sizeof(double); // corrected pointer increment here

		lidar_point.point[1] = *reinterpret_cast<const double*>(data + ptr);
		ptr += sizeof(double);

		lidar_point.point[2] = *reinterpret_cast<const double*>(data + ptr);
		ptr += sizeof(double);

		if (lidar_point.reflectivity==0) return; // FIXME filter points before adding them, which field to use?
		if (currentFrameId!=lidar_point.frame_id) { // check if this is a new scan // FIXME seems like block is more stable to use then frame_id
			if (currentFrameId!=0) {
				//std::cout << "Number of points in scan " << currentFrameId << " = " << cloud.size() << std::endl;
				if (!cloud.isempty()) addData(cloud.header.stamp, cloud);
			}

			currentFrameId = lidar_point.frame_id;
			
			cloud.clear();
			Header header;
			header.stamp = lidar_point.timestamp;
			header.frame_id = "ouster_frame";
			cloud.header = header;
			cloud.height = 1;
		}
		cloud.width+=1;

		Point point;
		point.x = lidar_point.point[0]; // set x-coordinate
		point.y = lidar_point.point[1]; // set y-coordinate
		point.z = lidar_point.point[2]; // set z-coordinate
		point.intensity = lidar_point.signal; // set reflectivity
		point.range = lidar_point.range; // set range
		point.bearing = lidar_point.bearing; // set bearing
		point.elevation = lidar_point.elevation; // set elevation
		//point.offset = lidar_point.timestamp - header.stamp; // set offset

		cloud.points.emplace_back(point); // Use emplace_back instead of push_back for efficiency: 
		// The emplace_back method directly constructs the object in place, avoiding the extra copy or move operation required by push_back.
	}

	/* add data to the deque */
	void addData(double timestamp, const Cloud& data) {
		std::lock_guard<std::mutex> guard(mtx); 
		while (!deque_.empty() && deque_.back()->getTimestamp() > timestamp) {
			deque_.pop_front();
		}
		deque_.push_back(std::make_shared<CloudData>(timestamp, data));
	}
    
};

/* derived imu class */

class ImuReader : public SensorReader {
public:
    ImuReader(dataQueue& deque, const std::string& ip, const short& port, const std::string& format)
        : SensorReader(deque, ip, port, format) {}

private:

	void processImpl() override {
		comma::csv::format format(format_);
		boost::asio::streambuf buffer;
		auto mutable_buffers = buffer.prepare(format.size());
		boost::asio::read(*socket_, buffer, boost::asio::transfer_exactly(format.size()));
		//std::cout << "\n format size " << format.size() << std::endl;

		// Process sensor data based on its type
		Imu reading = processImu(buffer);
	    addData(reading.header.stamp, reading);
	}

	/* Imu specific function using <advanced-navigation-imu>*/
/*
	Imu processImu(boost::asio::streambuf& buffer) {

		const char* data = boost::asio::buffer_cast<const char*>(buffer.data());
		
		size_t ptr = 0;

		// Create a header
		Header header;
		header.seq = 1;
		header.stamp = *reinterpret_cast<const uint64_t*>(data + ptr); // timestamp from buffer
		ptr += sizeof(uint64_t);
		header.frame_id = "imu_frame";

		// Initialize the Imu data from <advanced-navigation-imu>
		Imu imu;
		imu.header = header;
		for (int i = 0; i < 3; i++) {
            imu.accelerometer[i] = *reinterpret_cast<const float*>(data + ptr);
            ptr += sizeof(float);
        }
		for (int i = 0; i < 3; i++) {
            imu.gyroscope[i] = *reinterpret_cast<const float*>(data + ptr);
            ptr += sizeof(float);
        }
		for (int i = 0; i < 3; i++) {
            imu.magnetometer[i] = *reinterpret_cast<const float*>(data + ptr);
            ptr += sizeof(float);
        }
		for (int i = 0; i < 3; i++) {
            imu.orientation[i] = *reinterpret_cast<const float*>(data + ptr);
            ptr += sizeof(float);
        }
		for (int i = 0; i < 3; i++) {
            imu.orientation_stddev[i] = *reinterpret_cast<const float*>(data + ptr);
            ptr += sizeof(float);
        }
		
		//std::cout << imu.accelerometer[0] << " " << imu.accelerometer[1] << " " << imu.accelerometer[2] << " "
		//		  << imu.gyroscope[0]     << " " << imu.gyroscope[1]     << " " << imu.gyroscope[2]     << " "
		//		  << imu.magnetometer[0]  << " " << imu.magnetometer[1]  << " " << imu.magnetometer[2]  << " "
		//		  << imu.orientation[0]   << " " << imu.orientation[1]   << " " << imu.orientation[2]   << "\n";

		return imu;
	}
*/

	/* Imu specific function using <advanced-navigation>*/
	Imu processImu(boost::asio::streambuf& buffer) {

		const char* data = boost::asio::buffer_cast<const char*>(buffer.data());
		
		size_t ptr = 0;

		// Create a header
		Header header;
		header.seq = 1;
		//1 
		header.stamp = *reinterpret_cast<const uint64_t*>(data + ptr); // timestamp from buffer
		ptr += 1 * sizeof(uint64_t);
		header.frame_id = "imu_frame";

		// Initialize the Imu data
		Imu imu;
		imu.header = header;

		// Skip irrelevant data 2-4
		ptr += 3* sizeof(double); // coordinates (latitude, longitude, height)

		// orientation 5-7
		for (int i = 0; i < 3; i++) { //orientation/x,orientation/y,orientation/z
            imu.orientation[i] = *reinterpret_cast<const float*>(data + ptr);
            ptr += sizeof(float);
        }

		// Skip irrelevant data
		ptr += 2 * sizeof(uint16_t);  //system_status, filter_status 8-9
		ptr += 3 * sizeof(float); // position_stddev (x, y, z) 10-12
		ptr += 3 * sizeof(float); // orientation_stddev (x, y, z) 13-15
		ptr += 3 * sizeof(float); // velocity_stddev (x, y, z) 16-18
		ptr += 3 * sizeof(float); // velocity (x, y, z) 19-21

		//body_acceleration
		ptr += 3 * sizeof(float); // body_acceleration (x, y, z) 22-24
		//for (int i = 0; i < 3; i++) { // accelerometer/x,accelerometer/y,accelerometer/z
        //    imu.accelerometer[i] = *reinterpret_cast<const float*>(data + ptr);
        //    ptr += sizeof(float);
        //}

		ptr += 1 * sizeof(float); //g_force; 25

		//angular_velocity
		ptr += 3 * sizeof(float); // angular_velocity (x, y, z) 26-28
		//for (int i = 0; i < 3; i++) { //gyroscope/x,gyroscope/y,gyroscope/z
        //    imu.gyroscope[i] = *reinterpret_cast<const float*>(data + ptr);
        //    ptr += sizeof(float);
        //}

		//accelerometer (29,30,31)
		//ptr -= 1 * sizeof(float); // FIXME where is the -1 coming from ???
		//ptr += 3 * sizeof(float); // accelerometers (x, y, z)
		for (int i = 0; i < 3; i++) { // accelerometer/x,accelerometer/y,accelerometer/z
            imu.accelerometer[i] = *reinterpret_cast<const float*>(data + ptr);
            ptr += sizeof(float);
        }

		//gyroscope (32,33,34)
		//ptr += 3 * sizeof(float); // gyroscopes (x, y, z)
		for (int i = 0; i < 3; i++) { //gyroscope/x,gyroscope/y,gyroscope/z
            imu.gyroscope[i] = *reinterpret_cast<const float*>(data + ptr);
            ptr += sizeof(float);
        }

		//magnetometer (35,36,37)
		for (int i = 0; i < 3; i++) {  //magnetometer/x,magnetometer/y,magnetometer/z
            imu.magnetometer[i] = *reinterpret_cast<const float*>(data + ptr);
            ptr += sizeof(float);
        }
/*
		std::cout << imu.accelerometer[0] << " " << imu.accelerometer[1] << " " << imu.accelerometer[2] << " "
				  << imu.gyroscope[0]     << " " << imu.gyroscope[1]     << " " << imu.gyroscope[2]     << " "
				  << imu.magnetometer[0]  << " " << imu.magnetometer[1]  << " " << imu.magnetometer[2]  << " "
				  << imu.orientation[0]   << " " << imu.orientation[1]   << " " << imu.orientation[2]   << "\n";
*/
		return imu;
	}

	/* add data to the deque */
	void addData(double timestamp, const Imu& data) {
		std::lock_guard<std::mutex> guard(mtx); 
		while (!deque_.empty() && deque_.back()->getTimestamp() > timestamp) {
			deque_.pop_front();
		}
		deque_.push_back(std::make_shared<ImuData>(timestamp, data));
	}
};

/* derived nav class */

class NavReader : public SensorReader {
public:
    NavReader(dataQueue& deque, const std::string& ip, const short& port, const std::string& format)
        : SensorReader(deque, ip, port, format) {}

private:

	void processImpl() override {
		comma::csv::format format(format_);
		boost::asio::streambuf buffer;
		auto mutable_buffers = buffer.prepare(format.size());
		boost::asio::read(*socket_, buffer, boost::asio::transfer_exactly(format.size()));
		//std::cout << "\n format size " << format.size() << std::endl;

		// Process sensor data based on its type
		Nav reading = processNav(buffer);
	    addData(reading.header.stamp, reading);
	}

	/* Nav specific function using <advanced-navigation>*/
	Nav processNav(boost::asio::streambuf& buffer) {

		const char* data = boost::asio::buffer_cast<const char*>(buffer.data());
		
		size_t ptr = 0;

		// create a header
		Header header;
		header.seq = 1;

		// timestamp
		header.stamp = *reinterpret_cast<const uint64_t*>(data + ptr); // timestamp from buffer
		ptr += 1 * sizeof(uint64_t);
		header.frame_id = "ned_frame";

		// Initialize the Nav data
		Nav nav;
		nav.header = header;

		// coordinates 2-4
		for (int i = 0; i < 3; i++) { //orientation/x,orientation/y,orientation/z
            nav.coordinates[i] = *reinterpret_cast<const double*>(data + ptr);
            ptr += sizeof(double);
        }

		// orientation 5-7
		for (int i = 0; i < 3; i++) { //orientation/x,orientation/y,orientation/z
            nav.orientation[i] = *reinterpret_cast<const float*>(data + ptr);
            ptr += sizeof(float);
        }

		return nav;
	}

	/* add data to the deque */
	void addData(double timestamp, const Nav& data) {
		std::lock_guard<std::mutex> guard(mtx); 
		while (!deque_.empty() && deque_.back()->getTimestamp() > timestamp) {
			deque_.pop_front();
		}
		deque_.push_back(std::make_shared<NavData>(timestamp, data));
	}
};

#endif // TCP_INTERFACE_HPP
