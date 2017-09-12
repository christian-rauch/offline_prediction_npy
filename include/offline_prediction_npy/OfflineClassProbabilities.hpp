#ifndef OFFLINECLASSPROBABILITIES_HPP
#define OFFLINECLASSPROBABILITIES_HPP

#include <ros/ros.h>
#include <sensor_msgs/CompressedImage.h>
#include <image_transport/image_transport.h>
#include <image_classification_msgs/PixelProbabilityList2.h>

#include <eigen3/Eigen/Eigen>

class OfflineClassProbabilities {
public:
    OfflineClassProbabilities(const std::string img_colour_topic,
                              const std::string img_depth_topic,
                              const std::string prob_topic,
                              const std::string label_topic,
                              const bool publish = true);

    void setClassProbPath(const std::string pred_npy_path) {
        this->pred_npy_path = pred_npy_path;
    }

    bool setLabelColour(const std::string class_id_file);

    image_classification_msgs::PixelProbabilityList2ConstPtr getPP(const std_msgs::HeaderConstPtr header);

private:
    typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matf32XXrm;
    typedef Eigen::Matrix<uint16_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matui16XXrm;
    typedef Eigen::Matrix<uint16_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Matui16XXcm;
    typedef Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Matui8XXrm;

    void cb(const sensor_msgs::CompressedImageConstPtr& msg);

    ros::NodeHandle n;
    ros::Subscriber sub_img_colour;
    ros::Subscriber sub_img_depth;
    ros::Publisher pub_class_prob;
    ros::Publisher pub_label_colour;

    std::string pred_npy_path;
    std::vector<std::string> link_names;
};

#endif // OFFLINECLASSPROBABILITIES_HPP
