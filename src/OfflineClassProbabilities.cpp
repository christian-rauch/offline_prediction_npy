#include <offline_prediction_npy/OfflineClassProbabilities.hpp>
#include <cnpy.h>

#include <dart_msgs/LabelColours.h>

#include <fstream>

OfflineClassProbabilities::OfflineClassProbabilities(const std::string img_colour_topic,
                                                     const std::string img_depth_topic,
                                                     const std::string prob_topic,
                                                     const std::string label_topic,
                                                     const bool publish) :
    n("~"),
    pub_label_colour(n.advertise<dart_msgs::LabelColours>(label_topic, 1, true))
{
    if(publish) {
        sub_img_colour = n.subscribe(img_colour_topic, 1, &OfflineClassProbabilities::cb, this);
        sub_img_depth = n.subscribe(img_depth_topic, 1, &OfflineClassProbabilities::cb, this);
        pub_class_prob = n.advertise<dart_msgs::PixelProbabilityList2>(prob_topic, 1);
    }
    std::string link_class_file_path;
    if(!n.getParam("link_class_file", link_class_file_path)) {
        throw std::runtime_error("parameter 'link_class_file' not provided");
    }

    if(!setLabelColour(link_class_file_path)) {
        throw std::runtime_error("error reading "+link_class_file_path);
    }

    if(!n.getParam("pred_npy_path", pred_npy_path)) {
        throw std::runtime_error("parameter 'pred_npy_path' not provided");
    }
}

bool OfflineClassProbabilities::setLabelColour(const std::string class_id_file) {
    dart_msgs::LabelColours lc;

    // read CSV
    std::ifstream csv_file;
    csv_file.open(class_id_file);

    if(!csv_file.good()) {
        std::cerr<<"error openning file: "<<class_id_file<<std::endl;
        return false;
    }

    std::string csv_row;
    while (std::getline(csv_file, csv_row)) {
        std::istringstream iss_row(csv_row);

        std::string link_name;
        std::getline(iss_row, link_name, ' ');
        lc.link_names.push_back(link_name);

        std::string str_id;
        std::getline(iss_row, str_id, ' ');
        lc.class_ids.push_back(std::stoi(str_id));

        std_msgs::ColorRGBA c;
        std::string str_r;
        std::getline(iss_row, str_r, ' ');
        c.r = std::stof(str_r);
        std::string str_g;
        std::getline(iss_row, str_g, ' ');
        c.g = std::stof(str_g);
        std::string str_b;
        std::getline(iss_row, str_b, ' ');
        c.b = std::stof(str_b);
        c.a = 1.0;

        lc.colours.push_back(c);
    }

    link_names = lc.link_names;

    pub_label_colour.publish(lc);

    return true;
}

dart_msgs::PixelProbabilityList2ConstPtr
OfflineClassProbabilities::getPP(const std_msgs::HeaderConstPtr header)
{
    // we just need the image time
    const uint64_t msg_stamp = header->stamp.toNSec();

    if(pred_npy_path=="") {
        return dart_msgs::PixelProbabilityList2ConstPtr(new dart_msgs::PixelProbabilityList2);
    }

    cnpy::npz_t pred_npz;
    try {
        pred_npz = cnpy::npz_load(pred_npy_path+"/prob_"+std::to_string(msg_stamp)+".npz");
    } catch (const std::runtime_error) {
        return dart_msgs::PixelProbabilityList2ConstPtr(new dart_msgs::PixelProbabilityList2);
    }

    const cnpy::NpyArray class_id_npy = pred_npz.at("class_id");
    const cnpy::NpyArray coord_npy = pred_npz.at("coord");
    const cnpy::NpyArray prob_npy = pred_npz.at("prob");

    dart_msgs::PixelProbabilityList2Ptr pp_msg(new dart_msgs::PixelProbabilityList2());
    pp_msg->header = *header;
    if(coord_npy.fortran_order==false) {
        // is row-major
        pp_msg->coordinates.assign(coord_npy.data<uint16_t>(), coord_npy.data<uint16_t>()+coord_npy.shape[0]*coord_npy.shape[1]);
    }
    else {
        // is column-major, convert to row major
        const Matui16XXcm m_cm = Eigen::Map<const Matui16XXcm>(coord_npy.data<uint16_t>(), coord_npy.shape[0], coord_npy.shape[1]).transpose();
        Matui16XXrm m_rm = Matui16XXrm(coord_npy.shape[0],coord_npy.shape[1]);
        m_rm.col(0) = m_cm.row(0);
        m_rm.col(1) = m_cm.row(1);
        pp_msg->coordinates.assign((uint16_t*)m_rm.data(), (uint16_t*)m_rm.data()+(coord_npy.shape[0]*coord_npy.shape[1]));
    }
    if(prob_npy.fortran_order==false) {
        pp_msg->probability.assign(prob_npy.data<float>(), prob_npy.data<float>()+(prob_npy.shape[0]*prob_npy.shape[1]));
    }
    else {
        throw std::runtime_error("prob_npy is in column-major order");
    }

    if(link_names.size()==0) {
        throw std::runtime_error("link names not loaded, run 'setLabelColour' first");
    }

    pp_msg->link_names = link_names;

    // check if last class is 100 (occlusion)
    const uint8_t l = (class_id_npy.data<uint8_t>())[class_id_npy.shape[0]-1];
    if(l == 100) {
        pp_msg->link_names.push_back("occlusion");
    }

    return pp_msg;
}

void OfflineClassProbabilities::cb(const sensor_msgs::CompressedImageConstPtr& img_msg) {
    std_msgs::HeaderConstPtr header_ptr(new std_msgs::Header(img_msg->header));
    pub_class_prob.publish(getPP(header_ptr));
}
