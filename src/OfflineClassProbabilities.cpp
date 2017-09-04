#include <offline_prediction_npy/OfflineClassProbabilities.hpp>
#include <cnpy/cnpy.h>

#include <image_classification_msgs/LabelColours.h>

#include <fstream>

OfflineClassProbabilities::OfflineClassProbabilities(const std::string img_topic, const std::string prob_topic, const std::string label_topic) :
    n("~"),
    sub_img(n.subscribe(img_topic, 1, &OfflineClassProbabilities::cb, this)),
    pub_class_prob(n.advertise<image_classification_msgs::PixelProbabilityList2>(prob_topic, 1)),
    pub_label_colour(n.advertise<image_classification_msgs::LabelColours>(label_topic, 1, true))
{
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
    image_classification_msgs::LabelColours lc;

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

image_classification_msgs::PixelProbabilityList2ConstPtr
OfflineClassProbabilities::getPP(const std_msgs::HeaderConstPtr header)
{
    // we just need the image time
    const uint64_t msg_stamp = header->stamp.toNSec();

    if(pred_npy_path=="") {
        return image_classification_msgs::PixelProbabilityList2ConstPtr();
    }
    const std::string npz_dir = pred_npy_path;
    const std::string npy_path = npz_dir+"/prob_"+std::to_string(msg_stamp)+".npz_FILES";

    cnpy::NpyArray class_id_npy;
    cnpy::NpyArray coord_npy;
    cnpy::NpyArray prob_npy;
    try {
        class_id_npy = cnpy::npy_gzload(npy_path+"/class_id.npy");
        coord_npy = cnpy::npy_gzload(npy_path+"/coord.npy");
        prob_npy = cnpy::npy_gzload(npy_path+"/prob.npy");
    }
    catch(const cnpy::cnpy_error &e) {
//        std::cerr << e.what() << std::endl;
//        std::cerr << npy_path << std::endl;
        return image_classification_msgs::PixelProbabilityList2ConstPtr();
    }

    image_classification_msgs::PixelProbabilityList2Ptr pp_msg(new image_classification_msgs::PixelProbabilityList2());
    pp_msg->header = *header;
    if(coord_npy.fortran_order==false) {
        // is row-major
        pp_msg->coordinates.assign((uint16_t*)coord_npy.data, (uint16_t*)coord_npy.data+coord_npy.shape[0]*coord_npy.shape[1]);
    }
    else {
        // is column-major, convert to row major
        const Matui16XXcm m_cm = Eigen::Map<Matui16XXcm>((uint16_t*)coord_npy.data, coord_npy.shape[0], coord_npy.shape[1]).transpose();
        Matui16XXrm m_rm = Matui16XXrm(coord_npy.shape[0],coord_npy.shape[1]);
        m_rm.col(0) = m_cm.row(0);
        m_rm.col(1) = m_cm.row(1);
        pp_msg->coordinates.assign((uint16_t*)m_rm.data(), (uint16_t*)m_rm.data()+(coord_npy.shape[0]*coord_npy.shape[1]));
    }
    if(prob_npy.fortran_order==false) {
        pp_msg->probability.assign((float*)prob_npy.data, (float*)prob_npy.data+(prob_npy.shape[0]*prob_npy.shape[1]));
    }
    else {
        throw std::runtime_error("prob_npy is in column-major order");
    }

    if(link_names.size()==0) {
        throw std::runtime_error("link names not loaded, run 'setLabelColour' first");
    }

    pp_msg->link_names = link_names;

    // check if last class is 100 (occlusion)
    const uint8_t l = ((uint8_t*)class_id_npy.data)[class_id_npy.shape[0]-1];
    if(l == 100) {
        pp_msg->link_names.push_back("occlusion");
    }

    class_id_npy.destruct();
    coord_npy.destruct();
    prob_npy.destruct();

    return pp_msg;
}

void OfflineClassProbabilities::cb(const sensor_msgs::CompressedImageConstPtr& img_msg) {
    std_msgs::HeaderConstPtr header_ptr(new std_msgs::Header(img_msg->header));
    pub_class_prob.publish(getPP(header_ptr));
}
