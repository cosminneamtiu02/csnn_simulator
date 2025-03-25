#include "dataset/CK_Plus.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <iomanip> // For std::setw and std::setfill

namespace dataset {

CK_Plus::CK_Plus(const std::string& csv_path, const std::string& images_dir, int image_width, int image_height)
    : m_csv_path(csv_path), m_images_dir(images_dir), m_image_width(image_width), m_image_height(image_height), m_data() {
}

CK_Plus::~CK_Plus() {
}

bool CK_Plus::load() {
    std::ifstream file(m_csv_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open CSV file: " << m_csv_path << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string subject, ipostase_str, emotion_str, fold_str;
        
        std::getline(ss, subject, ',');
        std::getline(ss, ipostase_str, ',');
        std::getline(ss, emotion_str, ',');
        std::getline(ss, fold_str, ',');
        
        int ipostase = std::stoi(ipostase_str);
        int emotion = std::stoi(emotion_str);
        int fold = std::stoi(fold_str);
        
        // Create and populate the image sequence
        ImageSequence seq;
        seq.subject = subject;
        seq.ipostase = ipostase;
        seq.emotion = emotion;
        seq.frames = loadImageSequence(subject, ipostase);
        
        // Add to our data structure
        m_data[fold][emotion].push_back(seq);
    }
    
    return true;
}

std::vector<CK_Plus::ImageSequence> CK_Plus::getSequences(int fold, int emotion) {
    if (m_data.find(fold) != m_data.end() && m_data[fold].find(emotion) != m_data[fold].end()) {
        return m_data[fold][emotion];
    }
    return {};
}

std::vector<CK_Plus::ImageSequence> CK_Plus::getTrainingSequences(int test_fold) {
    std::vector<ImageSequence> training_sequences;
    
    for (int fold = 1; fold <= getNumFolds(); fold++) {
        if (fold != test_fold) {
            for (int emotion = 1; emotion <= getNumEmotions(); emotion++) {
                auto sequences = getSequences(fold, emotion);
                training_sequences.insert(training_sequences.end(), sequences.begin(), sequences.end());
            }
        }
    }
    
    return training_sequences;
}

std::vector<CK_Plus::ImageSequence> CK_Plus::getTestSequences(int test_fold) {
    std::vector<ImageSequence> test_sequences;
    
    for (int emotion = 1; emotion <= getNumEmotions(); emotion++) {
        auto sequences = getSequences(test_fold, emotion);
        test_sequences.insert(test_sequences.end(), sequences.begin(), sequences.end());
    }
    
    return test_sequences;
}

std::shared_ptr<Tensor<float>> CK_Plus::loadImage(const std::string& path) {
    std::cout << "Loading image: " << path << std::endl;
    
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << path << std::endl;
        return nullptr;
    }
    
    // Resize if necessary
    if (image.rows != m_image_height || image.cols != m_image_width) {
        cv::resize(image, image, cv::Size(m_image_width, m_image_height));
    }
    
    // Create shape for the tensor
    std::vector<size_t> dims = {static_cast<size_t>(m_image_height), 
                               static_cast<size_t>(m_image_width), 
                               1, 1};
    Shape shape(dims);
    
    // Convert to Tensor using the Shape constructor
    auto tensor = std::make_shared<Tensor<float>>(shape);
    
    // Copy image data to tensor using at() method
    for (int y = 0; y < m_image_height; y++) {
        for (int x = 0; x < m_image_width; x++) {
            float pixel_value = static_cast<float>(image.at<uchar>(y, x)) / 255.0f;
            tensor->at(y, x, 0, 0) = pixel_value;
        }
    }
    
    std::cout << "  → Created 2D tensor [" << m_image_height << "×" << m_image_width 
              << "] from image: " << std::filesystem::path(path).filename().string() << std::endl;
    
    return tensor;
}

std::vector<std::shared_ptr<Tensor<float>>> CK_Plus::loadImageSequence(const std::string& subject, int ipostase) {
    std::vector<std::shared_ptr<Tensor<float>>> frames;
    
    // Format ipostase with leading zeros (3-digit format: 000, 001, etc.)
    std::ostringstream ss;
    ss << std::setw(3) << std::setfill('0') << ipostase;
    std::string ipostase_str = ss.str();
    
    // Construct the directory path for this subject and ipostase with zero-padded folder name
    std::string subject_dir = m_images_dir + "/" + subject + "/" + ipostase_str;
    
    std::cout << "Loading sequence from: " << subject_dir << std::endl;
    
    try {
        // Collect all image paths first and sort them
        std::vector<std::filesystem::path> image_paths;
        
        // Iterate through all image files in the directory
        for (const auto& entry : std::filesystem::directory_iterator(subject_dir)) {
            if (entry.path().extension() == ".png" || entry.path().extension() == ".jpg") {
                image_paths.push_back(entry.path());
            }
        }
        
        // Sort paths by filename
        std::sort(image_paths.begin(), image_paths.end());
        
        // Now load images in sorted order
        for (size_t i = 0; i < image_paths.size(); i++) {
            auto frame = loadImage(image_paths[i].string());
            if (frame) {
                frames.push_back(frame);
                std::cout << "  Frame " << i << ": " << image_paths[i].filename().string() << std::endl;
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading image sequence: " << e.what() << " [" << subject_dir << "]" << std::endl;
    }
    
    std::cout << "Loaded " << frames.size() << " frames for subject " << subject 
              << ", ipostase " << ipostase << std::endl;
    
    return frames;
}

std::shared_ptr<Tensor<float>> CK_Plus::sequenceToTensor(const ImageSequence& seq) {
    if (seq.frames.empty()) {
        return nullptr;
    }
    
    int depth = seq.frames.size();
    int width = m_image_width;
    int height = m_image_height;
    
    // Use regular string instead of calling getEmotionName to avoid the error
    std::string emotion_name;
    switch(seq.emotion) {
        case 1: emotion_name = "Happy"; break;
        case 2: emotion_name = "Fear"; break;
        case 3: emotion_name = "Surprise"; break;
        case 4: emotion_name = "Anger"; break;
        case 5: emotion_name = "Disgust"; break;
        case 6: emotion_name = "Sadness"; break;
        default: emotion_name = "Unknown"; break;
    }
    
    std::cout << "Converting sequence to 3D tensor: Subject=" << seq.subject 
              << ", Ipostase=" << seq.ipostase 
              << ", Emotion=" << seq.emotion << " (" << emotion_name << ")"
              << ", Frames=" << depth << std::endl;
    
    // Create a 3D tensor using Shape
    std::vector<size_t> dims = {static_cast<size_t>(height), 
                               static_cast<size_t>(width), 
                               static_cast<size_t>(depth), 
                               1};
    Shape shape(dims);
    auto tensor = std::make_shared<Tensor<float>>(shape);
    
    // Remove metadata operations that are causing issues
    
    // Copy frame data to tensor
    for (int z = 0; z < depth; z++) {
        auto& frame = seq.frames[z];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float value = frame->at(y, x, 0, 0);
                tensor->at(y, x, z, 0) = value;
            }
        }
        
        std::cout << "  Added frame " << z << " to tensor at depth position " << z << std::endl;
    }
    
    std::cout << "Created 3D tensor for emotion " << emotion_name 
              << " with dimensions [" << height << "×" << width << "×" << depth << "×1]" << std::endl;
    
    return tensor;
}

// Remove the implementation of getEmotionName since it's causing issues

// Remove the implementation of printTensorInfo since it's causing issues

std::shared_ptr<Input> CK_Plus::sequenceToInput(const ImageSequence& seq) {
    // Suppress unused parameter warning
    (void)seq;
    
    // Since Input is an abstract class, we need a concrete implementation
    // For now, return nullptr until you implement a concrete Input class
    return nullptr;
}

std::shared_ptr<Spike> CK_Plus::sequenceToSpike(const ImageSequence& seq) {
    // Suppress unused parameter warning
    (void)seq;
    
    // Since Spike has no default constructor, we need to provide parameters
    // For now, return nullptr until you implement proper conversion to Spike
    return nullptr;
}

} // namespace dataset
