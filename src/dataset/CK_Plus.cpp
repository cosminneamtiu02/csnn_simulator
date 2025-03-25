#include "dataset/CK_Plus.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <iomanip> // For std::setw and std::setfill
#include <random>
#include <algorithm>

namespace dataset {

CK_Plus::CK_Plus(const std::string& csv_path, const std::string& images_dir, 
                 int num_folds, unsigned int random_seed, 
                 int image_width, int image_height)
    : m_csv_path(csv_path), m_images_dir(images_dir), 
      m_image_width(image_width), m_image_height(image_height),
      m_num_folds(num_folds), m_random_seed(random_seed), m_data() {
}

CK_Plus::~CK_Plus() {
}

bool CK_Plus::load() {
    std::ifstream file(m_csv_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open CSV file: " << m_csv_path << std::endl;
        return false;
    }

    // Temporary storage for all sequences before fold assignment
    std::vector<ImageSequence> all_sequences;
    
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string subject, ipostase_str, emotion_str;
        
        std::getline(ss, subject, ',');
        std::getline(ss, ipostase_str, ',');
        std::getline(ss, emotion_str, ',');
        
        int ipostase = std::stoi(ipostase_str);
        int emotion = std::stoi(emotion_str);
        
        // Create and populate the image sequence
        ImageSequence seq;
        seq.subject = subject;
        seq.ipostase = ipostase;
        seq.emotion = emotion;
        seq.frames = loadImageSequence(subject, ipostase);
        
        // Only add sequences that have frames
        if (!seq.frames.empty()) {
            all_sequences.push_back(seq);
        }
    }
    
    // Distribute sequences randomly but evenly across folds
    distributeSequences(all_sequences);
    
    return true;
}

void CK_Plus::distributeSequences(std::vector<ImageSequence>& sequences) {
    // Clear existing data
    m_data.clear();
    
    // Group sequences by emotion
    std::map<int, std::vector<ImageSequence>> sequences_by_emotion;
    for (auto& seq : sequences) {
        sequences_by_emotion[seq.emotion].push_back(seq);
    }
    
    // Seed random generator for reproducibility
    std::mt19937 rng(m_random_seed);
    
    // For each emotion, randomly distribute sequences across folds
    for (auto& [emotion, emotion_sequences] : sequences_by_emotion) {
        // Shuffle sequences to randomize distribution
        std::shuffle(emotion_sequences.begin(), emotion_sequences.end(), rng);
        
        // Distribute sequences evenly across folds
        for (size_t i = 0; i < emotion_sequences.size(); i++) {
            int fold = (i % m_num_folds) + 1; // Folds are 1-indexed
            m_data[fold][emotion].push_back(emotion_sequences[i]);
        }
    }
    
    // Print distribution statistics
    printEmotionDistribution();
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

std::string CK_Plus::getEmotionName(int emotion) const {
    switch(emotion) {
        case 1: return "Happy";
        case 2: return "Fear";
        case 3: return "Surprise";
        case 4: return "Anger";
        case 5: return "Disgust";
        case 6: return "Sadness";
        default: return "Unknown";
    }
}

std::map<int, std::map<int, int>> CK_Plus::getEmotionCounts() const {
    std::map<int, std::map<int, int>> counts;
    
    // Initialize all counters to zero for all folds and emotions
    for (int fold = 1; fold <= m_num_folds; fold++) {
        for (int emotion = 1; emotion <= getNumEmotions(); emotion++) {
            counts[fold][emotion] = 0;
        }
    }
    
    // Count sequences for each emotion in each fold
    for (const auto& [fold, emotions] : m_data) {
        for (const auto& [emotion, sequences] : emotions) {
            counts[fold][emotion] = sequences.size();
        }
    }
    
    return counts;
}

void CK_Plus::printEmotionDistribution() const {
    auto counts = getEmotionCounts();
    
    // Calculate totals
    std::map<int, int> emotion_totals;
    std::map<int, int> fold_totals;
    int grand_total = 0;
    
    for (int fold = 1; fold <= m_num_folds; fold++) {
        for (int emotion = 1; emotion <= getNumEmotions(); emotion++) {
            int count = counts[fold][emotion];
            emotion_totals[emotion] += count;
            fold_totals[fold] += count;
            grand_total += count;
        }
    }
    
    // Print header
    std::cout << "Emotion Distribution Across Folds:" << std::endl;
    std::cout << std::setw(10) << "Fold";
    for (int emotion = 1; emotion <= getNumEmotions(); emotion++) {
        std::cout << " | " << std::setw(8) << getEmotionName(emotion);
    }
    std::cout << " | " << std::setw(8) << "Total" << std::endl;
    
    // Print separator
    std::cout << std::string(10 + (getNumEmotions() + 1) * 11, '-') << std::endl;
    
    // Print counts for each fold
    for (int fold = 1; fold <= m_num_folds; fold++) {
        std::cout << std::setw(10) << fold;
        for (int emotion = 1; emotion <= getNumEmotions(); emotion++) {
            std::cout << " | " << std::setw(8) << counts.at(fold).at(emotion);
        }
        std::cout << " | " << std::setw(8) << fold_totals[fold] << std::endl;
    }
    
    // Print separator
    std::cout << std::string(10 + (getNumEmotions() + 1) * 11, '-') << std::endl;
    
    // Print totals
    std::cout << std::setw(10) << "Total";
    for (int emotion = 1; emotion <= getNumEmotions(); emotion++) {
        std::cout << " | " << std::setw(8) << emotion_totals[emotion];
    }
    std::cout << " | " << std::setw(8) << grand_total << std::endl;
}

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
