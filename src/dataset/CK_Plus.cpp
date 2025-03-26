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

// Initialize static log level (0=none, 1=minimal, 2=verbose)
static const int LOG_LEVEL = 1;

// Helper macro for conditional logging
#define LOG_INFO(level, message) if (LOG_LEVEL >= level) { std::cout << message; }
#define LOG_ERROR(message) std::cerr << message;

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
        LOG_ERROR("Failed to open CSV file: " << m_csv_path << std::endl);
        return false;
    }

    LOG_INFO(1, "Loading dataset from " << m_csv_path << std::endl);
    std::vector<ImageSequence> all_sequences;
    
    std::string line;
    int line_count = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string subject, ipostase_str, emotion_str;
        
        std::getline(ss, subject, ',');
        std::getline(ss, ipostase_str, ',');
        std::getline(ss, emotion_str, ',');
        
        try {
            int ipostase = std::stoi(ipostase_str);
            int emotion = std::stoi(emotion_str);
            
            // Create and populate the image sequence
            ImageSequence seq;
            seq.subject = subject;
            seq.ipostase = ipostase;
            seq.emotion = emotion;
            
            // Use verbosity level 2 for detailed image loading logs
            seq.frames = loadImageSequence(subject, ipostase, LOG_LEVEL >= 2);
            
            // Only add sequences that have frames
            if (!seq.frames.empty()) {
                all_sequences.push_back(seq);
                LOG_INFO(1, "Added sequence: Subject=" << subject 
                           << ", Ipostase=" << ipostase 
                           << ", Emotion=" << emotion
                           << ", Frames=" << seq.frames.size() << std::endl);
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Error processing line " << line_count << ": " << e.what() << std::endl);
        }
        
        line_count++;
    }
    
    LOG_INFO(1, "Loaded " << all_sequences.size() << " valid sequences" << std::endl);
    
    // Distribute sequences randomly but evenly across folds
    distributeSequences(all_sequences);
    
    return !all_sequences.empty();
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
    if (LOG_LEVEL >= 1) {
        printEmotionDistribution();
    }
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

std::shared_ptr<Tensor<float>> CK_Plus::loadImage(const std::string& path, bool verbose) {
    if (verbose) {
        LOG_INFO(2, "Loading image: " << path << std::endl);
    }
    
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        LOG_ERROR("Failed to load image: " << path << std::endl);
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
    
    if (verbose) {
        LOG_INFO(2, "  → Created 2D tensor [" << m_image_height << "×" << m_image_width 
                << "] from image: " << std::filesystem::path(path).filename().string() << std::endl);
    }
    
    return tensor;
}

std::vector<std::shared_ptr<Tensor<float>>> CK_Plus::loadImageSequence(const std::string& subject, int ipostase, bool verbose) {
    std::vector<std::shared_ptr<Tensor<float>>> frames;
    
    // Format ipostase with leading zeros (3-digit format: 000, 001, etc.)
    std::ostringstream ss;
    ss << std::setw(3) << std::setfill('0') << ipostase;
    std::string ipostase_str = ss.str();
    
    // Construct the directory path for this subject and ipostase with zero-padded folder name
    std::string subject_dir = m_images_dir + "/" + subject + "/" + ipostase_str;
    
    if (verbose) {
        LOG_INFO(2, "Loading sequence from: " << subject_dir << std::endl);
    }
    
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
            auto frame = loadImage(image_paths[i].string(), verbose);
            if (frame) {
                frames.push_back(frame);
                if (verbose) {
                    LOG_INFO(2, "  Frame " << i << ": " << image_paths[i].filename().string() << std::endl);
                }
            }
        }
    }
    catch (const std::exception& e) {
        LOG_ERROR("Error loading image sequence: " << e.what() << " [" << subject_dir << "]" << std::endl);
    }
    
    if (verbose || frames.empty()) {
        LOG_INFO(1, "Loaded " << frames.size() << " frames for subject " << subject 
               << ", ipostase " << ipostase << std::endl);
    }
    
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
    
    LOG_INFO(1, "Converting sequence to 3D tensor: Subject=" << seq.subject 
              << ", Ipostase=" << seq.ipostase 
              << ", Emotion=" << seq.emotion << " (" << emotion_name << ")"
              << ", Frames=" << depth << std::endl);
    
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
        
        LOG_INFO(2, "  Added frame " << z << " to tensor at depth position " << z << std::endl);
    }
    
    LOG_INFO(2, "Created 3D tensor for emotion " << emotion_name 
              << " with dimensions [" << height << "×" << width << "×" << depth << "×1]" << std::endl);
    
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

std::shared_ptr<Spike> CK_Plus::sequenceToSpike(const ImageSequence& seq) {
    // Suppress unused parameter warning
    (void)seq;
    
    // Since Spike has no default constructor, we need to provide parameters
    // For now, return nullptr until you implement proper conversion to Spike
    return nullptr;
}

// Implementation of CkPlusInput class
CK_Plus_Input::CK_Plus_Input(const CK_Plus::ImageSequence& sequence, int image_width, int image_height) 
    : Input(), 
      _sequence(sequence), 
      _width(image_width), 
      _height(image_height), 
      _label(std::to_string(_sequence.emotion)), 
      _class_name(std::to_string(_sequence.emotion)),
      _current_index(0),
      _shape(createShape(sequence, image_width, image_height)) {
    
    // Validation moved to a separate method
    validateSequence();
    
    LOG_INFO(2, "Created CK+ input with shape: " 
             << _shape.dim(0) << "×" << _shape.dim(1) << "×" 
             << _shape.dim(2) << "×" << _shape.dim(3) << std::endl);
}

CK_Plus_Input::~CK_Plus_Input() {}

const Shape& CK_Plus_Input::shape() const {
    return _shape;
}

bool CK_Plus_Input::has_next() const {
    return _current_index < 1; // Only one item per sequence
}

std::pair<std::string, Tensor<float>> CK_Plus_Input::next() {
    if (!has_next()) {
        throw std::runtime_error("No more data in CK_Plus_Input");
    }
    
    try {
        Tensor<float> result(_shape);
        
        // Copy frame data to tensor
        for (size_t z = 0; z < _sequence.frames.size(); z++) {
            auto& frame = _sequence.frames[z];
            if (!frame) {
                LOG_ERROR("Warning: Null frame at position " << z << std::endl);
                continue;
            }
            
            for (int y = 0; y < _height; y++) {
                for (int x = 0; x < _width; x++) {
                    result.at(y, x, z, 0) = frame->at(y, x, 0, 0);
                }
            }
        }
        
        _current_index++;
        return std::make_pair(_label, result);
    }
    catch (const std::exception& e) {
        LOG_ERROR("Error creating tensor: " << e.what() << std::endl);
        throw;
    }
}

void CK_Plus_Input::reset() {
    _current_index = 0;
}

void CK_Plus_Input::close() {
    // Nothing to close
}

std::string CK_Plus_Input::to_string() const {
    return "CkPlusInput: Emotion=" + _label + 
           ", Frames=" + std::to_string(_sequence.frames.size());
}

// Factory method to create input from sequence
std::shared_ptr<Input> CK_Plus::createInput(const ImageSequence& seq) {
    return std::make_shared<CK_Plus_Input>(seq, m_image_width, m_image_height);
}

Shape CK_Plus_Input::createShape(const CK_Plus::ImageSequence& sequence, int width, int height) {
    // Initialize shape with explicit size_t values to avoid unexpected conversions
    size_t h = static_cast<size_t>(height);
    size_t w = static_cast<size_t>(width);
    size_t d = sequence.frames.empty() ? 1 : static_cast<size_t>(sequence.frames.size());
    size_t c = 1; // channels
    
    return Shape({h, w, d, c});
}

void CK_Plus_Input::validateSequence() {
    // Verify that we have valid dimensions and non-empty frames
    if (_sequence.frames.empty()) {
        throw std::runtime_error("Sequence has no frames");
    }
    
    // Make sure all dimensions are valid (non-zero)
    if (_width <= 0 || _height <= 0) {
        throw std::runtime_error("Invalid dimensions: width=" + std::to_string(_width) + 
                                ", height=" + std::to_string(_height));
    }
}

} // namespace dataset
