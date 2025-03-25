#include "dataset/CK_Plus.h"
#include <iostream>

int main() {
    // Number of folds and random seed
    int num_folds = 5;
    unsigned int random_seed = 42;
    
    std::cout << "Initializing CK+ dataset with " << num_folds << " folds and random seed " << random_seed << std::endl;
    
    // Initialize dataset with custom number of folds and random seed
    dataset::CK_Plus ck_plus("/home/cosmin/proiecte/datasets/CK+_TIM10/CK+_emotion.csv", 
                             "/home/cosmin/proiecte/datasets/CK+_TIM10",
                             num_folds, random_seed);
    
    // Load the dataset
    if (!ck_plus.load()) {
        std::cerr << "Failed to load CK+ dataset" << std::endl;
        return 1;
    }
    
    // Print emotion distribution across folds
    ck_plus.printEmotionDistribution();
    
    // Example of cross-validation
    for (int test_fold = 1; test_fold <= ck_plus.getNumFolds(); test_fold++) {
        std::cout << "\nCross-validation iteration " << test_fold << std::endl;
        
        // Get training and testing sequences
        auto training_sequences = ck_plus.getTrainingSequences(test_fold);
        auto testing_sequences = ck_plus.getTestSequences(test_fold);
        
        std::cout << "Training sequences: " << training_sequences.size() << std::endl;
        std::cout << "Testing sequences: " << testing_sequences.size() << std::endl;
        
        // Count emotions in training set
        std::map<int, int> emotion_counts;
        for (const auto& seq : training_sequences) {
            emotion_counts[seq.emotion]++;
        }
        
        std::cout << "Training emotion distribution:" << std::endl;
        for (int emotion = 1; emotion <= ck_plus.getNumEmotions(); emotion++) {
            std::cout << "  " << ck_plus.getEmotionName(emotion) << ": " 
                      << emotion_counts[emotion] << std::endl;
        }
        
        // Example: Convert a sequence to tensor for use in 3D convolution
        if (!testing_sequences.empty()) {
            auto tensor = ck_plus.sequenceToTensor(testing_sequences[0]);
            std::cout << "Created tensor with dimensions: " 
                      << tensor->shape().dim(1) << "x"  // width
                      << tensor->shape().dim(0) << "x"  // height
                      << tensor->shape().dim(2) << std::endl;  // depth
        }
    }
    
    return 0;
}
