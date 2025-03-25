#include "dataset/CK_Plus.h"
#include <iostream>

int main() {
    // Initialize dataset
    dataset::CK_Plus ck_plus("/home/cosmin/proiecte/data/CK+_split/emotion-data-modified.csv", 
                             "/home/cosmin/proiecte/data/CK+_TIM10");
    
    // Load the dataset
    if (!ck_plus.load()) {
        std::cerr << "Failed to load CK+ dataset" << std::endl;
        return 1;
    }
    
    // Example of cross-validation
    for (int test_fold = 1; test_fold <= ck_plus.getNumFolds(); test_fold++) {
        std::cout << "Cross-validation iteration " << test_fold << std::endl;
        
        // Get training and testing sequences
        auto training_sequences = ck_plus.getTrainingSequences(test_fold);
        auto testing_sequences = ck_plus.getTestSequences(test_fold);
        
        std::cout << "Training sequences: " << training_sequences.size() << std::endl;
        std::cout << "Testing sequences: " << testing_sequences.size() << std::endl;
        
        // Here you would train your model with the training sequences
        // and evaluate it with the testing sequences
        
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
