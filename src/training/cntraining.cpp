/******************************************************************************
 *
 * File:        cntraining_modern.cpp (Rewritten version of cntraining.cpp)
 * Purpose:     Generates normproto using modern C++ practices.
 *              Interfaces with Tesseract's C API for core functionality.
 * Author:      Based on original Tesseract code by Dan Johnson, Christy Russon
 *              C++ Rewrite Contributor: AI Assistant
 *
 * Original License (Apache 2.0):
 * (c) Copyright Hewlett-Packard Company, 1988.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *****************************************************************************/

// Define required for M_PI in <cmath> on some platforms
#define _USE_MATH_DEFINES

// Standard C++ Headers
#include <cmath>       // for M_PI, std::pow, std::sqrt, std::log, std::max, std::min
#include <cstdio>      // for FILE, fopen, fprintf, fclose, sscanf, perror
#include <cstring>     // for memset
#include <vector>      // for std::vector
#include <string>      // for std::string
#include <memory>      // for std::unique_ptr (used in ClustererWrapper)
#include <stdexcept>   // for std::exception, std::runtime_error
#include <iostream>    // for std::cerr, std::cout
#include <filesystem>  // Requires C++17 for path manipulation
#include <utility>     // for std::pair, std::move

// Tesseract C API Headers (Ensure include paths are set correctly)
#include <tesseract/baseapi.h> // For CheckSharedLibraryVersion, TessBaseAPI::Version()

#include "cluster.h"         // PROTOTYPE, CLUSTERER, CLUSTERCONFIG, FreeProtoList, FreeClusterer, MakeClusterer, MakeSample, ClusterSamples
#include "clusttool.h"       // WritePrototype, WriteParamDesc
#include "commontraining.h"  // ReadTrainingSamples, FreeTrainingSamples, LABELEDLIST, FEATURE_DEFS_STRUCT, ParseArguments, SetUpForClustering, Config (global)
#include "featdefs.h"        // InitFeatureDefs, ShortNameToFeatureType, FEATURE_DEFS_STRUCT, PARAM_DESC
#include "intproto.h"        // Needed by commontraining.h potentially
#include "oldlist.h"         // For the LIST type definition and iterate macros (crucial for C API interaction)
#include "params.h"          // For command line flags (FLAGS_D)
#include "unicharset.h"      // Needed by commontraining.h

// Define the feature type for this program
#define PROGRAM_FEATURE_TYPE "cn"

// Filesystem namespace alias (requires C++17)
namespace fs = std::filesystem;
// Using Tesseract's namespace
using namespace tesseract;

// --- RAII Wrapper for CLUSTERER ---
// Manages the CLUSTERER* pointer lifecycle
class ClustererWrapper {
    CLUSTERER* clusterer_ = nullptr;

public:
    // Constructor takes ownership of the pointer
    explicit ClustererWrapper(CLUSTERER* clusterer) : clusterer_(clusterer) {
        if (!clusterer_) {
             // Warning is acceptable here, the caller should check validity
             fprintf(stderr, "Warning: ClustererWrapper received nullptr!\n");
        }
    }

    // Destructor automatically frees the resource via the Tesseract C API function
    ~ClustererWrapper() {
        if (clusterer_) {
            // fprintf(stderr, "DEBUG: Auto-freeing Clusterer %p\n", clusterer_);
            FreeClusterer(clusterer_);
        }
    }

    // --- Resource Management Semantics ---
    // Prevent copying to avoid double freeing
    ClustererWrapper(const ClustererWrapper&) = delete;
    ClustererWrapper& operator=(const ClustererWrapper&) = delete;

    // Allow moving ownership
    ClustererWrapper(ClustererWrapper&& other) noexcept : clusterer_(other.clusterer_) {
        other.clusterer_ = nullptr; // Source object relinquishes ownership
    }
    ClustererWrapper& operator=(ClustererWrapper&& other) noexcept {
        if (this != &other) {
            // Free existing resource if any
            if (clusterer_) {
                FreeClusterer(clusterer_);
            }
            // Transfer ownership
            clusterer_ = other.clusterer_;
            other.clusterer_ = nullptr;
        }
        return *this;
    }

    // --- Access and Validity ---
    // Get the underlying raw pointer (use with caution)
    CLUSTERER* get() const {
        return clusterer_;
    }

    // Check if the wrapper holds a valid (non-null) pointer
    explicit operator bool() const {
        return clusterer_ != nullptr;
    }
};


// --- C++ Structures to Hold COPIED Data ---

// Holds COPIED data from a C PROTOTYPE structure
struct ProtoDataCpp {
    bool significant = false;
    unsigned style = spherical; // Default to spherical, consistent with PROTOTYPE
    unsigned num_samples = 0;
    std::vector<float> mean;
    std::vector<tesseract::DISTRIBUTION> distrib; // Only used for mixed style
    float total_magnitude = 0.0f;
    float log_magnitude = 0.0f;
    // Variance/Magnitude/Weight use FLOATUNION in C, store appropriately here
    float spherical_variance = 0.0f;
    std::vector<float> elliptical_variance; // Only valid if style is elliptical/mixed
    float spherical_magnitude = 0.0f;       // (Calculated, not directly in file)
    std::vector<float> elliptical_magnitude;// (Calculated, not directly in file)
    float spherical_weight = 0.0f;          // (Calculated, not directly in file)
    std::vector<float> elliptical_weight;   // (Calculated, not directly in file)

    // Constructor to deep copy data from a C PROTOTYPE
    explicit ProtoDataCpp(const PROTOTYPE* proto_c) {
        if (!proto_c) return; // Handle null input gracefully

        significant = proto_c->Significant;
        style = proto_c->Style;
        num_samples = proto_c->NumSamples;
        mean = proto_c->Mean;           // Deep copy std::vector
        distrib = proto_c->Distrib;     // Deep copy std::vector

        total_magnitude = proto_c->TotalMagnitude; // Copy calculated value
        log_magnitude = proto_c->LogMagnitude;     // Copy calculated value

        // Copy Variance based on style. Magnitude/Weight are often recalculated or not stored.
        if (style == spherical) {
            spherical_variance = proto_c->Variance.Spherical;
            // Recalculate derived fields if needed, though WritePrototype only uses Variance
            spherical_magnitude = (spherical_variance > 0) ? (1.0f / std::sqrt(2.0f * M_PI * spherical_variance)) : 0.0f;
            spherical_weight = (spherical_variance > 0) ? (1.0f / spherical_variance) : 0.0f;

        } else { // elliptical or mixed
             if (proto_c->Variance.Elliptical != nullptr && !mean.empty()) {
                 // Assuming size matches mean size, as calculated in NewEllipticalProto
                 size_t n_dims = mean.size();
                 elliptical_variance.assign(proto_c->Variance.Elliptical, proto_c->Variance.Elliptical + n_dims);
                 // Recalculate derived fields if needed
                 elliptical_magnitude.resize(n_dims);
                 elliptical_weight.resize(n_dims);
                 for(size_t i=0; i<n_dims; ++i) {
                     elliptical_magnitude[i] = (elliptical_variance[i] > 0) ? (1.0f / std::sqrt(2.0f * M_PI * elliptical_variance[i])) : 0.0f;
                     elliptical_weight[i] = (elliptical_variance[i] > 0) ? (1.0f / elliptical_variance[i]) : 0.0f;
                 }

             } else {
                // Handle case where Elliptical pointer is null (shouldn't happen for these styles?)
                fprintf(stderr, "Warning: Elliptical/Mixed prototype %p has null variance pointer!\n", proto_c);
             }
        }
    }

    // Default constructor needed for vector operations
    ProtoDataCpp() = default;
};

// Holds essential info for a character class and a pointer to the original C data.
// After clustering, it will hold the copied resulting prototypes.
struct LabeledProtoListCpp {
    std::string label;
    int sample_count = 0;
    // Store pointer to the original C structure containing the raw FEATURE_SET list
    // Needed as input for SetUpForClustering
    const LABELEDLISTNODE* original_c_list_node = nullptr; // Read-only pointer

    // Store the COPIED prototype data resulting from clustering
    std::vector<ProtoDataCpp> prototypes;

    // Constructor stores label, count, and pointer to original C data.
    // Does NOT copy prototypes initially.
    explicit LabeledProtoListCpp(const LABELEDLISTNODE* labeled_list_c) :
        original_c_list_node(labeled_list_c) // Store original pointer
    {
        if (!labeled_list_c) return;
        label = labeled_list_c->Label; // Copy std::string
        sample_count = labeled_list_c->SampleCount; // Copy initial sample count
    }

    // Default constructor
    LabeledProtoListCpp() = default;

    // --- Rule of 5/6 (Modern C++) ---
    // Prevent copying pointers accidentally after construction
    LabeledProtoListCpp(const LabeledProtoListCpp&) = delete;
    LabeledProtoListCpp& operator=(const LabeledProtoListCpp&) = delete;
    // Allow moving (transfers ownership of potentially large prototype vector)
    LabeledProtoListCpp(LabeledProtoListCpp&& other) noexcept = default;
    LabeledProtoListCpp& operator=(LabeledProtoListCpp&& other) noexcept = default;
    // Default destructor is fine as members manage their own resources (std::string, std::vector)

    // Helper method to count significant prototypes *after* clustering results are copied
    size_t num_significant_protos() const {
        size_t count = 0;
        for(const auto& p_data : prototypes) {
            if (p_data.significant) {
                count++;
            }
        }
        return count;
    }
};

// --- Static Helper Function Prototypes ---
// Reads training samples using C API, wraps C pointers in C++ structs. Returns C list head too.
static std::pair<std::vector<LabeledProtoListCpp>, LIST> ReadAndWrapTrainingSamples(
    const FEATURE_DEFS_STRUCT& feature_defs,
    const char** filenames);

// Writes the normproto file using C++ data structures.
static void WriteNormProtosCpp(const std::string& directory,
                               const std::vector<LabeledProtoListCpp>& labeled_proto_list_cpp,
                               const FEATURE_DESC_STRUCT *feature_desc);

// Helper for WriteNormProtosCpp - writes individual prototypes.
static void WriteProtosCpp(FILE *file, uint16_t n_params, // n = NumParams
                           const std::vector<ProtoDataCpp>& proto_list_cpp,
                           bool write_sig_protos, bool write_insig_protos);


// --- Global Clustering Configuration ---
// Initialized with default values from original code comment or commonTraining.h defaults
// This is extern in commontraining.h and defined in commontraining.cpp. We don't
// need a static copy here, but we use CNConfig below for defaults before ParseArgs.
static CLUSTERCONFIG CNConfigDefaults = {elliptical, 0.025, 0.05, 0.8, 1e-3, 0};


// --- Main Function ---
int main(int argc, char *argv[]) {
    LIST original_char_list_c = NIL_LIST; // Store head of C list to free later
    try {
        tesseract::CheckSharedLibraryVersion(); // Verify library compatibility

        // Set the Tesseract global Config parameters from our defaults FIRST.
        // ParseArguments will overwrite these with command-line flag values.
        // Note: We access the 'Config' declared extern in commontraining.h
        tesseract::Config = CNConfigDefaults;

        FEATURE_DEFS_STRUCT feature_defs;
        InitFeatureDefs(&feature_defs); // Initialize Tesseract's feature definitions

        // Use Tesseract's C API for argument parsing. This function reads command-line
        // flags (like --clusterconfig_min_samples_fraction) AND UPDATES THE GLOBAL Config struct.
        ParseArguments(&argc, &argv); // argv should now point to the first filename

        // Optionally apply clamping to the global Config values AFTER ParseArguments,
        // mimicking the behavior in the original commontraining.cpp ParseArguments.
        tesseract::Config.MinSamples = std::max(0.0, std::min(1.0, tesseract::Config.MinSamples));
        tesseract::Config.MaxIllegal = std::max(0.0, std::min(1.0, tesseract::Config.MaxIllegal));
        tesseract::Config.Independence = std::max(0.0, std::min(1.0, tesseract::Config.Independence));
        tesseract::Config.Confidence = std::max(0.0, std::min(1.0, tesseract::Config.Confidence));

        // Read training samples using C API, wrap results in C++ structs,
        // but keep the original C LIST alive for SetUpForClustering.
        auto read_result = ReadAndWrapTrainingSamples(feature_defs, const_cast<const char**>(argv));
        std::vector<LabeledProtoListCpp> char_list_cpp = std::move(read_result.first);
        original_char_list_c = read_result.second; // Keep track of original C list head

        if (char_list_cpp.empty()) {
             std::cerr << "Error: No training samples loaded." << std::endl;
             // Need to free C list if it was allocated but resulted in empty C++ list
             if (original_char_list_c != NIL_LIST) FreeTrainingSamples(original_char_list_c);
             return EXIT_FAILURE;
        }

        std::cout << "Clustering " << char_list_cpp.size() << " characters..." << std::endl;

        // This vector will hold the final clustered results (as C++ copies)
        std::vector<LabeledProtoListCpp> norm_proto_list_cpp;

        // --- Main Clustering Loop ---
        for (const auto& char_sample_cpp : char_list_cpp) { // Iterate C++ wrappers
            std::cout << "Clustering for: " << char_sample_cpp.label << std::endl;

            // Get the pointer to the original C structure needed by SetUpForClustering
            if (!char_sample_cpp.original_c_list_node) {
                std::cerr << "Error: Original C node missing for " << char_sample_cpp.label << std::endl;
                continue; // Skip this character if data is inconsistent
            }
            // SetUpForClustering takes LABELEDLIST (which is LABELEDLISTNODE*)
            // Cast away constness is safe as we verified SetUpForClustering only reads.
            LABELEDLIST c_list_for_setup = const_cast<LABELEDLIST>(char_sample_cpp.original_c_list_node);

            // Set up the clusterer using the C API, managed by RAII wrapper
            ClustererWrapper clusterer_wrapper(
                SetUpForClustering(feature_defs, c_list_for_setup, PROGRAM_FEATURE_TYPE));

            if (!clusterer_wrapper) { // Check if SetUpForClustering returned null
                std::cerr << "Error: SetUpForClustering failed for " << char_sample_cpp.label << "!" << std::endl;
                continue; // Skip to next character
            }

            // Store original MinSamples, as it might be modified in the retry loop
            // Use the global Config struct directly (already updated by ParseArguments)
            float saved_min_samples = tesseract::Config.MinSamples;
            // Set MagicSamples based on the initial sample count for this character
            tesseract::Config.MagicSamples = char_sample_cpp.sample_count;

            std::vector<ProtoDataCpp> clustered_protos_cpp; // Holds copied results for this char
            bool cluster_success = false;

            // --- Clustering Retry Loop ---
            // Use the global Config struct directly here as well
            while (tesseract::Config.MinSamples > 0.001 && !cluster_success) {
                fprintf(stderr, "DEBUG: Calling ClusterSamples for %s with MinSamples=%.4f\n",
                        char_sample_cpp.label.c_str(), tesseract::Config.MinSamples);

                // Call the core C clustering function, passing address of global Config
                LIST c_result_list = ClusterSamples(clusterer_wrapper.get(), &tesseract::Config);

                clustered_protos_cpp.clear(); // Clear results from previous retry attempt
                size_t num_significant = 0;

                if (c_result_list != NIL_LIST) {
                    // Iterate the C list result and COPY data into C++ vector
                    LIST current_proto_node = c_result_list;
                    iterate(current_proto_node) {
                        PROTOTYPE* proto_c = reinterpret_cast<PROTOTYPE*>(current_proto_node->first_node());
                        if (proto_c) {
                            // Calls ProtoDataCpp copy constructor
                            clustered_protos_cpp.emplace_back(proto_c);
                            if (proto_c->Significant) {
                                num_significant++;
                            }
                        }
                    }
                    // IMPORTANT: Free the C list and C prototypes returned by ClusterSamples
                    // Pass address because FreeProtoList takes LIST* and modifies it (sets to NIL).
                    FreeProtoList(&c_result_list);
                } // else: c_result_list was NIL, nothing to copy or free

                fprintf(stderr, "DEBUG: ClusterSamples produced %zu significant protos\n", num_significant);

                if (num_significant > 0) {
                    cluster_success = true; // Found significant protos, exit retry loop
                } else {
                    // No significant protos, reduce threshold and retry
                    fprintf(stderr, "Warning: 0 significant protos for %s. Retrying...\n", char_sample_cpp.label.c_str());
                    tesseract::Config.MinSamples *= 0.95; // Decrease MinSamples requirement
                }
            } // End retry loop
            tesseract::Config.MinSamples = saved_min_samples; // Restore original MinSamples setting

            // --- Process Clustering Result ---
            if (cluster_success) {
                // Create a new entry for the final list and move the copied protos into it
                LabeledProtoListCpp final_entry;
                final_entry.label = char_sample_cpp.label;
                // Use the original sample count? Or size of clustered_protos_cpp? Depends on normproto needs.
                final_entry.sample_count = char_sample_cpp.sample_count;
                final_entry.prototypes = std::move(clustered_protos_cpp); // Move vector of copies
                // original_c_list_node pointer is not needed in the final output list
                norm_proto_list_cpp.push_back(std::move(final_entry));
            } else {
                std::cerr << "Error: Failed to generate significant protos for "
                          << char_sample_cpp.label << " even after retries." << std::endl;
            }
            // clusterer_wrapper goes out of scope here, automatically calling FreeClusterer via destructor
        } // End loop over characters


        // --- Write Output ---
        // Find the feature description index needed for writing the normproto header
        int desc_index = ShortNameToFeatureType(feature_defs, PROGRAM_FEATURE_TYPE);
        // Corrected access to NumFeatureDescs (removed leading space)
        if (desc_index < 0 || desc_index >= feature_defs.NumFeatureDescs) {
             std::cerr << "Error: Feature type '" << PROGRAM_FEATURE_TYPE << "' not found!" << std::endl;
             // Need to free C list before exiting
             if (original_char_list_c != NIL_LIST) FreeTrainingSamples(original_char_list_c);
             return EXIT_FAILURE;
        }

        // Get output directory from command-line flags (parsed by ParseArguments into global FLAGS_D)
        std::string output_dir = FLAGS_D.c_str();

        // Write the normproto file using the collected C++ data structures
        WriteNormProtosCpp(output_dir, norm_proto_list_cpp, feature_defs.FeatureDesc[desc_index]);

        std::cout << "\nProcessing finished successfully." << std::endl;

    // --- Exception Handling ---
    } catch (const std::exception& e) {
        std::cerr << "\nError: Exception caught: " << e.what() << std::endl;
        // Ensure C memory is freed even if an exception occurs
        if (original_char_list_c != NIL_LIST) {
             fprintf(stderr, "DEBUG: Calling FreeTrainingSamples on C LIST %p (exception path: %s)\n",
                     original_char_list_c, e.what());
             FreeTrainingSamples(original_char_list_c);
        }
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "\nError: Unknown exception caught!" << std::endl;
        // Ensure C memory is freed
        if (original_char_list_c != NIL_LIST) {
             fprintf(stderr, "DEBUG: Calling FreeTrainingSamples on C LIST %p (unknown exception path)\n",
                     original_char_list_c);
             FreeTrainingSamples(original_char_list_c);
        }
        return EXIT_FAILURE;
    }

    // --- Final C Memory Cleanup (Normal Exit Path) ---
    // This is crucial because the C list was kept alive during the main processing.
    if (original_char_list_c != NIL_LIST) {
         fprintf(stderr, "DEBUG: Calling FreeTrainingSamples on C LIST %p (normal exit path)\n",
                 original_char_list_c);
         FreeTrainingSamples(original_char_list_c);
         original_char_list_c = NIL_LIST; // Avoid potential double free if error occurs later
    }

    return EXIT_SUCCESS;

} // main


// --- Static Helper Function Implementations ---

/**
 * Reads training samples using Tesseract's C API, creates C++ wrapper objects
 * that store pointers to the original C data structures, and returns both the
 * vector of C++ wrappers and the head of the original C LIST.
 *
 * @param feature_defs Tesseract feature definitions.
 * @param filenames Null-terminated list of input .tr filenames.
 * @return A pair containing:
 *         1. std::vector<LabeledProtoListCpp> holding wrappers.
 *         2. LIST pointing to the head of the original C list (must be freed later).
 */
static std::pair<std::vector<LabeledProtoListCpp>, LIST> ReadAndWrapTrainingSamples(
    const FEATURE_DEFS_STRUCT& feature_defs,
    const char** filenames)
{
    LIST char_list_c = NIL_LIST; // Head of the C LIST populated by ReadTrainingSamples

    // Loop through input filenames and call the C API reading function
    for (const char *page_name = *filenames; page_name != nullptr; page_name = *++filenames) {
        std::cout << "Reading " << page_name << " ..." << std::endl;
        // Use TFile for potentially better platform compatibility if available?
        // Original uses FILE*, sticking to that for now.
        FILE *training_page = fopen(page_name, "rb");
        if (!training_page) {
            std::cerr << "Warning: Cannot open training page " << page_name << std::endl;
            continue; // Skip this file
        }
        // ReadTrainingSamples appends to the list passed by pointer (&char_list_c)
        // Max samples set to 0 means read all.
        ReadTrainingSamples(feature_defs, PROGRAM_FEATURE_TYPE, 0 /*max_samples*/,
                            nullptr /*unicharset ptr, not needed here*/, training_page, &char_list_c);
        fclose(training_page);
    }

    std::vector<LabeledProtoListCpp> result_vector; // Vector of C++ wrappers
    if (char_list_c == NIL_LIST) {
         std::cerr << "Warning: No samples read from any file." << std::endl;
         // Return empty vector and NIL list pointer
         return {std::move(result_vector), char_list_c};
    }

    // Iterate over the C LIST and create C++ wrapper objects
    LIST current_node = char_list_c;
    iterate(current_node) {
        // Get pointer to the C structure stored in the list node
        LABELEDLISTNODE* labeled_list_c = reinterpret_cast<LABELEDLISTNODE*>(current_node->first_node());
        if (labeled_list_c) {
            // Create a LabeledProtoListCpp wrapper. The constructor copies the label,
            // count, and importantly, stores the pointer labeled_list_c.
            result_vector.emplace_back(labeled_list_c);
        }
    }

    // DO NOT FREE the C list here! It's needed for SetUpForClustering.
    // Return the C++ wrapper vector and the head of the C LIST.
    return {std::move(result_vector), char_list_c};
}


/**
 * Writes the final normalized prototypes (stored in C++ data structures)
 * to the "normproto" file in the specified directory.
 *
 * @param directory Output directory path (string).
 * @param labeled_proto_list_cpp Vector containing the clustered results.
 * @param feature_desc Tesseract description for the feature type used.
 */
static void WriteNormProtosCpp(const std::string& directory,
                               const std::vector<LabeledProtoListCpp>& labeled_proto_list_cpp,
                               const FEATURE_DESC_STRUCT *feature_desc)
{
    fs::path output_path;
    // Construct the output path using C++17 filesystem library
    try {
        if (!directory.empty()) {
            output_path = directory;
            // Ensure output directory exists, creating it if necessary
            if (!fs::exists(output_path)) {
                 fprintf(stderr, "DEBUG: Creating output directory: %s\n", output_path.string().c_str());
                fs::create_directories(output_path);
            } else if (!fs::is_directory(output_path)) {
                 // Handle case where path exists but isn't a directory
                 std::cerr << "\nError: Output path exists but is not a directory: " << directory << std::endl;
                 output_path = "."; // Fallback to current directory
                 std::cerr << "Writing to current directory instead." << std::endl;
            }
        } else {
            output_path = "."; // Default to current directory if none specified
        }
    } catch (const std::exception& e) {
        // Catch filesystem errors (e.g., permissions)
        std::cerr << "\nError accessing/creating directory '" << directory << "': " << e.what() << std::endl;
        output_path = "."; // Fallback to current directory
        std::cerr << "Writing to current directory instead." << std::endl;
    }

    output_path /= "normproto"; // Append the standard filename

    std::cout << "\nWriting " << output_path.string() << " ..." << std::flush;
    // Open file using C stdio. Using "w" (text mode) as normproto is textual.
    FILE *file = fopen(output_path.string().c_str(), "w");

    if (!file) {
        std::cerr << "\nError: Cannot open output file '" << output_path.string() << "' for writing!" << std::endl;
        perror("fopen failed"); // Print system error message
        return; // Or throw std::runtime_error("Failed to open normproto file");
    }

    // --- Write Header ---
    // Use Tesseract's C API function to write parameter descriptions
    fprintf(file, "%d\n", feature_desc->NumParams);
    WriteParamDesc(file, feature_desc->NumParams, feature_desc->ParamDesc);

    // --- Write Data for Each Character Class ---
    for (const auto& labeled_proto_cpp : labeled_proto_list_cpp) {
        // Use helper method on C++ struct to count significant prototypes
        size_t num_significant = labeled_proto_cpp.num_significant_protos();

        if (num_significant < 1) {
            // If clustering failed or yielded no significant protos for a char
            std::cout << "\nWarning: Skipping character '" << labeled_proto_cpp.label
                      << "' in normproto file - 0 significant protos found." << std::endl;
            continue; // Skip writing this character to the file
        }

        // Write the character label and the count of significant prototypes
        fprintf(file, "\n%s %zu\n", labeled_proto_cpp.label.c_str(), num_significant); // Use %zu for size_t

        // Call helper to write the actual prototype data for this character
        WriteProtosCpp(file, feature_desc->NumParams, labeled_proto_cpp.prototypes,
                       true,  // Write significant protos
                       false); // Do not write insignificant protos
    }

    fclose(file);
    std::cout << " Done." << std::endl;
}


/**
 * Helper function for WriteNormProtosCpp. Writes the individual prototype
 * data for a single character class to the output file. It reconstructs
 * temporary C PROTOTYPE structs on the stack to pass to the underlying
 * Tesseract C API function WritePrototype.
 *
 * @param file Open FILE pointer for the output file.
 * @param n_params Number of parameters (dimensions) in the feature space.
 * @param proto_list_cpp Vector containing the copied prototype data (ProtoDataCpp).
 * @param write_sig_protos Flag to write significant prototypes.
 * @param write_insig_protos Flag to write insignificant prototypes.
 */
static void WriteProtosCpp(FILE *file, uint16_t n_params,
                           const std::vector<ProtoDataCpp>& proto_list_cpp,
                           bool write_sig_protos, bool write_insig_protos)
{
    for (const auto& proto_data_cpp : proto_list_cpp) { // Iterate over copied C++ protos

        // Check if this prototype should be written based on significance flags
        if ((proto_data_cpp.significant && write_sig_protos) ||
            (!proto_data_cpp.significant && write_insig_protos))
        {
            // --- Reconstruct a temporary C PROTOTYPE on the stack ---
            PROTOTYPE temp_proto_c;
            memset(&temp_proto_c, 0, sizeof(temp_proto_c)); // Zero out structure

            // Copy fields required by WritePrototype from the C++ struct
            temp_proto_c.Significant = proto_data_cpp.significant;
            temp_proto_c.Style = proto_data_cpp.style;
            temp_proto_c.NumSamples = proto_data_cpp.num_samples;
            temp_proto_c.Cluster = nullptr; // WritePrototype doesn't use this field
            temp_proto_c.Distrib = proto_data_cpp.distrib; // Copy vector (used only if Style==mixed)
            temp_proto_c.Mean = proto_data_cpp.mean;       // Copy vector

            // Only Variance field is needed by WritePrototype
            std::vector<float> temp_ellip_var; // Temporary storage if needed

            if (temp_proto_c.Style == spherical) {
                temp_proto_c.Variance.Spherical = proto_data_cpp.spherical_variance;
                temp_proto_c.Variance.Elliptical = nullptr; // Ensure pointer is null
            } else { // Elliptical or mixed
                temp_proto_c.Variance.Spherical = 0.0f; // Set default

                // Point the Elliptical pointer to the data in the copied C++ vector.
                temp_ellip_var = proto_data_cpp.elliptical_variance; // Copy vector content
                // Point to the data buffer of the temporary vector
                temp_proto_c.Variance.Elliptical = temp_ellip_var.empty() ? nullptr : temp_ellip_var.data();
            }
            // --- End Reconstruction ---

            // Call the Tesseract C API function to write this single prototype
            WritePrototype(file, n_params, &temp_proto_c);
        }
    } // End loop over prototypes for this character
}
