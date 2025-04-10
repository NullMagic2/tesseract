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
#include <cmath>       // for M_PI, std::pow, std::sqrt, std::log
#include <cstdio>      // for FILE, fopen, fprintf, fclose, sscanf, perror
#include <cstring>     // for memset
#include <vector>      // for std::vector
#include <string>      // for std::string
#include <memory>      // for std::unique_ptr (used in ClustererWrapper)
#include <stdexcept>   // for std::exception, std::runtime_error
#include <iostream>    // for std::cerr, std::cout
#include <filesystem>  // Requires C++17 for path manipulation
#include <utility>     // for std::pair, std::move
#include <algorithm>   // Needed for std::max and std::min

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
             fprintf(stderr, "Warning: ClustererWrapper received nullptr!\n");
        }
    }

    // Destructor automatically frees the resource via the Tesseract C API function
    ~ClustererWrapper() {
        if (clusterer_) {
            FreeClusterer(clusterer_);
        }
    }

    // Prevent copying to avoid double free.
    ClustererWrapper(const ClustererWrapper&) = delete;
    ClustererWrapper& operator=(const ClustererWrapper&) = delete;

    // Allow moving ownership
    ClustererWrapper(ClustererWrapper&& other) noexcept : clusterer_(other.clusterer_) {
        other.clusterer_ = nullptr;
    }
    ClustererWrapper& operator=(ClustererWrapper&& other) noexcept {
        if (this != &other) {
            if (clusterer_) {
                FreeClusterer(clusterer_);
            }
            clusterer_ = other.clusterer_;
            other.clusterer_ = nullptr;
        }
        return *this;
    }

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

struct ProtoDataCpp {
    bool significant = false;
    unsigned style = spherical; // Default to spherical.
    unsigned num_samples = 0;
    std::vector<float> mean;
    std::vector<tesseract::DISTRIBUTION> distrib; // Only used for mixed style.
    float total_magnitude = 0.0f;
    float log_magnitude = 0.0f;
    float spherical_variance = 0.0f;
    std::vector<float> elliptical_variance;
    float spherical_magnitude = 0.0f;
    std::vector<float> elliptical_magnitude;
    float spherical_weight = 0.0f;
    std::vector<float> elliptical_weight;

    explicit ProtoDataCpp(const PROTOTYPE* proto_c) {
        if (!proto_c) return;

        significant = proto_c->Significant;
        style = proto_c->Style;
        num_samples = proto_c->NumSamples;
        mean = proto_c->Mean;
        distrib = proto_c->Distrib;

        total_magnitude = proto_c->TotalMagnitude;
        log_magnitude = proto_c->LogMagnitude;

        if (style == spherical) {
            spherical_variance = proto_c->Variance.Spherical;
            spherical_magnitude = (spherical_variance > 0) ? (1.0f / std::sqrt(2.0f * M_PI * spherical_variance)) : 0.0f;
            spherical_weight = (spherical_variance > 0) ? (1.0f / spherical_variance) : 0.0f;
        } else {
             if (proto_c->Variance.Elliptical != nullptr && !mean.empty()) {
                 size_t n_dims = mean.size();
                 elliptical_variance.assign(proto_c->Variance.Elliptical, proto_c->Variance.Elliptical + n_dims);
                 elliptical_magnitude.resize(n_dims);
                 elliptical_weight.resize(n_dims);
                 for(size_t i = 0; i < n_dims; ++i) {
                     elliptical_magnitude[i] = (elliptical_variance[i] > 0) ? (1.0f / std::sqrt(2.0f * M_PI * elliptical_variance[i])) : 0.0f;
                     elliptical_weight[i] = (elliptical_variance[i] > 0) ? (1.0f / elliptical_variance[i]) : 0.0f;
                 }
             } else {
                fprintf(stderr, "Warning: Elliptical/Mixed prototype %p has null variance pointer!\n", proto_c);
             }
        }
    }

    ProtoDataCpp() = default;
};

struct LabeledProtoListCpp {
    std::string label;
    int sample_count = 0;
    const LABELEDLISTNODE* original_c_list_node = nullptr;
    std::vector<ProtoDataCpp> prototypes;

    explicit LabeledProtoListCpp(const LABELEDLISTNODE* labeled_list_c) :
        original_c_list_node(labeled_list_c) {
        if (!labeled_list_c) return;
        label = labeled_list_c->Label;
        sample_count = labeled_list_c->SampleCount;
    }

    LabeledProtoListCpp() = default;

    LabeledProtoListCpp(const LabeledProtoListCpp&) = delete;
    LabeledProtoListCpp& operator=(const LabeledProtoListCpp&) = delete;
    LabeledProtoListCpp(LabeledProtoListCpp&& other) noexcept = default;
    LabeledProtoListCpp& operator=(LabeledProtoListCpp&& other) noexcept = default;

    size_t num_significant_protos() const {
        size_t count = 0;
        for(const auto& p_data : prototypes)
            if (p_data.significant)
                count++;
        return count;
    }
};

// --- Static Helper Function Prototypes ---
static std::pair<std::vector<LabeledProtoListCpp>, LIST> ReadAndWrapTrainingSamples(
    const FEATURE_DEFS_STRUCT& feature_defs,
    const char** filenames);

static void WriteNormProtosCpp(const std::string& directory,
                               const std::vector<LabeledProtoListCpp>& labeled_proto_list_cpp,
                               const FEATURE_DESC_STRUCT *feature_desc);

static void WriteProtosCpp(FILE *file, uint16_t n_params,
                           const std::vector<ProtoDataCpp>& proto_list_cpp,
                           bool write_sig_protos, bool write_insig_protos);

// --- Global Clustering Configuration ---
static CLUSTERCONFIG CNConfigDefaults = {elliptical, 0.025, 0.05, 0.8, 1e-3, 0};

// --- Main Function ---
int main(int argc, char *argv[]) {
    LIST original_char_list_c = NIL_LIST;
    try {
        tesseract::CheckSharedLibraryVersion();

        tesseract::Config = CNConfigDefaults;

        FEATURE_DEFS_STRUCT feature_defs;
        InitFeatureDefs(&feature_defs);

        ParseArguments(&argc, &argv);

        // --- Corrected: Use explicit float template parameters for clamping ---
        tesseract::Config.MinSamples = std::max<float>(0.0f, std::min<float>(1.0f, tesseract::Config.MinSamples));
        tesseract::Config.MaxIllegal = std::max<float>(0.0f, std::min<float>(1.0f, tesseract::Config.MaxIllegal));
        tesseract::Config.Independence = std::max<float>(0.0f, std::min<float>(1.0f, tesseract::Config.Independence));
        tesseract::Config.Confidence = std::max<float>(0.0f, std::min<float>(1.0f, tesseract::Config.Confidence));

        auto read_result = ReadAndWrapTrainingSamples(feature_defs, const_cast<const char**>(argv + 1));
        std::vector<LabeledProtoListCpp> char_list_cpp = std::move(read_result.first);
        original_char_list_c = read_result.second;

        if (char_list_cpp.empty()) {
             std::cerr << "Error: No training samples loaded." << std::endl;
             if (original_char_list_c != NIL_LIST) FreeTrainingSamples(original_char_list_c);
             return EXIT_FAILURE;
        }

        std::cout << "Clustering " << char_list_cpp.size() << " characters..." << std::endl;
        std::vector<LabeledProtoListCpp> norm_proto_list_cpp;

        for (const auto& char_sample_cpp : char_list_cpp) {
            std::cout << "Clustering for: " << char_sample_cpp.label << std::endl;

            if (!char_sample_cpp.original_c_list_node) {
                std::cerr << "Error: Original C node missing for " << char_sample_cpp.label << std::endl;
                continue;
            }
            LABELEDLIST c_list_for_setup = const_cast<LABELEDLIST>(char_sample_cpp.original_c_list_node);

            ClustererWrapper clusterer_wrapper(
                SetUpForClustering(feature_defs, c_list_for_setup, PROGRAM_FEATURE_TYPE));

            if (!clusterer_wrapper) {
                std::cerr << "Error: SetUpForClustering failed for " << char_sample_cpp.label << "!" << std::endl;
                continue;
            }

            float saved_min_samples = tesseract::Config.MinSamples;
            tesseract::Config.MagicSamples = char_sample_cpp.sample_count;
            std::vector<ProtoDataCpp> clustered_protos_cpp;
            bool cluster_success = false;

            while (tesseract::Config.MinSamples > 0.001f && !cluster_success) {
                fprintf(stderr, "DEBUG: Calling ClusterSamples for %s with MinSamples=%.4f\n",
                        char_sample_cpp.label.c_str(), tesseract::Config.MinSamples);

                LIST c_result_list = ClusterSamples(clusterer_wrapper.get(), &tesseract::Config);
                clustered_protos_cpp.clear();
                size_t num_significant = 0;

                if (c_result_list != NIL_LIST) {
                    LIST current_proto_node = c_result_list;
                    iterate(current_proto_node) {
                        PROTOTYPE* proto_c = reinterpret_cast<PROTOTYPE*>(current_proto_node->first_node());
                        if (proto_c) {
                            clustered_protos_cpp.emplace_back(proto_c);
                            if (proto_c->Significant)
                                num_significant++;
                        }
                    }
                    FreeProtoList(&c_result_list);
                }

                fprintf(stderr, "DEBUG: ClusterSamples produced %zu significant protos\n", num_significant);
                if (num_significant > 0)
                    cluster_success = true;
                else {
                    fprintf(stderr, "Warning: 0 significant protos for %s. Retrying...\n", char_sample_cpp.label.c_str());
                    tesseract::Config.MinSamples *= 0.95f;
                }
            }
            tesseract::Config.MinSamples = saved_min_samples;

            if (cluster_success) {
                LabeledProtoListCpp final_entry;
                final_entry.label = char_sample_cpp.label;
                final_entry.sample_count = char_sample_cpp.sample_count;
                final_entry.prototypes = std::move(clustered_protos_cpp);
                norm_proto_list_cpp.push_back(std::move(final_entry));
            } else {
                std::cerr << "Error: Failed to generate significant protos for "
                          << char_sample_cpp.label << " even after retries." << std::endl;
            }
        }

        // --- Write Output ---
        // Use the number of feature types stored in feature_defs, not a non-existent member.
        int desc_index = ShortNameToFeatureType(feature_defs, PROGRAM_FEATURE_TYPE);
        if (desc_index < 0 || desc_index >= feature_defs.NumFeatureTypes) {
             std::cerr << "Error: Feature type '" << PROGRAM_FEATURE_TYPE << "' not found!" << std::endl;
             if (original_char_list_c != NIL_LIST) FreeTrainingSamples(original_char_list_c);
             return EXIT_FAILURE;
        }

        std::string output_dir = FLAGS_D.c_str();
        WriteNormProtosCpp(output_dir, norm_proto_list_cpp, feature_defs.FeatureDesc[desc_index]);

        std::cout << "\nProcessing finished successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "\nError: Exception caught: " << e.what() << std::endl;
        if (original_char_list_c != NIL_LIST) {
             fprintf(stderr, "DEBUG: Calling FreeTrainingSamples on C LIST %p (exception path: %s)\n",
                     original_char_list_c, e.what());
             FreeTrainingSamples(original_char_list_c);
        }
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "\nError: Unknown exception caught!" << std::endl;
        if (original_char_list_c != NIL_LIST) {
             fprintf(stderr, "DEBUG: Calling FreeTrainingSamples on C LIST %p (unknown exception path)\n",
                     original_char_list_c);
             FreeTrainingSamples(original_char_list_c);
        }
        return EXIT_FAILURE;
    }

    if (original_char_list_c != NIL_LIST) {
         fprintf(stderr, "DEBUG: Calling FreeTrainingSamples on C LIST %p (normal exit path)\n",
                 original_char_list_c);
         FreeTrainingSamples(original_char_list_c);
         original_char_list_c = NIL_LIST;
    }

    return EXIT_SUCCESS;
}


// --- Static Helper Function Implementations ---

static std::pair<std::vector<LabeledProtoListCpp>, LIST> ReadAndWrapTrainingSamples(
    const FEATURE_DEFS_STRUCT& feature_defs,
    const char** filenames)
{
    LIST char_list_c = NIL_LIST;

    for (const char *page_name = *filenames; page_name != nullptr; page_name = *++filenames) {
        std::cout << "Reading " << page_name << " ..." << std::endl;
        FILE *training_page = fopen(page_name, "rb");
        if (!training_page) {
            std::cerr << "Warning: Cannot open training page " << page_name << std::endl;
            continue;
        }
        ReadTrainingSamples(feature_defs, PROGRAM_FEATURE_TYPE, 0, nullptr, training_page, &char_list_c);
        fclose(training_page);
    }

    std::vector<LabeledProtoListCpp> result_vector;
    if (char_list_c == NIL_LIST) {
         std::cerr << "Warning: No samples read from any file." << std::endl;
         return {std::move(result_vector), char_list_c};
    }

    LIST current_node = char_list_c;
    iterate(current_node) {
        LABELEDLISTNODE* labeled_list_c = reinterpret_cast<LABELEDLISTNODE*>(current_node->first_node());
        if (labeled_list_c)
            result_vector.emplace_back(labeled_list_c);
    }

    return {std::move(result_vector), char_list_c};
}


static void WriteNormProtosCpp(const std::string& directory,
                               const std::vector<LabeledProtoListCpp>& labeled_proto_list_cpp,
                               const FEATURE_DESC_STRUCT *feature_desc)
{
    fs::path output_path;
    try {
        if (!directory.empty()) {
            output_path = directory;
            if (!fs::exists(output_path)) {
                 fprintf(stderr, "DEBUG: Creating output directory: %s\n", output_path.string().c_str());
                fs::create_directories(output_path);
            } else if (!fs::is_directory(output_path)) {
                 std::cerr << "\nError: Output path exists but is not a directory: " << directory << std::endl;
                 output_path = ".";
                 std::cerr << "Writing to current directory instead." << std::endl;
            }
        } else {
            output_path = ".";
        }
    } catch (const std::exception& e) {
        std::cerr << "\nError accessing/creating directory '" << directory << "': " << e.what() << std::endl;
        output_path = ".";
        std::cerr << "Writing to current directory instead." << std::endl;
    }

    output_path /= "normproto";
    std::cout << "\nWriting " << output_path.string() << " ..." << std::flush;
    FILE *file = fopen(output_path.string().c_str(), "w");

    if (!file) {
        std::cerr << "\nError: Cannot open output file '" << output_path.string() << "' for writing!" << std::endl;
        perror("fopen failed");
        return;
    }

    fprintf(file, "%d\n", feature_desc->NumParams);
    WriteParamDesc(file, feature_desc->NumParams, feature_desc->ParamDesc);

    for (const auto& labeled_proto_cpp : labeled_proto_list_cpp) {
        size_t num_significant = labeled_proto_cpp.num_significant_protos();
        if (num_significant < 1) {
            std::cout << "\nWarning: Skipping character '" << labeled_proto_cpp.label
                      << "' in normproto file - 0 significant protos found." << std::endl;
            continue;
        }
        fprintf(file, "\n%s %zu\n", labeled_proto_cpp.label.c_str(), num_significant);
        WriteProtosCpp(file, feature_desc->NumParams, labeled_proto_cpp.prototypes, true, false);
    }

    fclose(file);
    std::cout << " Done." << std::endl;
}


static void WriteProtosCpp(FILE *file, uint16_t n_params,
                           const std::vector<ProtoDataCpp>& proto_list_cpp,
                           bool write_sig_protos, bool write_insig_protos)
{
    for (const auto& proto_data_cpp : proto_list_cpp) {
        if ((proto_data_cpp.significant && write_sig_protos) ||
            (!proto_data_cpp.significant && write_insig_protos))
        {
            PROTOTYPE temp_proto_c;
            memset(&temp_proto_c, 0, sizeof(temp_proto_c));

            temp_proto_c.Significant = proto_data_cpp.significant;
            temp_proto_c.Style = proto_data_cpp.style;
            temp_proto_c.NumSamples = proto_data_cpp.num_samples;
            temp_proto_c.Cluster = nullptr;
            temp_proto_c.Distrib = proto_data_cpp.distrib;
            temp_proto_c.Mean = proto_data_cpp.mean;

            std::vector<float> temp_ellip_var;
            if (temp_proto_c.Style == spherical) {
                temp_proto_c.Variance.Spherical = proto_data_cpp.spherical_variance;
                temp_proto_c.Variance.Elliptical = nullptr;
            } else {
                temp_proto_c.Variance.Spherical = 0.0f;
                temp_ellip_var = proto_data_cpp.elliptical_variance;
                temp_proto_c.Variance.Elliptical = temp_ellip_var.empty() ? nullptr : temp_ellip_var.data();
            }
            WritePrototype(file, n_params, &temp_proto_c);
        }
    }
}
