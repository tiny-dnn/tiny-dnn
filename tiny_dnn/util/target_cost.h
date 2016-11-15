#pragma once
#include "tiny_dnn/util/util.h"
#include <numeric> // std::accumulate

namespace tiny_dnn {

// calculate the number of samples for each class label
//  - for example, if there are 10 samples having label 0, and
//    20 samples having label 1, returns a vector [10, 20]
inline std::vector<serial_size_t> calculate_label_counts(const std::vector<label_t>& t) {
    std::vector<serial_size_t> label_counts;
    for (label_t label : t) {
        if (label >= label_counts.size()) {
            label_counts.resize(label + 1);
        }
        label_counts[label]++;
    }
    assert(std::accumulate(label_counts.begin(), label_counts.end(), static_cast<serial_size_t>(0)) == t.size());
    return label_counts;
}

// calculate the weight of a given sample needed for a balanced target cost
// NB: we call a target cost matrix "balanced", if the cost of each *class* is equal
//     (this happens when the product weight * sample count is equal between the different
//      classes, and the sum of these products equals the total number of samples)
inline float_t get_sample_weight_for_balanced_target_cost(serial_size_t classes, serial_size_t total_samples, serial_size_t this_class_samples)
{
    assert(this_class_samples <= total_samples);
    return total_samples / static_cast<float_t>(classes * this_class_samples);
}

// create a target cost matrix implying equal cost for each *class* (distinct label)
//  - by default, each *sample* has an equal cost, which means e.g. that a classifier
//    may prefer to always guess the majority class (in case the degree of imbalance
//    is relatively high, and the classification task is relatively difficult)
//  - the parameter w can be used to fine-tune the balance:
//    * use 0 to have an equal cost for each *sample* (equal to not supplying any target costs at all)
//    * use 1 to have an equal cost for each *class* (default behaviour of this function)
//    * use a value between 0 and 1 to have something between the two extremes
inline std::vector<vec_t> create_balanced_target_cost(const std::vector<label_t>& t, float_t w = 1.0)
{
    const auto label_counts = calculate_label_counts(t);
    const serial_size_t total_sample_count = static_cast<serial_size_t>(t.size());
    const serial_size_t class_count = static_cast<serial_size_t>(label_counts.size());

    std::vector<vec_t> target_cost(t.size());

    for (serial_size_t i = 0; i < total_sample_count; ++i) {
        vec_t& sample_cost = target_cost[i];
        sample_cost.resize(class_count);
        const float_t balanced_weight = get_sample_weight_for_balanced_target_cost(class_count, total_sample_count, label_counts[t[i]]);
        const float_t unbalanced_weight = 1;
        const float_t sample_weight = w * balanced_weight + (1 - w) * unbalanced_weight;
        std::fill(sample_cost.begin(), sample_cost.end(), sample_weight);
    }

    return target_cost;
}

} // namespace tiny_dnn
