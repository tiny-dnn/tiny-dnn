/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XCSV_HPP
#define XCSV_HPP

#include <exception>
#include <istream>
#include <iterator>
#include <sstream>
#include <string>
#include <utility>

#include "xtensor.hpp"

namespace xt
{

    /**************************************
     * load_csv and dump_csv declarations *
     **************************************/

    template <class T, class A = std::allocator<T>>
    xtensor_container<std::vector<T, A>, 2> load_csv(std::istream& stream);

    template <class E>
    void dump_csv(std::ostream& stream, const xexpression<E>& e);

    /*****************************************
     * load_csv and dump_csv implementations *
     *****************************************/

    namespace detail
    {
        template <class T>
        inline T lexical_cast(const std::string& cell)
        {
            T res;
            std::istringstream iss(cell);
            iss >> res;
            return res;
        }

        template <>
        inline float lexical_cast<float>(const std::string& cell) { return std::stof(cell); }

        template <>
        inline double lexical_cast<double>(const std::string& cell) { return std::stod(cell); }

        template <>
        inline long double lexical_cast<long double>(const std::string& cell) { return std::stold(cell); }

        template <>
        inline int lexical_cast<int>(const std::string& cell) { return std::stoi(cell); }

        template <>
        inline long lexical_cast<long>(const std::string& cell) { return std::stol(cell); }

        template <>
        inline long long lexical_cast<long long>(const std::string& cell) { return std::stoll(cell); }

        template <>
        inline unsigned int lexical_cast<unsigned int>(const std::string& cell) { return static_cast<unsigned int>(std::stoul(cell)); }

        template <>
        inline unsigned long lexical_cast<unsigned long>(const std::string& cell) { return std::stoul(cell); }

        template <>
        inline unsigned long long lexical_cast<unsigned long long>(const std::string& cell) { return std::stoull(cell); }

        template <class ST, class T, class OI>
        ST load_csv_row(std::istream& row_stream, OI output, std::string cell)
        {
            ST length = 0;
            while (std::getline(row_stream, cell, ','))
            {
                *output++ = lexical_cast<T>(cell);
                ++length;
            }
            return length;
        }
    }

    /**
     * @brief Load tensor from CSV.
     * 
     * Returns an \ref xexpression for the parsed CSV
     * @param stream the input stream containing the CSV encoded values
     */
    template <class T, class A>
    xtensor_container<std::vector<T, A>, 2> load_csv(std::istream& stream)
    {
        using container_type = typename std::vector<T, A>;
        using tensor_type = xtensor_container<container_type, 2>;
        using size_type = typename tensor_type::size_type;
        using inner_shape_type = typename tensor_type::inner_shape_type;
        using inner_strides_type = typename tensor_type::inner_strides_type;
        using output_iterator = std::back_insert_iterator<container_type>;

        container_type data;
        size_type nbrow = 0, nbcol = 0;
        {
            output_iterator output(data);
            std::string row, cell;
            while (std::getline(stream, row))
            {
                std::stringstream row_stream(row);
                nbcol = detail::load_csv_row<size_type, T, output_iterator>(row_stream, output, cell);
                ++nbrow;
            }
        }
        inner_shape_type shape = {nbrow, nbcol};
        inner_strides_type strides;  // no need for initializer list for stack-allocated strides_type
        size_type data_size = compute_strides(shape, layout_type::row_major, strides);
        // Sanity check for data size.
        if (data.size() != data_size)
        {
            throw std::runtime_error("Inconsistent row lengths in CSV");
        }
        return tensor_type(std::move(data), std::move(shape), std::move(strides));
    }

    /**
     * @brief Dump tensor to CSV.
     * 
     * @param stream the output stream to write the CSV encoded values
     * @param e the tensor expression to serialize
     */
    template <class E>
    void dump_csv(std::ostream& stream, const xexpression<E>& e)
    {
        using size_type = typename E::size_type;
        const E& ex = e.derived_cast();
        if (ex.dimension() != 2)
        {
            throw std::runtime_error("Only 2-D expressions can be serialized to CSV");
        }
        size_type nbrows = ex.shape()[0], nbcols = ex.shape()[1];
        auto st = ex.stepper_begin(ex.shape());
        for (size_type r = 0; r != nbrows; ++r)
        {
            for (size_type c = 0; c != nbcols; ++c)
            {
                stream << *st;
                if (c != nbcols - 1)
                {
                    st.step(1);
                    stream << ',';
                }
                else
                {
                    st.reset(1);
                    st.step(0);
                    stream << std::endl;
                }
            }
        }
    }
}

#endif
