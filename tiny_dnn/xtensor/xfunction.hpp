/***************************************************************************
* Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTENSOR_FUNCTION_HPP
#define XTENSOR_FUNCTION_HPP

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <utility>

#include "xtl/xsequence.hpp"
#include "xtl/xtype_traits.hpp"

#include "xexpression.hpp"
#include "xiterable.hpp"
#include "xlayout.hpp"
#include "xscalar.hpp"
#include "xstrides.hpp"
#include "xtensor_simd.hpp"
#include "xutils.hpp"

namespace xt
{

    namespace detail
    {

        /********************
         * common_size_type *
         ********************/

        template <class... Args>
        struct common_size_type
        {
            using type = std::common_type_t<typename Args::size_type...>;
        };

        template <>
        struct common_size_type<>
        {
            using type = std::size_t;
        };

        template <class... Args>
        using common_size_type_t = typename common_size_type<Args...>::type;

        template <bool... B>
        using conjunction_c = xtl::conjunction<std::integral_constant<bool, B>...>;

        /**************************
         * common_difference type *
         **************************/

        template <class... Args>
        struct common_difference_type
        {
            using type = std::common_type_t<typename Args::difference_type...>;
        };

        template <>
        struct common_difference_type<>
        {
            using type = std::ptrdiff_t;
        };

        template <class... Args>
        using common_difference_type_t = typename common_difference_type<Args...>::type;

        /*********************
         * common_value_type *
         *********************/

        template <class... Args>
        struct common_value_type
        {
            using type = promote_type_t<xvalue_type_t<Args>...>;
        };

        template <class... Args>
        using common_value_type_t = typename common_value_type<Args...>::type;

        template <class F, class R, class = void_t<>>
        struct simd_return_type
        {
        };

        template <class F, class R>
        struct simd_return_type<F, R, void_t<decltype(&F::simd_apply)>>
        {
            using type = R;
        };

        template <class F, class R>
        using simd_return_type_t = typename simd_return_type<F, R>::type;
    }

    template <class F, class R, class... CT>
    class xfunction_iterator;

    template <class F, class R, class... CT>
    class xfunction_stepper;

    template <class F, class R, class... CT>
    class xfunction_base;

    template <class F, class R, class... CT>
    struct xiterable_inner_types<xfunction_base<F, R, CT...>>
    {
        using inner_shape_type = promote_shape_t<typename std::decay_t<CT>::shape_type...>;
        using const_stepper = xfunction_stepper<F, R, CT...>;
        using stepper = const_stepper;
    };

    /******************
     * xfunction_base *
     ******************/

#define DL DEFAULT_LAYOUT

    /**
     * @class xfunction_base
     * @brief Base class for multidimensional function operating on
     * xexpression.
     *
     * The xfunction_base class implements a multidimensional function
     * operating on xexpression. Inheriting classes specify which
     * kind of xexpression the xfunction_base operates on.
     *
     * @tparam F the function type
     * @tparam R the return type of the function
     * @tparam CT the closure types for arguments of the function
     */
    template <class F, class R, class... CT>
    class xfunction_base : private xconst_iterable<xfunction_base<F, R, CT...>>
    {
    public:

        using self_type = xfunction_base<F, R, CT...>;
        using only_scalar = all_xscalar<CT...>;
        using functor_type = typename std::remove_reference<F>::type;

        using value_type = R;
        using reference = value_type;
        using const_reference = value_type;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using size_type = detail::common_size_type_t<std::decay_t<CT>...>;
        using difference_type = detail::common_difference_type_t<std::decay_t<CT>...>;
        using simd_value_type = xsimd::simd_type<value_type>;
        using iterable_base = xconst_iterable<xfunction_base<F, R, CT...>>;
        using inner_shape_type = typename iterable_base::inner_shape_type;
        using shape_type = inner_shape_type;

        using stepper = typename iterable_base::stepper;
        using const_stepper = typename iterable_base::const_stepper;

        static constexpr layout_type static_layout = compute_layout(std::decay_t<CT>::static_layout...);
        static constexpr bool contiguous_layout = detail::conjunction_c<std::decay_t<CT>::contiguous_layout...>::value;

        template <layout_type L>
        using layout_iterator = typename iterable_base::template layout_iterator<L>;
        template <layout_type L>
        using const_layout_iterator = typename iterable_base::template const_layout_iterator<L>;
        template <layout_type L>
        using reverse_layout_iterator = typename iterable_base::template reverse_layout_iterator<L>;
        template <layout_type L>
        using const_reverse_layout_iterator = typename iterable_base::template const_reverse_layout_iterator<L>;

        template <class S, layout_type L>
        using broadcast_iterator = typename iterable_base::template broadcast_iterator<S, L>;
        template <class S, layout_type L>
        using const_broadcast_iterator = typename iterable_base::template const_broadcast_iterator<S, L>;
        template <class S, layout_type L>
        using reverse_broadcast_iterator = typename iterable_base::template reverse_broadcast_iterator<S, L>;
        template <class S, layout_type L>
        using const_reverse_broadcast_iterator = typename iterable_base::template const_reverse_broadcast_iterator<S, L>;

        using const_storage_iterator = xfunction_iterator<F, R, CT...>;
        using storage_iterator = const_storage_iterator;
        using const_reverse_storage_iterator = std::reverse_iterator<const_storage_iterator>;
        using reverse_storage_iterator = std::reverse_iterator<storage_iterator>;

        using iterator = typename iterable_base::iterator;
        using const_iterator = typename iterable_base::const_iterator;
        using reverse_iterator = typename iterable_base::reverse_iterator;
        using const_reverse_iterator = typename iterable_base::const_reverse_iterator;

        size_type size() const noexcept;
        size_type dimension() const noexcept;
        const shape_type& shape() const;
        layout_type layout() const noexcept;

        template <class... Args>
        const_reference operator()(Args... args) const;

        template <class... Args>
        const_reference at(Args... args) const;

        template <class S>
        disable_integral_t<S, const_reference> operator[](const S& index) const;
        template <class I>
        const_reference operator[](std::initializer_list<I> index) const;
        const_reference operator[](size_type i) const;

        template <class It>
        const_reference element(It first, It last) const;

        template <class S>
        bool broadcast_shape(S& shape, bool reuse_cache = false) const;

        template <class S>
        bool is_trivial_broadcast(const S& strides) const noexcept;

        using iterable_base::begin;
        using iterable_base::end;
        using iterable_base::cbegin;
        using iterable_base::cend;
        using iterable_base::rbegin;
        using iterable_base::rend;
        using iterable_base::crbegin;
        using iterable_base::crend;

        template <layout_type L = DL>
        const_storage_iterator storage_begin() const noexcept;
        template <layout_type L = DL>
        const_storage_iterator storage_end() const noexcept;
        template <layout_type L = DL>
        const_storage_iterator storage_cbegin() const noexcept;
        template <layout_type L = DL>
        const_storage_iterator storage_cend() const noexcept;

        template <layout_type L = DL>
        const_reverse_storage_iterator storage_rbegin() const noexcept;
        template <layout_type L = DL>
        const_reverse_storage_iterator storage_rend() const noexcept;
        template <layout_type L = DL>
        const_reverse_storage_iterator storage_crbegin() const noexcept;
        template <layout_type L = DL>
        const_reverse_storage_iterator storage_crend() const noexcept;

        template <class S>
        const_stepper stepper_begin(const S& shape) const noexcept;
        template <class S>
        const_stepper stepper_end(const S& shape, layout_type l) const noexcept;

        const_reference data_element(size_type i) const;

        template <class UT = self_type, class = typename std::enable_if<UT::only_scalar::value>::type>
        operator value_type() const;

        template <class align, class simd = simd_value_type>
        detail::simd_return_type_t<functor_type, simd> load_simd(size_type i) const;

        const std::tuple<CT...>& arguments() const noexcept;

    protected:

        template <class Func, class U = std::enable_if<!std::is_base_of<Func, self_type>::value>>
        xfunction_base(Func&& f, CT... e) noexcept;

        ~xfunction_base() = default;

        xfunction_base(const xfunction_base&) = default;
        xfunction_base& operator=(const xfunction_base&) = default;

        xfunction_base(xfunction_base&&) = default;
        xfunction_base& operator=(xfunction_base&&) = default;

    private:

        template <std::size_t... I>
        layout_type layout_impl(std::index_sequence<I...>) const noexcept;

        template <std::size_t... I, class... Args>
        const_reference access_impl(std::index_sequence<I...>, Args... args) const;

        template <std::size_t... I, class It>
        const_reference element_access_impl(std::index_sequence<I...>, It first, It last) const;

        template <std::size_t... I>
        const_reference data_element_impl(std::index_sequence<I...>, size_type i) const;

        template <class align, class simd, std::size_t... I>
        simd load_simd_impl(std::index_sequence<I...>, size_type i) const;

        template <class Func, std::size_t... I>
        const_stepper build_stepper(Func&& f, std::index_sequence<I...>) const noexcept;

        template <class Func, std::size_t... I>
        const_storage_iterator build_iterator(Func&& f, std::index_sequence<I...>) const noexcept;

        size_type compute_dimension() const noexcept;

        std::tuple<CT...> m_e;
        functor_type m_f;
        mutable shape_type m_shape;
        mutable bool m_shape_trivial;
        mutable bool m_shape_computed;

        friend class xfunction_iterator<F, R, CT...>;
        friend class xfunction_stepper<F, R, CT...>;
        friend class xconst_iterable<self_type>;
    };

#undef DL

    /**********************
     * xfunction_iterator *
     **********************/

    template <class CT>
    class xscalar;

    namespace detail
    {
        template <class C>
        struct get_iterator_impl
        {
            using type = typename C::storage_iterator;
        };

        template <class C>
        struct get_iterator_impl<const C>
        {
            using type = typename C::const_storage_iterator;
        };

        template <class CT>
        struct get_iterator_impl<xscalar<CT>>
        {
            using type = typename xscalar<CT>::dummy_iterator;
        };

        template <class CT>
        struct get_iterator_impl<const xscalar<CT>>
        {
            using type = typename xscalar<CT>::const_dummy_iterator;
        };
    }

    template <class C>
    using get_iterator = typename detail::get_iterator_impl<C>::type;

    template <class F, class R, class... CT>
    class xfunction_iterator : public xtl::xrandom_access_iterator_base<xfunction_iterator<F, R, CT...>,
                                                                        typename xfunction_base<F, R, CT...>::value_type,
                                                                        typename xfunction_base<F, R, CT...>::difference_type,
                                                                        typename xfunction_base<F, R, CT...>::pointer,
                                                                        typename xfunction_base<F, R, CT...>::reference>
    {
    public:

        using self_type = xfunction_iterator<F, R, CT...>;
        using functor_type = typename std::remove_reference<F>::type;
        using xfunction_type = xfunction_base<F, R, CT...>;

        using value_type = typename xfunction_type::value_type;
        using reference = typename xfunction_type::value_type;
        using pointer = typename xfunction_type::const_pointer;
        using difference_type = typename xfunction_type::difference_type;
        using iterator_category = std::random_access_iterator_tag;

        template <class... It>
        xfunction_iterator(const xfunction_type* func, It&&... it) noexcept;

        self_type& operator++();
        self_type& operator--();

        self_type& operator+=(difference_type n);
        self_type& operator-=(difference_type n);

        difference_type operator-(const self_type& rhs) const;

        reference operator*() const;

        bool equal(const self_type& rhs) const;
        bool less_than(const self_type& rhs) const;

    private:

        using data_type = std::tuple<get_iterator<const std::decay_t<CT>>...>;

        template <std::size_t... I>
        reference deref_impl(std::index_sequence<I...>) const;

        template <std::size_t... I>
        difference_type tuple_max_diff(std::index_sequence<I...>,
                                       const data_type& lhs,
                                       const data_type& rhs) const;

        const xfunction_type* p_f;
        data_type m_it;
    };

    template <class F, class R, class... CT>
    bool operator==(const xfunction_iterator<F, R, CT...>& it1,
                    const xfunction_iterator<F, R, CT...>& it2);

    template <class F, class R, class... CT>
    bool operator<(const xfunction_iterator<F, R, CT...>& it1,
                   const xfunction_iterator<F, R, CT...>& it2);

    /*********************
     * xfunction_stepper *
     *********************/

    template <class F, class R, class... CT>
    class xfunction_stepper
    {
    public:

        using self_type = xfunction_stepper<F, R, CT...>;
        using functor_type = typename std::remove_reference<F>::type;
        using xfunction_type = xfunction_base<F, R, CT...>;

        using value_type = typename xfunction_type::value_type;
        using reference = typename xfunction_type::value_type;
        using pointer = typename xfunction_type::const_pointer;
        using size_type = typename xfunction_type::size_type;
        using difference_type = typename xfunction_type::difference_type;

        using shape_type = typename xfunction_type::shape_type;

        template <class... It>
        xfunction_stepper(const xfunction_type* func, It&&... it) noexcept;

        void step(size_type dim, size_type n = 1);
        void step_back(size_type dim, size_type n = 1);
        void reset(size_type dim);
        void reset_back(size_type dim);

        void to_begin();
        void to_end(layout_type l);

        reference operator*() const;

        bool equal(const self_type& rhs) const;

    private:

        template <std::size_t... I>
        reference deref_impl(std::index_sequence<I...>) const;

        const xfunction_type* p_f;
        std::tuple<typename std::decay_t<CT>::const_stepper...> m_it;
    };

    template <class F, class R, class... CT>
    bool operator==(const xfunction_stepper<F, R, CT...>& it1,
                    const xfunction_stepper<F, R, CT...>& it2);

    template <class F, class R, class... CT>
    bool operator!=(const xfunction_stepper<F, R, CT...>& it1,
                    const xfunction_stepper<F, R, CT...>& it2);

    /*************
     * xfunction *
     *************/

    /**
     * @class xfunction
     * @brief Multidimensional function operating on
     * xtensor expressions.
     *
     * The xfunction class implements a multidimensional function
     * operating on xtensor expressions.
     *
     * @tparam F the function type
     * @tparam R the return type of the function
     * @tparam CT the closure types for arguments of the function
     */
    template <class F, class R, class... CT>
    class xfunction : public xfunction_base<F, R, CT...>,
                      public xexpression<xfunction<F, R, CT...>>
    {
    public:

        using self_type = xfunction<F, R, CT...>;
        using base_type = xfunction_base<F, R, CT...>;

        template <class Func, class U = std::enable_if<!std::is_base_of<Func, self_type>::value>>
        xfunction(Func&& f, CT... e) noexcept;

        ~xfunction() = default;

        xfunction(const xfunction&) = default;
        xfunction& operator=(const xfunction&) = default;

        xfunction(xfunction&&) = default;
        xfunction& operator=(xfunction&&) = default;
    };

    /*********************************
     * xfunction_base implementation *
     *********************************/

    /**
     * @name Constructor
     */
    //@{
    /**
     * Constructs an xfunction_base applying the specified function to the given
     * arguments.
     * @param f the function to apply
     * @param e the \ref xexpression arguments
     */
    template <class F, class R, class... CT>
    template <class Func, class U>
    inline xfunction_base<F, R, CT...>::xfunction_base(Func&& f, CT... e) noexcept
        : m_e(e...), m_f(std::forward<Func>(f)), m_shape(xtl::make_sequence<shape_type>(0, size_type(1))),
          m_shape_computed(false)
    {
    }
    //@}

    /**
     * @name Size and shape
     */
    //@{
    /**
     * Returns the size of the expression.
     */
    template <class F, class R, class... CT>
    inline auto xfunction_base<F, R, CT...>::size() const noexcept -> size_type
    {
        return compute_size(shape());
    }

    /**
     * Returns the number of dimensions of the function.
     */
    template <class F, class R, class... CT>
    inline auto xfunction_base<F, R, CT...>::dimension() const noexcept -> size_type
    {
        size_type dimension = m_shape_computed ? m_shape.size() : compute_dimension();
        return dimension;
    }

    /**
     * Returns the shape of the xfunction.
     */
    template <class F, class R, class... CT>
    inline auto xfunction_base<F, R, CT...>::shape() const -> const shape_type&
    {
        if (!m_shape_computed)
        {
            m_shape = xtl::make_sequence<shape_type>(compute_dimension(), size_type(1));
            m_shape_trivial = broadcast_shape(m_shape, false);
            m_shape_computed = true;
        }
        return m_shape;
    }

    /**
     * Returns the layout_type of the xfunction.
     */
    template <class F, class R, class... CT>
    inline layout_type xfunction_base<F, R, CT...>::layout() const noexcept
    {
        return layout_impl(std::make_index_sequence<sizeof...(CT)>());
    }
    //@}

    /**
     * @name Data
     */
    /**
     * Returns a constant reference to the element at the specified position in the function.
     * @param args a list of indices specifying the position in the function. Indices
     * must be unsigned integers, the number of indices should be equal or greater than
     * the number of dimensions of the function.
     */
    template <class F, class R, class... CT>
    template <class... Args>
    inline auto xfunction_base<F, R, CT...>::operator()(Args... args) const -> const_reference
    {
        // The static cast prevents the compiler from instantiating the template methods with signed integers,
        // leading to warning about signed/unsigned conversions in the deeper layers of the access methods
        return access_impl(std::make_index_sequence<sizeof...(CT)>(), static_cast<size_type>(args)...);
    }

    /**
     * Returns a constant reference to the element at the specified position in the expression,
     * after dimension and bounds checking.
     * @param args a list of indices specifying the position in the function. Indices
     * must be unsigned integers, the number of indices should be equal to the number of dimensions
     * of the expression.
     * @exception std::out_of_range if the number of argument is greater than the number of dimensions
     * or if indices are out of bounds.
     */
    template <class F, class R, class... CT>
    template <class... Args>
    inline auto xfunction_base<F, R, CT...>::at(Args... args) const -> const_reference
    {
        check_access(shape(), static_cast<size_type>(args)...);
        return this->operator()(args...);
    }

    template <class F, class R, class... CT>
    template <class S>
    inline auto xfunction_base<F, R, CT...>::operator[](const S& index) const
        -> disable_integral_t<S, const_reference>
    {
        return element(index.cbegin(), index.cend());
    }

    template <class F, class R, class... CT>
    template <class I>
    inline auto xfunction_base<F, R, CT...>::operator[](std::initializer_list<I> index) const -> const_reference
    {
        return element(index.begin(), index.end());
    }

    template <class F, class R, class... CT>
    inline auto xfunction_base<F, R, CT...>::operator[](size_type i) const -> const_reference
    {
        return operator()(i);
    }

    /**
     * Returns a constant reference to the element at the specified position in the function.
     * @param first iterator starting the sequence of indices
     * @param last iterator ending the sequence of indices
     * The number of indices in the sequence should be equal to or greater
     * than the number of dimensions of the container.
     */
    template <class F, class R, class... CT>
    template <class It>
    inline auto xfunction_base<F, R, CT...>::element(It first, It last) const -> const_reference
    {
        return element_access_impl(std::make_index_sequence<sizeof...(CT)>(), first, last);
    }
    //@}

    /**
     * @name Broadcasting
     */
    //@{
    /**
     * Broadcast the shape of the function to the specified parameter.
     * @param shape the result shape
     * @param reuse_cache boolean for reusing a previously computed shape
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class F, class R, class... CT>
    template <class S>
    inline bool xfunction_base<F, R, CT...>::broadcast_shape(S& shape, bool reuse_cache) const
    {
        if (reuse_cache && m_shape_computed)
        {
            std::copy(m_shape.cbegin(), m_shape.cend(), shape.begin());
            return m_shape_trivial;
        }
        else
        {
            // e.broadcast_shape must be evaluated even if b is false
            auto func = [&shape](bool b, auto&& e) { return e.broadcast_shape(shape) && b; };
            return accumulate(func, true, m_e);
        }
    }

    /**
     * Compares the specified strides with those of the container to see whether
     * the broadcasting is trivial.
     * @return a boolean indicating whether the broadcasting is trivial
     */
    template <class F, class R, class... CT>
    template <class S>
    inline bool xfunction_base<F, R, CT...>::is_trivial_broadcast(const S& strides) const noexcept
    {
        auto func = [&strides](bool b, auto&& e) { return b && e.is_trivial_broadcast(strides); };
        return accumulate(func, true, m_e);
    }
    //@}

    template <class F, class R, class... CT>
    template <layout_type L>
    inline auto xfunction_base<F, R, CT...>::storage_begin() const noexcept -> const_storage_iterator
    {
        return storage_cbegin<L>();
    }

    template <class F, class R, class... CT>
    template <layout_type L>
    inline auto xfunction_base<F, R, CT...>::storage_end() const noexcept -> const_storage_iterator
    {
        return storage_cend<L>();
    }

    template <class F, class R, class... CT>
    template <layout_type L>
    inline auto xfunction_base<F, R, CT...>::storage_cbegin() const noexcept -> const_storage_iterator
    {
        auto f = [](const auto& e) noexcept { return detail::trivial_begin(e); };
        return build_iterator(f, std::make_index_sequence<sizeof...(CT)>());
    }

    template <class F, class R, class... CT>
    template <layout_type L>
    inline auto xfunction_base<F, R, CT...>::storage_cend() const noexcept -> const_storage_iterator
    {
        auto f = [](const auto& e) noexcept { return detail::trivial_end(e); };
        return build_iterator(f, std::make_index_sequence<sizeof...(CT)>());
    }

    template <class F, class R, class... CT>
    template <layout_type L>
    inline auto xfunction_base<F, R, CT...>::storage_rbegin() const noexcept -> const_reverse_storage_iterator
    {
        return storage_crbegin<L>();
    }

    template <class F, class R, class... CT>
    template <layout_type L>
    inline auto xfunction_base<F, R, CT...>::storage_rend() const noexcept -> const_reverse_storage_iterator
    {
        return storage_crend<L>();
    }

    template <class F, class R, class... CT>
    template <layout_type L>
    inline auto xfunction_base<F, R, CT...>::storage_crbegin() const noexcept -> const_reverse_storage_iterator
    {
        return const_reverse_storage_iterator(storage_cend<L>());
    }

    template <class F, class R, class... CT>
    template <layout_type L>
    inline auto xfunction_base<F, R, CT...>::storage_crend() const noexcept -> const_reverse_storage_iterator
    {
        return const_reverse_storage_iterator(storage_cbegin<L>());
    }

    template <class F, class R, class... CT>
    template <class S>
    inline auto xfunction_base<F, R, CT...>::stepper_begin(const S& shape) const noexcept -> const_stepper
    {
        auto f = [&shape](const auto& e) noexcept { return e.stepper_begin(shape); };
        return build_stepper(f, std::make_index_sequence<sizeof...(CT)>());
    }

    template <class F, class R, class... CT>
    template <class S>
    inline auto xfunction_base<F, R, CT...>::stepper_end(const S& shape, layout_type l) const noexcept -> const_stepper
    {
        auto f = [&shape, l](const auto& e) noexcept { return e.stepper_end(shape, l); };
        return build_stepper(f, std::make_index_sequence<sizeof...(CT)>());
    }

    template <class F, class R, class... CT>
    inline auto xfunction_base<F, R, CT...>::data_element(size_type i) const -> const_reference
    {
        return data_element_impl(std::make_index_sequence<sizeof...(CT)>(), i);
    }

    template <class F, class R, class... CT>
    template <class UT, class>
    inline xfunction_base<F, R, CT...>::operator value_type() const
    {
        return operator()();
    }

    template <class F, class R, class... CT>
    template <class align, class simd>
    inline auto xfunction_base<F, R, CT...>::load_simd(size_type i) const -> detail::simd_return_type_t<functor_type, simd>
    {
        return load_simd_impl<align, simd>(std::make_index_sequence<sizeof...(CT)>(), i);
    }

    template <class F, class R, class... CT>
    inline auto xfunction_base<F, R, CT...>::arguments() const noexcept -> const std::tuple<CT...>&
    {
        return m_e;
    }

    template <class F, class R, class... CT>
    template <std::size_t... I>
    inline layout_type xfunction_base<F, R, CT...>::layout_impl(std::index_sequence<I...>) const noexcept
    {
        return compute_layout(std::get<I>(m_e).layout()...);
    }

    template <class F, class R, class... CT>
    template <std::size_t... I, class... Args>
    inline auto xfunction_base<F, R, CT...>::access_impl(std::index_sequence<I...>, Args... args) const -> const_reference
    {
        return m_f(detail::get_element(std::get<I>(m_e), args...)...);
    }

    template <class F, class R, class... CT>
    template <std::size_t... I, class It>
    inline auto xfunction_base<F, R, CT...>::element_access_impl(std::index_sequence<I...>, It first, It last) const -> const_reference
    {
        return m_f((std::get<I>(m_e).element(first, last))...);
    }

    template <class F, class R, class... CT>
    template <std::size_t... I>
    inline auto xfunction_base<F, R, CT...>::data_element_impl(std::index_sequence<I...>, size_type i) const -> const_reference
    {
        return m_f((std::get<I>(m_e).data_element(i))...);
    }

    template <class F, class R, class... CT>
    template <class align, class simd, std::size_t... I>
    inline auto xfunction_base<F, R, CT...>::load_simd_impl(std::index_sequence<I...>, size_type i) const -> simd
    {
        return m_f.simd_apply((std::get<I>(m_e).template load_simd<align, simd>(i))...);
    }

    template <class F, class R, class... CT>
    template <class Func, std::size_t... I>
    inline auto xfunction_base<F, R, CT...>::build_stepper(Func&& f, std::index_sequence<I...>) const noexcept -> const_stepper
    {
        return const_stepper(this, f(std::get<I>(m_e))...);
    }

    template <class F, class R, class... CT>
    template <class Func, std::size_t... I>
    inline auto xfunction_base<F, R, CT...>::build_iterator(Func&& f, std::index_sequence<I...>) const noexcept -> const_storage_iterator
    {
        return const_storage_iterator(this, f(std::get<I>(m_e))...);
    }

    template <class F, class R, class... CT>
    inline auto xfunction_base<F, R, CT...>::compute_dimension() const noexcept -> size_type
    {
        auto func = [](size_type d, auto&& e) noexcept { return std::max(d, e.dimension()); };
        return accumulate(func, size_type(0), m_e);
    }

    /*************************************
     * xfunction_iterator implementation *
     *************************************/

    template <class F, class R, class... CT>
    template <class... It>
    inline xfunction_iterator<F, R, CT...>::xfunction_iterator(const xfunction_type* func, It&&... it) noexcept
        : p_f(func), m_it(std::forward<It>(it)...)
    {
    }

    template <class F, class R, class... CT>
    inline auto xfunction_iterator<F, R, CT...>::operator++() -> self_type&
    {
        auto f = [](auto& it) { ++it; };
        for_each(f, m_it);
        return *this;
    }

    template <class F, class R, class... CT>
    inline auto xfunction_iterator<F, R, CT...>::operator--() -> self_type&
    {
        auto f = [](auto& it) { return --it; };
        for_each(f, m_it);
        return *this;
    }

    template <class F, class R, class... CT>
    inline auto xfunction_iterator<F, R, CT...>::operator+=(difference_type n) -> self_type&
    {
        auto f = [n](auto& it) { it += n; };
        for_each(f, m_it);
        return *this;
    }

    template <class F, class R, class... CT>
    inline auto xfunction_iterator<F, R, CT...>::operator-=(difference_type n) -> self_type&
    {
        auto f = [n](auto& it) { it -= n; };
        for_each(f, m_it);
        return *this;
    }

    template <class F, class R, class... CT>
    inline auto xfunction_iterator<F, R, CT...>::operator-(const self_type& rhs) const -> difference_type
    {
        return tuple_max_diff(std::make_index_sequence<sizeof...(CT)>(), m_it, rhs.m_it);
    }

    template <class F, class R, class... CT>
    inline auto xfunction_iterator<F, R, CT...>::operator*() const -> reference
    {
        return deref_impl(std::make_index_sequence<sizeof...(CT)>());
    }

    template <class F, class R, class... CT>
    inline bool xfunction_iterator<F, R, CT...>::equal(const self_type& rhs) const
    {
        return p_f == rhs.p_f && m_it == rhs.m_it;
    }

    template <class F, class R, class... CT>
    inline bool xfunction_iterator<F, R, CT...>::less_than(const self_type& rhs) const
    {
        return p_f == rhs.p_f && m_it < rhs.m_it;
    }

    template <class F, class R, class... CT>
    template <std::size_t... I>
    inline auto xfunction_iterator<F, R, CT...>::deref_impl(std::index_sequence<I...>) const -> reference
    {
        return (p_f->m_f)(*std::get<I>(m_it)...);
    }

    template <class F, class R, class... CT>
    template <std::size_t... I>
    inline auto xfunction_iterator<F, R, CT...>::tuple_max_diff(std::index_sequence<I...>,
                                                                const data_type& lhs,
                                                                const data_type& rhs) const -> difference_type
    {
        auto diff = std::make_tuple((std::get<I>(lhs) - std::get<I>(rhs))...);
        auto func = [](difference_type n, auto&& v) { return std::max(n, v); };
        return accumulate(func, difference_type(0), diff);
    }

    template <class F, class R, class... CT>
    inline bool operator==(const xfunction_iterator<F, R, CT...>& it1,
                           const xfunction_iterator<F, R, CT...>& it2)
    {
        return it1.equal(it2);
    }

    template <class F, class R, class... CT>
    inline bool operator<(const xfunction_iterator<F, R, CT...>& it1,
                          const xfunction_iterator<F, R, CT...>& it2)
    {
        return it1.less_than(it2);
    }

    /************************************
     * xfunction_stepper implementation *
     ************************************/

    template <class F, class R, class... CT>
    template <class... It>
    inline xfunction_stepper<F, R, CT...>::xfunction_stepper(const xfunction_type* func, It&&... it) noexcept
        : p_f(func), m_it(std::forward<It>(it)...)
    {
    }

    template <class F, class R, class... CT>
    inline void xfunction_stepper<F, R, CT...>::step(size_type dim, size_type n)
    {
        auto f = [dim, n](auto& it) { it.step(dim, n); };
        for_each(f, m_it);
    }

    template <class F, class R, class... CT>
    inline void xfunction_stepper<F, R, CT...>::step_back(size_type dim, size_type n)
    {
        auto f = [dim, n](auto& it) { it.step_back(dim, n); };
        for_each(f, m_it);
    }

    template <class F, class R, class... CT>
    inline void xfunction_stepper<F, R, CT...>::reset(size_type dim)
    {
        auto f = [dim](auto& it) { it.reset(dim); };
        for_each(f, m_it);
    }

    template <class F, class R, class... CT>
    inline void xfunction_stepper<F, R, CT...>::reset_back(size_type dim)
    {
        auto f = [dim](auto& it) { it.reset_back(dim); };
        for_each(f, m_it);
    }

    template <class F, class R, class... CT>
    inline void xfunction_stepper<F, R, CT...>::to_begin()
    {
        auto f = [](auto& it) { it.to_begin(); };
        for_each(f, m_it);
    }

    template <class F, class R, class... CT>
    inline void xfunction_stepper<F, R, CT...>::to_end(layout_type l)
    {
        auto f = [l](auto& it) { it.to_end(l); };
        for_each(f, m_it);
    }

    template <class F, class R, class... CT>
    inline auto xfunction_stepper<F, R, CT...>::operator*() const -> reference
    {
        return deref_impl(std::make_index_sequence<sizeof...(CT)>());
    }

    template <class F, class R, class... CT>
    inline bool xfunction_stepper<F, R, CT...>::equal(const self_type& rhs) const
    {
        return p_f == rhs.p_f && m_it == rhs.m_it;
    }

    template <class F, class R, class... CT>
    template <std::size_t... I>
    inline auto xfunction_stepper<F, R, CT...>::deref_impl(std::index_sequence<I...>) const -> reference
    {
        return (p_f->m_f)(*std::get<I>(m_it)...);
    }

    template <class F, class R, class... CT>
    inline bool operator==(const xfunction_stepper<F, R, CT...>& it1,
                           const xfunction_stepper<F, R, CT...>& it2)
    {
        return it1.equal(it2);
    }

    template <class F, class R, class... CT>
    inline bool operator!=(const xfunction_stepper<F, R, CT...>& it1,
                           const xfunction_stepper<F, R, CT...>& it2)
    {
        return !(it1.equal(it2));
    }

    /****************************
     * xfunction implementation *
     ****************************/

     /**
      * Constructs an xfunction applying the specified function to the given
      * arguments.
      * @param f the function to apply
      * @param e the \ref xexpression arguments
      */
    template <class F, class R, class... CT>
    template <class Func, class U>
    xfunction<F, R, CT...>::xfunction(Func&& f, CT... e) noexcept
        : base_type(std::forward<Func>(f), e...)
    {
    }
}

#endif
