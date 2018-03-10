/***************************************************************************
* Copyright (c) 2017, Sylvain Corlay and Johan Mabille                     *
*                                                                          *
* Distributed under the terms of the BSD 3-Clause License.                 *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#ifndef XTL_XMETA_UTILS_HPP
#define XTL_XMETA_UTILS_HPP

#include <cstddef>
#include <type_traits>

#include "xfunctional.hpp"

namespace xtl
{
    namespace mpl
    {
        /*************
         * mpl types *
         *************/

        template <class... T>
        struct vector
        {
        };

        template <bool B>
        using bool_ = std::integral_constant<bool, B>;

        template <std::size_t S>
        using size_t_ = std::integral_constant<std::size_t, S>;

        /*******
         * if_ *
         *******/

        template <bool B, class T, class F>
        struct if_c : std::conditional<B, T, F>
        {
        };

        template <bool B, class T, class F>
        using if_c_t = typename if_c<B, T, F>::type;

        template <class B, class T, class F>
        struct if_ : if_c<B::value, T, F>
        {
        };

        template <class B, class T, class F>
        using if_t = typename if_<B, T, F>::type;

        /***********
         * eval_if *
         ***********/

        template <bool B, class T, class F>
        struct eval_if_c
        {
            using type = typename T::type;
        };

        template <class T, class F>
        struct eval_if_c<false, T, F>
        {
            using type = typename F::type;
        };

        template <class B, class T, class F>
        struct eval_if : eval_if_c<B::value, T, F>
        {
        };

        template <class B, class T, class F>
        using eval_if_t = typename eval_if<B, T, F>::type;

        /********
         * cast *
         ********/

        namespace detail
        {
            template <class A, template <class...> class B>
            struct cast_impl;

            template <template <class...> class A, class... T, template <class...> class B>
            struct cast_impl<A<T...>, B>
            {
                using type = B<T...>;
            };
        }

        template <class A, template <class...> class B>
        struct cast : detail::cast_impl<A, B>
        {
        };

        template <class A, template <class...> class B>
        using cast_t = typename cast<A, B>::type;

        /********
         * size *
         ********/

        namespace detail
        {
            template <class L>
            struct size_impl;

            template <template <class...> class F, class... T>
            struct size_impl<F<T...>> : size_t_<sizeof...(T)>
            {
            };
        }

        template <class L>
        struct size : detail::size_impl<L>
        {
        };

        /*********
         * empty *
         *********/

        namespace detail
        {
            template <class L>
            struct empty_impl;

            template <template <class...> class F, class... T>
            struct empty_impl<F<T...>> : bool_<sizeof...(T) == std::size_t(0)>
            {
            };
        }

        template <class L>
        struct empty : detail::empty_impl<L>
        {
        };

        template <class L>
        using empty_t = typename empty<L>::type;

        /********
         * plus *
         ********/

        namespace detail
        {
            template <class... T>
            struct plus_impl;

            template <>
            struct plus_impl<> : size_t_<0>
            {
            };

            template <class T1, class... T>
            struct plus_impl<T1, T...> : size_t_<T1::value + plus_impl<T...>::value>
            {
            };
        }

        template <class... T>
        struct plus : detail::plus_impl<T...>
        {
        };

        /*********
         * count *
         *********/

        namespace detail
        {
            template <class L, class V>
            struct count_impl;

            template <template <class...> class L, class... T, class V>
            struct count_impl<L<T...>, V> : plus<std::is_same<T, V>...>
            {
            };
        }

        template <class L, class V>
        struct count : detail::count_impl<L, V>
        {
        };

        /************
         * count_if *
         ************/

        namespace detail
        {
            template <class L, template <class> class P>
            struct count_if_impl;

            template <template <class...> class L, class... T, template <class> class P>
            struct count_if_impl<L<T...>, P> : plus<P<T>...>
            {
            };
        }

        template <class L, template <class> class P>
        struct count_if : detail::count_if_impl<L, P>
        {
        };

        /************
         * contains *
         ************/

        template <class L, class V>
        using contains = bool_<count<L, V>::value != 0>;

        /*********
         * front *
         *********/

        namespace detail
        {
            template <class L>
            struct front_impl;

            template <template <class...> class L, class T, class... U>
            struct front_impl<L<T, U...>>
            {
                using type = T;
            };
        }

        template <class L>
        struct front : detail::front_impl<L>
        {
        };

        template <class L>
        using front_t = typename front<L>::type;

        /********
         * back *
         ********/

        namespace detail
        {
            template <class L>
            struct back_impl;

            template <template <class...> class L, class T>
            struct back_impl<L<T>>
            {
                using type = T;
            };

            // Compilation time improvement
            template <template <class...> class L, class T1, class T2>
            struct back_impl<L<T1, T2>>
            {
                using type = T2;
            };

            template <template <class...> class L, class T1, class T2, class T3>
            struct back_impl<L<T1, T2, T3>>
            {
                using type = T3;
            };

            template <template <class...> class L, class T1, class T2, class T3, class T4>
            struct back_impl<L<T1, T2, T3, T4>>
            {
                using type = T4;
            };

            template <template <class...> class L, class T, class... U>
            struct back_impl<L<T, U...>> : back_impl<L<U...>>
            {
            };
        }

        template <class L>
        struct back : detail::back_impl<L>
        {
        };

        template <class L>
        using back_t = typename back<L>::type;

        /**************
         * push_front *
         **************/

        namespace detail
        {
            template <class L, class... T>
            struct push_front_impl;

            template <template <class...> class L, class... U, class... T>
            struct push_front_impl<L<U...>, T...>
            {
                using type = L<T..., U...>;
            };
        }

        template <class L, class... T>
        struct push_front : detail::push_front_impl<L, T...>
        {
        };

        template <class L, class... T>
        using push_front_t = typename push_front<L, T...>::type;

        /*************
         * push_back *
         *************/

        namespace detail
        {
            template <class L, class... T>
            struct push_back_impl;

            template <template <class...> class L, class... U, class... T>
            struct push_back_impl<L<U...>, T...>
            {
                using type = L<U..., T...>;
            };
        }

        template <class L, class... T>
        struct push_back : detail::push_back_impl<L, T...>
        {
        };

        template <class L, class... T>
        using push_back_t = typename push_back<L, T...>::type;

        /*************
         * pop_front *
         *************/

        namespace detail
        {
            template <class L>
            struct pop_front_impl;

            template <template <class...> class L, class T, class... U>
            struct pop_front_impl<L<T, U...>>
            {
                using type = L<U...>;
            };
        }

        template <class L>
        struct pop_front : detail::pop_front_impl<L>
        {
        };

        template <class L>
        using pop_front_t = typename pop_front<L>::type;

        /************
         * pop_back *
         ************/

        namespace detail
        {
            template <class L>
            struct pop_back_impl;

            template <template <class...> class L, class T>
            struct pop_back_impl<L<T>>
            {
                using type = L<>;
            };

            template <template <class...> class L, class T1, class T2>
            struct pop_back_impl<L<T1, T2>>
            {
                using head = T1;
                using type = L<head>;
            };

            template <template <class...> class L, class T, class... U>
            struct pop_back_impl<L<T, U...>>
            {
                using head = T;
                using type = L<head, typename pop_back_impl<L<U...>>::head>;
            };
        }

        template <class L>
        struct pop_back : detail::pop_back_impl<L>
        {
        };

        template <class L>
        using pop_back_t = typename pop_back<L>::type;

        /*************
         * transform *
         *************/

        namespace detail
        {
            template <template <class...> class F, class L>
            struct transform_impl;

            template <template <class...> class F, template <class...> class L, class... T>
            struct transform_impl<F, L<T...>>
            {
                using type = L<F<T>...>;
            };
        }

        template <template <class...> class F, class L>
        struct transform : detail::transform_impl<F, L>
        {
        };

        template <template <class...> class F, class L>
        using transform_t = typename transform<F, L>::type;

        /*************
         * merge_set *
         *************/

        namespace detail
        {
            template <class S1, class S2>
            struct merge_set_impl;

            template <template <class...> class L, class... T>
            struct merge_set_impl<L<T...>, L<>>
            {
                using type = L<T...>;
            };

            template <template <class...> class L, class... T, class U1, class... U>
            struct merge_set_impl<L<T...>, L<U1, U...>>
            {
                using type = typename merge_set_impl<if_t<contains<L<T...>, U1>,
                                                          L<T...>,
                                                          L<T..., U1>>,
                                                     L<U...>>::type;
            };
        }

        template <class S1, class S2>
        struct merge_set : detail::merge_set_impl<S1, S2>
        {
        };

        template <class S1, class S2>
        using merge_set_t = typename merge_set<S1, S2>::type;

        /*************
         * static_if *
         *************/

        template <class TF, class FF>
        auto static_if(std::true_type, const TF& tf, const FF&)
        {
            return tf(identity());
        }

        template <class TF, class FF>
        auto static_if(std::false_type, const TF&, const FF& ff)
        {
            return ff(identity());
        }

        template <bool cond, class TF, class FF>
        auto static_if(const TF& tf, const FF& ff)
        {
            return static_if(std::integral_constant<bool, cond>(), tf, ff);
        }
    }
}

#endif
