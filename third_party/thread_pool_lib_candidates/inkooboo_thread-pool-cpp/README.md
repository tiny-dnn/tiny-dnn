thread-pool-cpp
=================

 * It is highly scalable and fast.
 * It is header only.
 * It implements both work-stealing and work-distribution balancing startegies.
 * It implements cooperative scheduling strategy for tasks.

Example run:
Post job to thread pool is much faster than for boost::asio based thread pool.

    Benchmark job reposting
    ***thread pool cpp***
    reposted 1000001 in 61.6754 ms
    reposted 1000001 in 62.0187 ms
    reposted 1000001 in 62.8785 ms
    reposted 1000001 in 70.2714 ms
    ***asio thread pool***
    reposted 1000001 in 1381.58 ms
    reposted 1000001 in 1390.35 ms
    reposted 1000001 in 1391.84 ms
    reposted 1000001 in 1393.19 ms

See benchmark/benchmark.cpp for benchmark code.

All code except [MPMCBoundedQueue](https://github.com/inkooboo/thread-pool-cpp/blob/master/thread_pool/mpsc_bounded_queue.hpp)
is under MIT license.

