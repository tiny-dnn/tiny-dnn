#include <thread_pool/worker.hpp>

using namespace tp;

size_t getWorkerIdForCurrentThread()
{
    return *detail::thread_id();
}

size_t getWorkerIdForCurrentThread2()
{
    return Worker::getWorkerIdForCurrentThread();
}
