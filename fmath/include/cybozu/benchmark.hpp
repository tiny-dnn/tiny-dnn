#pragma once
/**
	@file
	@brief measure exec time of function
	@author MITSUNARI Shigeo
*/
#if defined(_MSC_VER) && (MSC_VER <= 1500)
	#include <cybozu/inttype.hpp>
#else
	#include <stdint.h>
#endif
#include <stdio.h>

#if defined(_M_IX86) || defined(_M_X64) || defined(__i386__) || defined(__x86_64__)
	#define CYBOZU_BENCH_USE_RDTSC
#endif
#ifdef CYBOZU_BENCH_USE_RDTSC
#ifdef _MSC_VER
	#include <intrin.h>
#endif
#else
	#include <cybozu/time.hpp>
#endif

namespace cybozu {

#ifdef CYBOZU_BENCH_USE_RDTSC
class CpuClock {
public:
	static inline uint64_t getRdtsc()
	{
#ifdef _MSC_VER
		return __rdtsc();
#else
		unsigned int eax, edx;
		__asm__ volatile("rdtsc" : "=a"(eax), "=d"(edx));
		return ((uint64_t)edx << 32) | eax;
#endif
	}
	CpuClock()
		: clock_(0)
		, count_(0)
	{
	}
	void begin()
	{
		clock_ -= getRdtsc();
	}
	void end()
	{
		clock_ += getRdtsc();
		count_++;
	}
	int getCount() const { return count_; }
	uint64_t getClock() const { return clock_; }
	void clear() { count_ = 0; clock_ = 0; }
	void put(const char *msg = 0, int N = 1) const
	{
		double t = getClock() / double(getCount()) / N;
		if (msg && *msg) printf("%s ", msg);
		if (t > 1e6) {
			printf("%7.3fMclk", t * 1e-6);
		} else if (t > 1e3) {
			printf("%7.3fKclk", t * 1e-3);
		} else {
			printf("%6.2f clk", t);
		}
		if (msg && *msg) printf("\n");
	}
	// adhoc constatns for CYBOZU_BENCH
	static const int loopN1 = 1000;
	static const int loopN2 = 1000000;
	static const uint64_t maxClk = (uint64_t)3e8;
private:
	uint64_t clock_;
	int count_;
};
#else
class CpuClock {
	cybozu::Time t_;
	uint64_t clock_;
	int count_;
public:
	CpuClock() : clock_(0), count_(0) { t_.setTime(0, 0); }
	void begin()
	{
		if (count_ == 0) t_.setCurrentTime(); // start
	}
	/*
		@note QQQ ; this is not same api as rdtsc version
	*/
	void end()
	{
		cybozu::Time cur(true);
		int diffSec = (int)(cur.getTime() - t_.getTime());
		int diffMsec = cur.getMsec() - t_.getMsec();
		const int diff = diffSec * 1000 + diffMsec;
		clock_ = diff;
		count_++;
	}
	int getCount() const { return count_; }
	uint64_t getClock() const { return clock_; }
	void clear() { t_.setTime(0, 0); clock_ = 0; count_ = 0; }
	void put(const char *msg = 0, int N = 1) const
	{
		double t = getClock() / double(getCount()) / N;
		if (msg && *msg) printf("%s ", msg);
		if (t > 1) {
			printf("%6.2fmsec", t);
		} else if (t > 1e-3) {
			printf("%6.2fusec", t * 1e3);
		} else {
			printf("%6.2fnsec", t * 1e6);
		}
		if (msg && *msg) printf("\n");
	}
	// adhoc constatns for CYBOZU_BENCH
	static const int loopN1 = 1000000;
	static const int loopN2 = 1000;
	static const uint64_t maxClk = (uint64_t)500;
};
#endif

namespace bench {

static CpuClock g_clk;
#ifdef __GNUC__
	#define CYBOZU_UNUSED __attribute__((unused))
#else
	#define CYBOZU_UNUSED
#endif
static int CYBOZU_UNUSED g_loopNum;

} // cybozu::bench
/*
	loop counter is automatically determined
	CYBOZU_BENCH(<msg>, <func>, <param1>, <param2>, ...);
	if msg == "" then only set g_clk, g_loopNum
*/
#define CYBOZU_BENCH(msg, func, ...) \
{ \
	const uint64_t maxClk = cybozu::CpuClock::maxClk; \
	cybozu::CpuClock clk; \
	for (int i = 0; i < cybozu::CpuClock::loopN2; i++) { \
		clk.begin(); \
		for (int j = 0; j < cybozu::CpuClock::loopN1; j++) { func(__VA_ARGS__); } \
		clk.end(); \
		if (clk.getClock() > maxClk) break; \
	} \
	if (msg && *msg) clk.put(msg, cybozu::CpuClock::loopN1); \
	cybozu::bench::g_clk = clk; cybozu::bench::g_loopNum = cybozu::CpuClock::loopN1; \
}

/*
	loop counter N is given
	CYBOZU_BENCH_C(<msg>, <counter>, <func>, <param1>, <param2>, ...);
	if msg == "" then only set g_clk, g_loopNum
*/
#define CYBOZU_BENCH_C(msg, _N, func, ...) \
{ \
	cybozu::CpuClock clk; \
	clk.begin(); \
	for (int j = 0; j < _N; j++) { func(__VA_ARGS__); } \
	clk.end(); \
	if (msg && *msg) clk.put(msg, _N); \
	cybozu::bench::g_clk = clk; cybozu::bench::g_loopNum = _N; \
}

} // cybozu
