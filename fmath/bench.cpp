/*
    cl /Ox /Ob2 /arch:SSE2 /fp:fast bench.cpp -I../xbyak /EHsc /DNOMINMAX
    g++ -O3 -fomit-frame-pointer -fno-operator-names -mssse3 -mfpmath=sse -ffast-math
*/
//#define FMATH_USE_XBYAK

#include <stdio.h>
#include <string.h>
#include "fmath.hpp"
#include <cmath>
#include <algorithm>

#define XBYAK_NO_OP_NAMES
#include "xbyak/xbyak_util.h"

inline void put(const void *p)
{
	const float *f = (const float*)p;
	printf("{%e, %e, %e, %e}\n", f[0], f[1], f[2], f[3]);
}
inline void puti(const void *p)
{
	const unsigned int *i = (const unsigned int *)p;
	printf("{%d, %d, %d, %d}\n", i[0], i[1], i[2], i[3]);
	printf("{%x, %x, %x, %x}\n", i[0], i[1], i[2], i[3]);
}

static bool s_hasSSE41 = false;

float dummy(float x)
{
	return x;
}

__m128 dummy_ps(__m128 x)
{
	return x;
}

template<typename Func>
float func4(Func f, float x)
{
	return f(x) + f(x + 0.1f) + f(x + 0.2f) + f(x + 0.3f);
}

template<typename Func>
float func_ps(Func f, float x)
{
	MIE_ALIGN(16) float c[] = {
		0, 0.1f, 0.2f, 0.3f
	};
	__m128 in = _mm_set_ps(x, x, x, x);
	in = _mm_add_ps(in, *fmath::local::cast_to<__m128>(c));
	MIE_ALIGN(16) float out[4];
	*reinterpret_cast<__m128*>(out) = f(in);
	return out[0] + out[1] + out[2] + out[3];
}

static inline float dummy4(float x)
{
	return func4(dummy, x);
}

static inline float std_exp4(float x)
{
	return func4(expf, x);
}

static inline float std_log4(float x)
{
	return func4(logf, x);
}

static inline float fmath_exp4(float x)
{
	return func4<float (float)>(fmath::exp, x);
}

static inline float fmath_log4(float x)
{
	return func4(fmath::log, x);
}

static inline float fmath_exp_ps(float x)
{
	return func_ps(fmath::exp_ps, x);
}

#ifdef FMATH_USE_XBYAK
static inline float fmath_exp_psC(float x)
{
	return func_ps(fmath::exp_psC, x);
}
#endif

static inline float fmath_log_ps(float x)
{
	return func_ps(fmath::log_ps, x);
}

#ifdef __INTEL_COMPILER
#include <ia32intrin.h>
static inline float icl_exp_ps(float x)
{
	return func_ps(_mm_exp_ps, x);
}
static inline float icl_log_ps(float x)
{
	return func_ps(_mm_log_ps, x);
}
#endif

void validateAll(float (*org)(float), float (*f)(float), float bf, float ef)
{
	printf("range [%e, %e]\n", bf, ef);
	fmath::local::fi fi, fi2, fi3;
	fi.f = bf;
	const unsigned int b = fi.i;
	fi.f = ef;
	const unsigned int e = fi.i;
	const int n = 2 * 5 + 1, adj = (n - 1) / 2;
	unsigned int diffCount[n] = { 0 };
	unsigned int otherCount = 0;
	int maxDiff = 0;
	int sumDiff = 0;
	float maxDiff_f = 0;

	int count = 0;
	for (unsigned int i = b; i <= e; i++) {
		fi.i = i;
		fi2.f = org(fi.f);
		fi3.f = f(fi.f);
		int diff = fi3.i - fi2.i;
		sumDiff += abs(diff);
		float diff_f = fi3.f - fi2.f;
		if (-adj <= diff && diff < n - adj) {
			diffCount[diff + adj]++;
		} else {
			otherCount++;
//			printf("x=%e(%08x), diff=%d, %e(%08x), %e(%08x)\n", fi.f, fi.i, diff, fi2.f, fi2.i, fi3.f, fi3.i);
		}
		maxDiff = std::max(std::abs(diff), maxDiff);
		maxDiff_f = std::max(::fabsf(diff_f), maxDiff_f);
		count++;
		if (count == 100000000) {
			printf("i=0x%08x, max=%d\n", i, maxDiff);
			count = 0;
		}
	}
	double total = (e - b + 1) * 0.01;
	for (int i = n - 1; i >= 0; i--) {
		printf("%2d->%10d(%5.2f%%)\n", i - adj, diffCount[i], diffCount[i] / total);
	}
	printf("other->%d(%.2f%%), maxDiff=%d, maxDiff(f)=%e\n", otherCount, otherCount / total, maxDiff, maxDiff_f);
	printf("sum=%d, ave diff=%e\n", sumDiff, float(sumDiff) / count);
}

void validateExp(float (*f)(float), const char *msg, float e, bool verifyAll)
{
	float absSum = 0;
	float sum = 0;
	float max = 0;
	int count = 0;
	for (float x = 0; x <= 2; x += e) {
		float a = ::expf(x);
		float b = f(x);
		float diff = ::fabs(a - b);
		max = std::max(max, diff);
		absSum += diff;
		sum += a - b;
		count++;
	}
	printf("%s max=%e, abs ave=%e ave=%e ", msg, max, absSum / count, sum / count);
	max = 0;
	for (float x = -10; x <= 0; x += 0.25f) {
		float diff = ::fabs(::expf(x) - f(x));
		if (diff > 1e-3) printf("%f, %f, %f\n", x, ::expf(x), f(x));
	}
	printf("max=%f\n", max);
	if (verifyAll) {
		validateAll(std::exp, f, -88, 88);
	}
}

void benchmark(float (*f)(float), const char *msg, float b, float e, float d, int N, double *adj, double *pbase, int *pcount)
{
	Xbyak::util::Clock clk;
	float sum = 0;
	int count = 0;
	clk.begin();
	for (int i = 0; i < N; i++) {
		for (float x = b; x <= e; x+= d) {
			sum += f(x);
			count++;
		}
	}
	clk.end();
	if (*adj < 0) {
		*adj = (double)clk.getClock() / count;
		*pcount = count;
		return;
	}
	if (*pbase < 0) {
		*pbase = (double)clk.getClock() / count - *adj;
	}
	double ave = (double)clk.getClock() / count - *adj;
	printf("%20s sum=%10.8e ave=%5.1f/%5.1fclk(x%4.2f)\n", msg, sum / N, ave, ave + *adj, *pbase / ave);
}

struct FuncTbl {
	float (*func)(float);
	const char *name;
} tbl;

template<size_t tblN>
void test(const FuncTbl (&tbl)[tblN], float b, float e, float d, int N)
{
	printf("range=[%.2f, %.2f], d=%e, N=%d\n", b, e, d, N);

	double adj = -1, base = -1;
	int count = 0;
	for (size_t i = 0; i < tblN; i++) {
		if (tbl[i].func == 0) continue;
		benchmark(tbl[i].func, tbl[i].name, b, e, d, N, &adj, &base, &count);
	}
}

void testExp(float b, float e, float d, int N)
{
	const FuncTbl tbl[] = {
		{ dummy, "dummy" },
		{ ::expf, "std::exp" },
#if defined(DEBUG) && defined(FMATH_USE_XBYAK)
		{ fmath::expC, "fmath::expC" },
#endif
		{ fmath::exp, "fmath::exp" },
	};
	test(tbl, b, e, d, N);
}

void testExp_ps(float b, float e, float d, int N)
{
	const FuncTbl tbl[] = {
		{ dummy4, "dummy" },
		{ std_exp4, "std::expx4" },
		{ fmath_exp4, "fmath::expx4" },
#if defined(DEBUG) && defined(FMATH_USE_XBYAK)
		{ fmath_exp_psC, "fmath::exp_psC" },
#endif
		{ fmath_exp_ps, "fmath::exp_ps" },
#ifdef __INTEL_COMPILER
		{ icl_exp_ps, "icl::exp_ps" },
#endif
	};
	test(tbl, b, e, d, N);
}

void mainExp(bool verifyAll)
{
	float e = 1e-6f;
	const int N = 10;

	validateExp(fmath::exp, "fmath::exp", e, verifyAll);

	testExp(0, 2, 1e-6f, N);
	testExp_ps(0, 2, 1e-6f, N);
#if 0
	testExp(0, 4, 1e-5f, N);
	testExp(0, 20, 1e-5f, N);
	testExp(-1, 0, 1e-6f, N);
	testExp(-10, 0, 1e-5f, N);
#endif
}

void testLog(float b, float e, float d, int N)
{
	const FuncTbl tbl[] = {
		{ dummy, "dummy" },
		{ ::logf, "std log" },
		{ fmath::log, "fmath log" },
	};
	test(tbl, b, e, d, N);
}

void testLog_ps(float b, float e, float d, int N)
{
	const FuncTbl tbl[] = {
		{ dummy4, "dummy" },
		{ std_log4, "std logx4" },
		{ fmath_log4, "fmath logx4" },
		{ fmath_log_ps, "fmath log_ps" },
#ifdef __INTEL_COMPILER
		{ icl_log_ps, "icl log_ps" },
#endif
	};
	test(tbl, b, e, d, N);
}

void mainLog(bool verifyAll)
{
	if (verifyAll) {
		// maxDiff= 20
		validateAll(std::log, fmath::log, FLT_MIN, 0.98f);
		// 37
		validateAll(std::log, fmath::log, 0.98f, 0.99f);
		// 638
		validateAll(std::log, fmath::log, 0.99f, 0.9997557998f);

		// 4094
		validateAll(std::log, fmath::log, 0.9997557998f, 1.0f);

		// 4093
		validateAll(std::log, fmath::log, 1.0f, 1.000489f);
		// 512
		validateAll(std::log, fmath::log, 1.000489f, 1.1f);
		// 4
		validateAll(std::log, fmath::log, 1.1f, FLT_MAX);
	}
	float e = 1e-6f;
	const int N = 30;
	testLog(e, 2, 1e-6f, N);
	testLog_ps(e, 2, 1e-6f, N);
}

#ifdef __GNUC__

void testExp2(float b, float e, float d, int N)
{
	const FuncTbl tbl[] = {
		{ dummy, "dummy" },
		{ ::exp2f, "std::exp2" },
		{ fmath::exp2, "fmath::exp2" },
	};
	test(tbl, b, e, d, N);
}

#ifndef __CYGWIN__
void testLog2(float b, float e, float d, int N)
{
	const FuncTbl tbl[] = {
		{ dummy, "dummy" },
		{ ::log2f, "std::log2" },
		{ fmath::log2, "fmath::log2" },
	};
	test(tbl, b, e, d, N);
}
#endif

void test2()
{
	const int N = 30;
	testExp2(0, 2, 1e-6f, N);
#ifndef __CYGWIN__
	float e = 1e-6f;
	testLog2(e, 2, 1e-6f, N);
#endif
}
#endif

static float g_y = 1.234f;
static fmath::PowGenerator p(g_y);

static inline float powa(float x)
{
	return p.get(x);
}

static inline float std_pow(float x)
{
	return ::powf(x, g_y);
}

void testPow(float b, float e, float d, int N)
{
	const FuncTbl tbl[] = {
		{ dummy, "dummy" },
		{ std_pow, "std::pow" },
		{ powa, "fmath::pow" },
	};
	test(tbl, b, e, d, N);
}

void mainPow(bool verifyAll)
{
	if (verifyAll) {
		validateAll(std_pow, powa, 0, 0.98f);
		validateAll(std_pow, powa, 10.0f, 100.0f);
	}
	const int N = 30;
	testPow(0, 2, 1e-6f, N);
}

#ifdef XBYAK32
struct Test : public Xbyak::CodeGenerator {
	explicit Test(bool doFstp)
	{
		static const MIE_ALIGN(16) float in[4] = { 0.010001, -0.020023, 0.003, 0.0004 };
		push(ebp);
		push(esi);
		push(edi);
		mov(esi, ptr [esp + 4 + 12]); // ptr
		mov(edi, ptr [esp + 8 + 12]); // count
		mov(ebp, esp);
		sub(esp, 32);
		and_(esp, ~15u);
	L("lp");
		movaps(xm0, ptr [in]);
		movaps(ptr [esp], xm0);
		call(esi);
		if (doFstp) fstp(st0);
		dec(edi);
		jnz("lp");
		mov(esp, ebp);
		pop(edi);
		pop(esi);
		pop(ebp);
		ret();
	}
};
#endif

void PutVersion()
{
#ifdef XBYAK64
	printf("64bit ");
#else
	printf("32bit ");
#endif
#ifdef __INTEL_COMPILER
	printf("icc\n");
#elif defined(_MSC_VER)
	printf("_MSC_VER=%d\n", _MSC_VER);
#else
	printf("gcc %d.%d\n", __GNUC__, __GNUC_MINOR__);
#endif
#ifdef FMATH_USE_XBYAK
	puts("Xbyak version");
#else
	puts("intrinsic version");
#endif
}

int main(int argc, char *argv[])
{
	PutVersion();
	bool verifyAll = false;
	Xbyak::util::Cpu cpu;
	s_hasSSE41 = cpu.has(Xbyak::util::Cpu::tSSE41);

	if (s_hasSSE41) {
		puts("SSE41 enable");
	}
	argc--, argv++;
	while (argc > 0) {
		if (strcmp(*argv, "-all") == 0) {
			verifyAll = true;
		}
		argc--, argv++;
	}
	printf("verifyAll=%d\n", verifyAll);
#if 1 // #ifdef DEBUG
	MIE_ALIGN(16) float in[] = { 1.234f, 4.342f, -3.234f, 0.122f };
//	MIE_ALIGN(16) float in[] = { 12, -20, 10000, -100000 };
	__m128 out;
	fmath::exp(in[0]);
	printf("in {%e, %e, %e, %e}\n", in[0], in[1], in[2], in[3]);
	printf("std    {%e, %e, %e, %e}\n", ::exp(in[0]), ::exp(in[1]), ::exp(in[2]), ::exp(in[3]));
	printf("fmath  {%e, %e, %e, %e}\n", fmath::exp(in[0]), fmath::exp(in[1]), fmath::exp(in[2]), fmath::exp(in[3]));
	printf("exp_ps "); out = fmath::exp_ps(*(const __m128*)in);
	put(&out);
#ifdef __INTEL_COMPILER
	out = _mm_exp_ps(*(const __m128*)in);
	printf("icc ps "); put(&out);
#endif
#endif

#if defined(XBYAK32) && defined(DEBUG)
	{
		Test test(true);
		void (*func)(void*, int) = (void (*)(void*,int))test.getCode();

		const int n = 100000;
		struct Tbl {
			void *p;
			const char *name;
		} tbl[] = {
			{ (void*)dummy, "dummy" },
			{ (void*)std::expf, "std::exp" },
			{ (void*)fmath::exp, "fmath::exp" },
			{ (void*)fmath::expC, "fmath::expC" },
		};
		for (int i = 0; i < sizeof(tbl) / sizeof(*tbl); i++) {
			Xbyak::util::Clock clk;
			clk.begin();
			printf("% 15s ", tbl[i].name);
			func(tbl[i].p, n);
			clk.end();
			printf("%7.2fk\n", (double)clk.getClock() / n);
		}
	}
	{
		Test test(false);
		void (*func)(void*, int) = (void (*)(void*,int))test.getCode();

		const int n = 100000;
		struct Tbl {
			void *p;
			const char *name;
		} tbl[] = {
			{ (void*)dummy_ps, "dummy_ps" },
			{ (void*)fmath::exp_ps, "fmath::exp_ps" },
			{ (void*)fmath::exp_psC, "fmath::exp_psC" },
#ifdef __INTEL_COMPILER
			{ (void*)_mm_exp_ps, "_mm_exp_ps" },
#endif
		};
		for (int i = 0; i < sizeof(tbl) / sizeof(*tbl); i++) {
			Xbyak::util::Clock clk;
			clk.begin();
			printf("% 15s ", tbl[i].name);
			func(tbl[i].p, n);
			clk.end();
			printf("%7.2fk\n", (double)clk.getClock() / n);
		}
	}
#endif

	mainExp(verifyAll);
	mainLog(verifyAll);
	mainPow(verifyAll);
#ifdef __GNUC__
	test2();
#endif
}
