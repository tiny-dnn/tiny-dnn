#include <math.h>
#include <stdio.h>
#include "xbyak/xbyak_util.h"
/*
x = ([ax] + e)/a, |e| < 1/2
t = e/a
t^4/24 <= 2^(-52)
e/a = t <= (24 * 2^(-52))^(1/4) = 24^(1/4) 2^(-13)
a >= (4/24^(1/4)) * 2^(-10)

e^(1/a) = 2^(1/s) => s = a log 2 >= 1282.7


m=(e/a)^4/24 = (log 2)^4/(2^7 3) / s^4
s = 1024 m=5.46727341522255e-16
s = 2048 m=3.417045884514094e-17
*/
#include "fmath.hpp"
typedef unsigned long long uint64_t;

union di {
	double d;
	uint64_t i;
};

uint64_t roundC(double x)
{
	di di;
	di.d = x + (3LL << 51);
	return ((long long)(di.i << 13)) >> 13;
}

void err()
{
	double max = 0;
	double ave = 0;
	int count = 0;
	int up = 0;
	int down = 0;
	for (double x = 0; x < 2.00001; x += 1e-7) {
		double a = ::exp(x);
		double b = fmath::expd(x);
		if (a > b) up++;
		if (a < b) down++;
		double d = fabs(a - b);
		if (d > max) {
			max = d;
		}
		ave += d;
		count++;
//		printf("%e, %e, %e, %e\n", x, a, b, a - b);
	}
	printf("err:ave=%.10e, max=%.10e\n", ave / count, max);
	printf("up=%d, down=%d\n", up, down);
}

void benchmark(const char *str, double f(double))
{
	double a = 0;
	Xbyak::util::Clock clk;
	clk.begin();
	int n = 0;
	for (double x = 0; x < 1; x += 1e-8) {
		a += f(x);
		n++;
	}
	clk.end();
	printf("%s %.3fclk, a=%f\n", str, clk.getClock() / double(n), a);
}

void benchmark_v()
{
	const int n = 1024;
	MIE_ALIGN(16) double org[n], px[n];
	for (int i = 0; i < n; i++) {
		org[i] = i / 1000.0;
	}
	Xbyak::util::Clock clk;
	for (int i = 0; i < 100; i++) {
		std::copy(org, org + n, px);
		clk.begin();
		fmath::expd_v(px, n);
		clk.end();
	}
	printf("expd_v:%.3fclk\n", clk.getClock() / double(n) / clk.getCount());
}

int main()
{
	puts("IN");
	printf("exp  : %.17f\n",         exp(2.0000314));
	printf("expd : %.17f\n", fmath::expd(2.0000314));
//	printf("expdC: %.17f\n", fmath::expdC(2.0000314));
	MIE_ALIGN(16) double px[] = { 2.0000314, 1.0 };
	fmath::expd_v(px, 2);
	printf("%.17f %.17f\n", px[0], px[1]);
	err();
	benchmark("std::exp    ", ::exp);
//	benchmark("fmath::expdC", fmath::expdC);
	benchmark("fmath::expd ", fmath::expd);
	benchmark_v();
#if 0
	const double tbl[] = {
		0, 0.4, 0.5, 0.6, 1.2, 1.5, 1.9, 2.5, 3.5, 123.456,
		(1 << 22) - 1,
		(1 << 22) - 0.5,
		(1 << 22),
	};
	for (size_t i = 0; i < sizeof(tbl) / sizeof(*tbl); i++) {
		double x = tbl[i];
		printf("roundC( %f)= %lld\n", x, roundC(x));
		printf("roundC(%f)=%lld\n", -x, roundC(-x));
	}
#endif
}
