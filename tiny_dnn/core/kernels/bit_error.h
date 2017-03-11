#include <assert.h>
#include <random>
#include <bitset>
#define ADDER_ERROR_RATE 0.00001
#define MULTI_ERROR_RATE 0.00001
#ifndef BIT_ERROR
#define BIT_ERROR
namespace tiny_dnn {
namespace kernels {
int count_adder = 0;
int max_count_adder = 100000;
std::poisson_distribution<> pd_adder((int)(1.0/ADDER_ERROR_RATE));
std::random_device rd_adder;
std::mt19937 gen_adder(rd_adder());

std::random_device rd;
std::mt19937 gen(rd());

template<size_t size> 
typename std::bitset<size> random_bitset( double p = 0.5) {
	std::bernoulli_distribution d(p);
	typename std::bitset<size> bits;
        for ( int n = 0; n < size; n++ ) { 
                bits[n] = d(gen);
        } 
        return bits;
}       

inline double genErrorDouble(double errorRate, double orig) {
        union Flip
        {
                double input;
                long long output;
        } data, ret;
        data.input = orig;
    
        auto mask = random_bitset<sizeof(double)>(errorRate);
        std::bitset<sizeof(double)> bits(data.output);
        ret.output = ( bits ^ mask ).to_ullong();
        return ret.input;
}

inline float genErrorFloat(double errorRate, float orig) {
        union Flip
        {
                float input;
                long output;
        } data, ret;
        data.input = orig;
    
        auto mask = random_bitset<sizeof(float)>(errorRate);
        std::bitset<sizeof(float)> bits(data.output);
        ret.output = ( bits ^ mask ).to_ullong();
        return ret.input;
}

int count_multi = 0;
int max_count_multi = 100000;
std::poisson_distribution<> pd_multi((int)(1.0/MULTI_ERROR_RATE));
std::random_device rd_multi;
std::mt19937 gen_multi(rd_multi());

inline double FlipAdder(double operand) {
	if ( count_adder > max_count_adder ) {
		count_adder = 0;
		max_count_adder = pd_adder(gen_adder);
		double error = genErrorDouble(ADDER_ERROR_RATE, operand);
		return error;		
	} else {
		count_adder++;
		return operand;
	}
}

inline double FlipMulti(double operand) {
	if ( count_multi > max_count_multi ) {
		count_multi = 0;
		max_count_multi = pd_multi(gen_multi);
		double error = genErrorDouble(MULTI_ERROR_RATE, operand);
		return error;
	} else {
		count_multi++;
		return operand;
	}
}

}
}
#endif
