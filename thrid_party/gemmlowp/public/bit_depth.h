// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// bit_depth.h: defines the BitDepthSetting enum

#ifndef GEMMLOWP_PUBLIC_BIT_DEPTH_H_
#define GEMMLOWP_PUBLIC_BIT_DEPTH_H_

namespace gemmlowp {

// A specific bit depth to requantize an operand (Lhs or Rhs) to.
// The case tBits==8 means no requantization, since at the moment
// we only accept 8-bit input data.
template <int tBits>
struct BitDepth {
  static const int kBits = tBits;
  static_assert(kBits >= 1 && kBits <= 8, "bad bit depth");
};

// A rounding mode to use when requantizing an operand.
// The requantizing operation is:
//   dst = (src * maxval + rounding_offset) / 255;
// Where dst and src are uint8, maxval is 2^(dstbits)-1,
// and the intermediate values are computed as uint16s
// so no overflow occurs.
// The rounding_offset in the above formula is a value
// in [0..254] determined by the RoundingMode as follows:
enum class RoundingMode {
  Exact,                  // No rounding, do nothing. Use with bit_depth == 8.
  Nearest,                // rounding_offset = 127
  ProbabilisticXorshift,  // rounding_offset given by 8-bit Xorshift PRNG
  ProbabilisticAddmod     // rounding_offset given by 8-bit add/mod LDSG
};

// A rounding strategy is a heuristic for choosing a rounding mode.
// When the bit depth is 8 bit like the source, there is no
// quantization to be done, so this is moot. In this case, we use
// the following "no-op" "strategy",
struct ExactRoundingStrategyFor8Bit {
  static const RoundingMode kRoundingModeForSmallSizes = RoundingMode::Exact;
  static const RoundingMode kRoundingModeForLargeSizes = RoundingMode::Exact;
  static const int kRoundingModeSizeThreshold = 0;
};

// Default rounding strategy when actually requantizing to less than 8 bit.
// Round-to-nearest tends to give the best results for small enough
// accumulation sizes (i.e. accumulation depth, but we refrain from using
// the word "depth" here as it gets confusing with "bit depth").
// Some flavor of probabilistic tends to perform better for larger sizes.
// See doc/less-than-8-bit.txt for details.
struct DefaultRoundingStrategyForLessThan8Bit {
  static const RoundingMode kRoundingModeForSmallSizes = RoundingMode::Nearest;
  static const RoundingMode kRoundingModeForLargeSizes =
      RoundingMode::ProbabilisticAddmod;

  // The threshold on the depth dimension at which we switch to
  // probabilistic rounding instead of rounding-to-nearest when
  // requantizing input data. Indeed, both statistical theory and
  // empirical measurements show that for given input data and bit depth,
  // probabilistic rounding gives more accurate results for large enough
  // depth, while rounding-to-nearest does for smaller depth. This threshold
  // is naively determined from some experiments with Inception at 7bit/5bit
  // on a set of 10,000 images with 8-bit Xorshift probabilistic rounding:
  //
  //   7 bit weights, 5 bit activations, switch at 64:   59.82% top-1 accuracy
  //   7 bit weights, 5 bit activations, switch at 128:  59.58% top-1 accuracy
  //   7 bit weights, 5 bit activations, switch at 192:  63.37% top-1 accuracy
  //   7 bit weights, 5 bit activations, switch at 256:  63.47% top-1 accuracy
  //   7 bit weights, 5 bit activations, switch at 320:  63.71% top-1 accuracy
  //   7 bit weights, 5 bit activations, switch at 384:  63.71% top-1 accuracy
  //   7 bit weights, 5 bit activations, switch at 448:  63.58% top-1 accuracy
  //   7 bit weights, 5 bit activations, switch at 512:  64.10% top-1 accuracy
  //   7 bit weights, 5 bit activations, switch at 640:  62.49% top-1 accuracy
  //   7 bit weights, 5 bit activations, switch at 768:  62.49% top-1 accuracy
  //   7 bit weights, 5 bit activations, switch at 1024: 58.96% top-1 accuracy
  //
  // So here, 384 looks comfortably in the middle of a plateau of good values,
  // and it's a roundish number (3/2 * 256) so let's stick with that for now.
  // It would be nice to work out the theory of this, and understand how this
  // should depend on the distribution of inputs and the bit depth.
  //
  // Repeating the same evaluation with AddMod:
  //   7 bit weights, 5 bit activations, switch at 64:   62.65% top-1 accuracy
  //   7 bit weights, 5 bit activations, switch at 128:  62.65% top-1 accuracy
  //   7 bit weights, 5 bit activations, switch at 192:  63.81% top-1 accuracy
  //   7 bit weights, 5 bit activations, switch at 256:  64.23% top-1 accuracy
  //   7 bit weights, 5 bit activations, switch at 320:  64.16% top-1 accuracy
  //   7 bit weights, 5 bit activations, switch at 384:  64.16% top-1 accuracy
  //   7 bit weights, 5 bit activations, switch at 448:  64.16% top-1 accuracy
  //   7 bit weights, 5 bit activations, switch at 512:  64.52% top-1 accuracy
  //   7 bit weights, 5 bit activations, switch at 640:  62.74% top-1 accuracy
  //   7 bit weights, 5 bit activations, switch at 768:  62.74% top-1 accuracy
  //   7 bit weights, 5 bit activations, switch at 1024: 59.74% top-1 accuracy
  //
  // The behavior is similar, so 384 remains a good choice.

  static const int kRoundingModeSizeThreshold = 384;
};

struct DefaultL8R8BitDepthParams {
  typedef BitDepth<8> LhsBitDepth;
  typedef BitDepth<8> RhsBitDepth;
  typedef ExactRoundingStrategyFor8Bit RoundingStrategy;
};

struct DefaultL7R5BitDepthParams {
  typedef BitDepth<7> LhsBitDepth;
  typedef BitDepth<5> RhsBitDepth;
  typedef DefaultRoundingStrategyForLessThan8Bit RoundingStrategy;
};

}  // namespace gemmlowp

#endif  // GEMMLOWP_PUBLIC_BIT_DEPTH_H_
