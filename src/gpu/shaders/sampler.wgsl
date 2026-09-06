// The sample pattern, on the device. A port of `sampler.rs`, bit for bit:
// `tests/gpu_sampler.rs` compares the two over a grid of (pixel, frame, dim).
//
// Composed after the prelude by `shaders::trace_shader`, with the mask
// spliced in by `shaders::sampler_shader` as `BLUE_NOISE_MASK` — 128×128
// shift vectors, two bytes (u, v) each, packed four bytes to a u32, row-major.

const MASK_SIZE: u32 = 128u;

// A hash-based Owen scramble of the base-2 digits of `x`, most significant
// first, keyed by `seed` (PBRT-v4's FastOwenScrambler; see `sampler.rs`).
fn nested_uniform_scramble(x: u32, seed: u32) -> u32 {
    var v = reverseBits(x);
    v ^= v * 0x3d20adeau;
    v += seed;
    v *= (seed >> 16u) | 1u;
    v ^= v * 0x05526c56u;
    v ^= v * 0x53a22864u;
    return reverseBits(v);
}

// lowbias32.
fn hash_u32(x_in: u32) -> u32 {
    var x = x_in;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

// Sobol dimension 1's direction numbers: m_k = m_{k-1} ^ (m_{k-1} << 1)
// from m_1 = 1, each shifted to the top of the word. Dimension 0 is the
// bit-reversed index and needs no table.
const SOBOL_DIM1: array<u32, 32> = array<u32, 32>(
    0x80000000u, 0xc0000000u, 0xa0000000u, 0xf0000000u, 0x88000000u, 0xcc000000u, 0xaa000000u, 0xff000000u,
    0x80800000u, 0xc0c00000u, 0xa0a00000u, 0xf0f00000u, 0x88880000u, 0xcccc0000u, 0xaaaa0000u, 0xffff0000u,
    0x80008000u, 0xc000c000u, 0xa000a000u, 0xf000f000u, 0x88008800u, 0xcc00cc00u, 0xaa00aa00u, 0xff00ff00u,
    0x80808080u, 0xc0c0c0c0u, 0xa0a0a0a0u, 0xf0f0f0f0u, 0x88888888u, 0xccccccccu, 0xaaaaaaaau, 0xffffffffu,
);

fn sobol_dim1(index: u32) -> u32 {
    var x = 0u;
    var i = index;
    var k = 0u;
    loop {
        if i == 0u { break; }
        if (i & 1u) != 0u {
            x ^= SOBOL_DIM1[k];
        }
        i >>= 1u;
        k += 1u;
    }
    return x;
}

// The mask's shift vector at (x, y), on the torus: (u, v) in 256ths.
fn mask_at(x: u32, y: u32) -> vec2<u32> {
    let i = ((y % MASK_SIZE) * MASK_SIZE + (x % MASK_SIZE)) * 2u;
    let word = BLUE_NOISE_MASK[i >> 2u] >> ((i & 3u) * 8u);
    return vec2<u32>(word & 0xFFu, (word >> 8u) & 0xFFu);
}

// The blue-noise pattern at (pixel, frame, dim): one Owen-scrambled Sobol
// pair per dimension pair, indexed by the frame, shifted per pixel by the
// mask read at the dimension's own offset. See `sampler.rs` for why.
fn blue_noise_sample(pixel: vec2<u32>, frame: u32, dim: u32) -> f32 {
    let pair = dim >> 1u;
    let comp = dim & 1u;
    let index = nested_uniform_scramble(frame, hash_u32(pair + 0x9E3779B9u));
    var raw = reverseBits(index);
    if comp != 0u {
        raw = sobol_dim1(index);
    }
    let scrambled = nested_uniform_scramble(raw, hash_u32(dim + 0x7F4A7C15u));
    let value = f32(scrambled >> 8u) * (1.0 / 16777216.0);
    let shift = f32(mask_at(pixel.x, pixel.y)[comp]) * (1.0 / 256.0);
    return fract(value + shift);
}

// The white-noise hash the integrator has always drawn from, as a function
// of the same key, so the two patterns are compared under one harness.
fn white_noise_sample(pixel: vec2<u32>, frame: u32, dim: u32) -> f32 {
    var state = pixel.x * 1973u + pixel.y * 9277u + dim * 26699u + frame * 12345u + 1u;
    state = state * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    let r = (word >> 22u) ^ word;
    return f32(r) / 4294967296.0;
}
