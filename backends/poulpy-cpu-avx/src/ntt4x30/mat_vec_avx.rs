// ----------------------------------------------------------------------
// DISCLAIMER
//
// This module contains code derived from the spqlios-arithmetic library
// (https://github.com/tfhe/spqlios-arithmetic), which is licensed
// under the Apache License, Version 2.0.
//
// The current implementation is an updated version of the spqlios
// port. The main changes from the original:
//   - prime-major VMP kernels multiply centered residues as i32
//     against a matrix stored in Montgomery form;
//   - each accumulator ends with one signed Montgomery REDC and the
//     kernels return canonical residues.
//
// Both Poulpy and spqlios-arithmetic are distributed under the terms
// of the Apache License, Version 2.0. See the LICENSE file for details.
//
// ----------------------------------------------------------------------

//! AVX2-accelerated Q120 matrix-vector dot products.
//!
//! Rust ports of the BBC (q120b × q120c → q120b) inner-product functions
//! from `q120_arithmetic_avx2.c` in spqlios-arithmetic.
//!
//! # Layout conventions
//!
//! | Format | Bytes/element | AVX2 view |
//! |--------|--------------|-----------|
//! | q120b  | 32 (4 × u64) | one `__m256i` |
//! | q120c  | 32 (8 × u32) | one `__m256i` |
//! | x2-block | 64 (2 × q120b/c) | two `__m256i`s |
//!
//! All three exported functions share the same two-accumulator strategy:
//! accumulate low-32-bit and high-32-bit partial products separately,
//! then collapse with the precomputed `s2l_pow_red` / `s2h_pow_red`
//! constants from [`BbcMeta`].
//!
//! # Functions
//!
//! | AVX2 function | spqlios C equivalent | Trait |
//! |---|---|---|
//! | [`vec_mat1col_product_bbc_avx2`] | `q120_vec_mat1col_product_bbc_avx2` | [`NttMulBbc`] |
//! | [`vec_mat1col_product_x2_bbc_avx2`] | `q120x2_vec_mat1col_product_bbc_avx2` | [`NttMulBbc1ColX2`] |
//! | [`vec_mat2cols_product_x2_bbc_avx2`] | `q120x2_vec_mat2cols_product_bbc_avx2` | [`NttMulBbc2ColsX2`] |

use core::arch::x86_64::{
    __m256i, _mm_cvtsi64_si128, _mm256_add_epi64, _mm256_and_si256, _mm256_loadu_si256, _mm256_mul_epi32, _mm256_mul_epu32,
    _mm256_set1_epi64x, _mm256_setzero_si256, _mm256_srl_epi64, _mm256_srli_epi64, _mm256_storeu_si256,
};

use poulpy_cpu_portable::kernels::ntt4x30::{mat_vec::BbcMeta, primes::Primes30};

use super::arithmetic_avx::{Q_VEC, QINV, add_mod_canonical, signed_redc_canonical};

// ─────────────────────────────────────────────────────────────────────────────
// Shared final-reduction helper
// ─────────────────────────────────────────────────────────────────────────────

/// Collapse two accumulator vectors `(s_lo, s_hi)` into a q120b result.
///
/// Computes `res = s_lo + (s_hi & mask_h2) * S2L + (s_hi >> H2) * S2H`,
/// matching the final-reduction step shared by all three BBC functions.
#[inline(always)]
pub(crate) unsafe fn reduce_bbc(s_lo: __m256i, s_hi: __m256i, mask_h2: __m256i, h2: u64, s2l: __m256i, s2h: __m256i) -> __m256i {
    unsafe {
        let h2_count = _mm_cvtsi64_si128(h2 as i64);
        let hi_lo = _mm256_and_si256(s_hi, mask_h2);
        let hi_hi = _mm256_srl_epi64(s_hi, h2_count);
        let t = _mm256_add_epi64(s_lo, _mm256_mul_epu32(hi_lo, s2l));
        _mm256_add_epi64(t, _mm256_mul_epu32(hi_hi, s2h))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Single-column: q120b × q120c → q120b
// ─────────────────────────────────────────────────────────────────────────────

/// AVX2 inner product: `res = Σᵢ x[i] · y[i]` in q120b format.
///
/// Port of `q120_vec_mat1col_product_bbc_avx2`.
///
/// - `x`: q120b in u32 view — `ell` elements × 8 u32 (one `__m256i` each).
/// - `y`: q120c — `ell` elements × 8 u32 (one `__m256i` each).
/// - `res`: q120b output — at least 4 u64 (one `__m256i`).
///
/// # Safety
///
/// Caller must ensure AVX2 support. Slice lengths must satisfy
/// `x.len() >= 8 * ell`, `y.len() >= 8 * ell`, `res.len() >= 4`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vec_mat1col_product_bbc_avx2(meta: &BbcMeta<Primes30>, ell: usize, res: &mut [u64], x: &[u32], y: &[u32]) {
    unsafe {
        let mask32 = _mm256_set1_epi64x(u32::MAX as i64);
        let mut s1 = _mm256_setzero_si256(); // accumulator for low 32-bit partial products
        let mut s2 = _mm256_setzero_si256(); // accumulator for high 32-bit partial products

        let mut x_ptr = x.as_ptr() as *const __m256i;
        let mut y_ptr = y.as_ptr() as *const __m256i;

        for _ in 0..ell {
            let xv = _mm256_loadu_si256(x_ptr);
            let xl = _mm256_and_si256(xv, mask32); // low 32 bits of each u64 lane
            let xh = _mm256_srli_epi64::<32>(xv); // high 32 bits shifted down

            let yv = _mm256_loadu_si256(y_ptr);
            let y0 = _mm256_and_si256(yv, mask32); // y_i mod 2^32 (= r_i mod Q)
            let y1 = _mm256_srli_epi64::<32>(yv); // y_i >> 32 (= (r_i * 2^32) mod Q)

            // a = xl * y0  (xl and y0 are both 32-bit, result ≤ 64 bits)
            let a = _mm256_mul_epu32(xl, y0);
            // b = xh * y1
            let b = _mm256_mul_epu32(xh, y1);

            s1 = _mm256_add_epi64(s1, _mm256_and_si256(a, mask32));
            s1 = _mm256_add_epi64(s1, _mm256_and_si256(b, mask32));
            s2 = _mm256_add_epi64(s2, _mm256_srli_epi64::<32>(a));
            s2 = _mm256_add_epi64(s2, _mm256_srli_epi64::<32>(b));

            x_ptr = x_ptr.add(1);
            y_ptr = y_ptr.add(1);
        }

        let mask_h2 = _mm256_set1_epi64x(((1u64 << meta.h) - 1) as i64);
        let s2l_pow_red = _mm256_loadu_si256(meta.s2l_pow_red.as_ptr() as *const __m256i);
        let s2h_pow_red = _mm256_loadu_si256(meta.s2h_pow_red.as_ptr() as *const __m256i);

        let t = reduce_bbc(s1, s2, mask_h2, meta.h, s2l_pow_red, s2h_pow_red);
        _mm256_storeu_si256(res.as_mut_ptr() as *mut __m256i, t);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// x2-block, single column: two q120b × q120c pairs → two q120b results
// ─────────────────────────────────────────────────────────────────────────────

/// Tiled canonical-x bbc product: two sliding-window outputs per pass
/// (chained twice by the tile-4 entry). `x` rows hold the canonical u32
/// encoding (odd lanes zero), so the `x_hi * y_hi` product path is
/// identically zero and skipped.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vec_mat_tile2_bbc_canonical_avx2(
    meta: &BbcMeta<Primes30>,
    len: usize,
    res: &mut [u64],
    x: &[u32],
    y: &[u32],
) {
    unsafe {
        assert!(res.len() >= 16);
        assert!(len == 0 || x.len() >= 16 * (len + 1));
        assert!(y.len() >= 16 * len);

        let mask32 = _mm256_set1_epi64x(u32::MAX as i64);

        // Per output t and x2 pair half: (low, high) accumulators.
        let mut s0a = _mm256_setzero_si256();
        let mut s1a = _mm256_setzero_si256();
        let mut s0b = _mm256_setzero_si256();
        let mut s1b = _mm256_setzero_si256();
        let mut t0a = _mm256_setzero_si256();
        let mut t1a = _mm256_setzero_si256();
        let mut t0b = _mm256_setzero_si256();
        let mut t1b = _mm256_setzero_si256();

        let mut x_ptr = x.as_ptr() as *const __m256i;
        let mut y_ptr = y.as_ptr() as *const __m256i;

        for _ in 0..len {
            let ya = _mm256_loadu_si256(y_ptr);
            let yb = _mm256_loadu_si256(y_ptr.add(1));

            let x0a = _mm256_loadu_si256(x_ptr);
            let x0b = _mm256_loadu_si256(x_ptr.add(1));
            let x1a = _mm256_loadu_si256(x_ptr.add(2));
            let x1b = _mm256_loadu_si256(x_ptr.add(3));

            let p0a = _mm256_mul_epu32(x0a, ya);
            let p0b = _mm256_mul_epu32(x0b, yb);
            let p1a = _mm256_mul_epu32(x1a, ya);
            let p1b = _mm256_mul_epu32(x1b, yb);

            s0a = _mm256_add_epi64(s0a, _mm256_and_si256(p0a, mask32));
            s1a = _mm256_add_epi64(s1a, _mm256_srli_epi64::<32>(p0a));
            s0b = _mm256_add_epi64(s0b, _mm256_and_si256(p0b, mask32));
            s1b = _mm256_add_epi64(s1b, _mm256_srli_epi64::<32>(p0b));
            t0a = _mm256_add_epi64(t0a, _mm256_and_si256(p1a, mask32));
            t1a = _mm256_add_epi64(t1a, _mm256_srli_epi64::<32>(p1a));
            t0b = _mm256_add_epi64(t0b, _mm256_and_si256(p1b, mask32));
            t1b = _mm256_add_epi64(t1b, _mm256_srli_epi64::<32>(p1b));

            x_ptr = x_ptr.add(2);
            y_ptr = y_ptr.add(2);
        }

        let mask_h2 = _mm256_set1_epi64x(((1u64 << meta.h) - 1) as i64);
        let s2l_pow_red = _mm256_loadu_si256(meta.s2l_pow_red.as_ptr() as *const __m256i);
        let s2h_pow_red = _mm256_loadu_si256(meta.s2h_pow_red.as_ptr() as *const __m256i);

        let res_ptr = res.as_mut_ptr() as *mut __m256i;
        _mm256_storeu_si256(res_ptr, reduce_bbc(s0a, s1a, mask_h2, meta.h, s2l_pow_red, s2h_pow_red));
        _mm256_storeu_si256(
            res_ptr.add(1),
            reduce_bbc(s0b, s1b, mask_h2, meta.h, s2l_pow_red, s2h_pow_red),
        );
        _mm256_storeu_si256(
            res_ptr.add(2),
            reduce_bbc(t0a, t1a, mask_h2, meta.h, s2l_pow_red, s2h_pow_red),
        );
        _mm256_storeu_si256(
            res_ptr.add(3),
            reduce_bbc(t0b, t1b, mask_h2, meta.h, s2l_pow_red, s2h_pow_red),
        );
    }
}

/// AVX2 x2-block inner product: one column, two paired rows.
///
/// Port of `q120x2_vec_mat1col_product_bbc_avx2`.
///
/// Computes two q120b inner products simultaneously using two interleaved
/// q120b/q120c pairs per step:
/// - `res[0..4]` ← `Σᵢ x_a[i] · y_a[i]`
/// - `res[4..8]` ← `Σᵢ x_b[i] · y_b[i]`
///
/// - `x`: x2-block in u32 view — `ell` elements × 16 u32 (two `__m256i`s each).
/// - `y`: x2-block q120c — `ell` elements × 16 u32 (two `__m256i`s each).
/// - `res`: two q120b outputs — at least 8 u64.
///
/// # Safety
///
/// Caller must ensure AVX2 support. Slice lengths must satisfy
/// `x.len() >= 16 * ell`, `y.len() >= 16 * ell`, `res.len() >= 8`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vec_mat1col_product_x2_bbc_avx2(
    meta: &BbcMeta<Primes30>,
    ell: usize,
    res: &mut [u64],
    x: &[u32],
    y: &[u32],
) {
    unsafe {
        let mask32 = _mm256_set1_epi64x(u32::MAX as i64);
        // Pair A accumulators
        let mut s0 = _mm256_setzero_si256(); // low sums  for x_a * y_a
        let mut s1 = _mm256_setzero_si256(); // high sums for x_a * y_a
        // Pair B accumulators
        let mut s2 = _mm256_setzero_si256(); // low sums  for x_b * y_b
        let mut s3 = _mm256_setzero_si256(); // high sums for x_b * y_b

        let mut x_ptr = x.as_ptr() as *const __m256i;
        let mut y_ptr = y.as_ptr() as *const __m256i;

        for _ in 0..ell {
            // Pair A: x[2i] × y[2i]
            let xa = _mm256_loadu_si256(x_ptr);
            let xa_hi = _mm256_srli_epi64::<32>(xa);
            let ya = _mm256_loadu_si256(y_ptr);
            let ya_hi = _mm256_srli_epi64::<32>(ya);

            let prod_a_lo = _mm256_mul_epu32(xa, ya);
            let prod_a_hi = _mm256_mul_epu32(xa_hi, ya_hi);

            s0 = _mm256_add_epi64(s0, _mm256_and_si256(prod_a_lo, mask32));
            s0 = _mm256_add_epi64(s0, _mm256_and_si256(prod_a_hi, mask32));
            s1 = _mm256_add_epi64(s1, _mm256_srli_epi64::<32>(prod_a_lo));
            s1 = _mm256_add_epi64(s1, _mm256_srli_epi64::<32>(prod_a_hi));

            // Pair B: x[2i+1] × y[2i+1]
            let xb = _mm256_loadu_si256(x_ptr.add(1));
            let xb_hi = _mm256_srli_epi64::<32>(xb);
            let yb = _mm256_loadu_si256(y_ptr.add(1));
            let yb_hi = _mm256_srli_epi64::<32>(yb);

            let prod_b_lo = _mm256_mul_epu32(xb, yb);
            let prod_b_hi = _mm256_mul_epu32(xb_hi, yb_hi);

            s2 = _mm256_add_epi64(s2, _mm256_and_si256(prod_b_lo, mask32));
            s2 = _mm256_add_epi64(s2, _mm256_and_si256(prod_b_hi, mask32));
            s3 = _mm256_add_epi64(s3, _mm256_srli_epi64::<32>(prod_b_lo));
            s3 = _mm256_add_epi64(s3, _mm256_srli_epi64::<32>(prod_b_hi));

            x_ptr = x_ptr.add(2);
            y_ptr = y_ptr.add(2);
        }

        let mask_h2 = _mm256_set1_epi64x(((1u64 << meta.h) - 1) as i64);
        let s2l_pow_red = _mm256_loadu_si256(meta.s2l_pow_red.as_ptr() as *const __m256i);
        let s2h_pow_red = _mm256_loadu_si256(meta.s2h_pow_red.as_ptr() as *const __m256i);

        let res_ptr = res.as_mut_ptr() as *mut __m256i;
        let out0 = reduce_bbc(s0, s1, mask_h2, meta.h, s2l_pow_red, s2h_pow_red);
        let out1 = reduce_bbc(s2, s3, mask_h2, meta.h, s2l_pow_red, s2h_pow_red);
        _mm256_storeu_si256(res_ptr, out0);
        _mm256_storeu_si256(res_ptr.add(1), out1);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Block-pair, single column, prime-major input × packed prime-pair matrix
// ─────────────────────────────────────────────────────────────────────────────

/// Rows an accumulator takes per chunk: it sums products of centered residues (magnitude below `q^2 / 4`) from
/// zero and ends with one signed Montgomery redution, which needs `|acc| < q * 2^31`: 8 products with `q < 2^30`.
/// Longer sums are split into chunks whose canonical results are added mod `q`.
const CENTERED_ROWS: usize = 8;

/// `Σ x[row] * y[row] * 2^-32 mod q` over `rows >= 1` rows of one prime lane, canonical in `[0, q)`.
///
/// `x` and `y` hold centered residues of that prime in the low 32 bits of each 64-bit lane (`y` shifted down
/// for an odd prime of the pair), read as `i32` by `_mm256_mul_epi32`; `y` is in Montgomery form `y * 2^32`, so
/// the result is `Σ x * y`. `load(row)` returns the pair `(x, y)` of a row.
#[inline(always)]
unsafe fn centered_dot(rows: usize, p: usize, mut load: impl FnMut(usize) -> (__m256i, __m256i)) -> __m256i {
    unsafe {
        let q = _mm256_set1_epi64x(Q_VEC[p] as i64);
        let qinv = _mm256_set1_epi64x(QINV[p] as i64);
        let mut value = _mm256_setzero_si256();
        let mut row = 0;
        while row < rows {
            let end = rows.min(row + CENTERED_ROWS);
            let mut acc = _mm256_setzero_si256();
            while row < end {
                let (x, y) = load(row);
                acc = _mm256_add_epi64(acc, _mm256_mul_epi32(x, y));
                row += 1;
            }
            let r = signed_redc_canonical(acc, q, qinv);
            value = if end <= CENTERED_ROWS {
                r
            } else {
                add_mod_canonical(value, r, q)
            };
        }
        value
    }
}

/// AVX2 block-pair inner product over a prime-major VMP layout.
///
/// `x_pm` contains 4 prime planes. Each plane stores `ell` rows of 4 u64
/// values with lane order `[blk0.c0, blk0.c1, blk1.c0, blk1.c1]`, each a centered residue (as an `i64`).
///
/// `y_pm` contains two prime-pair planes with the same per-row lane order.
/// Each u64 packs two centered residues as `i32` words: `p_even | p_odd << 32`.
/// `y_pair_stride` is the distance, in u64, between the two planes.
///
/// `y` is in Montgomery form (`y * 2^32 mod q`, centered). The output is two x2-blocks of canonical residues
/// laid out as 16 u64: `[blk0.c0[4], blk0.c1[4], blk1.c0[4], blk1.c1[4]]`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vec_mat1col_product_blkpair_bbc_pm_avx2(
    ell: usize,
    res: &mut [u64],
    x_pm: &[u64],
    y_pm: &[u64],
    y_pair_stride: usize,
) {
    unsafe {
        assert!(res.len() >= 16);
        assert!(x_pm.len() >= 16 * ell);
        assert!(y_pm.len() >= y_pair_stride + 4 * ell);

        let x_plane_stride = 4 * ell;
        let mut prime_outputs = [0u64; 16];

        for p in 0..4usize {
            let x_ptr = x_pm.as_ptr().add(p * x_plane_stride) as *const __m256i;
            let y_ptr = y_pm.as_ptr().add((p / 2) * y_pair_stride) as *const __m256i;
            let odd_prime = p & 1 != 0;
            let out = centered_dot(ell, p, |row| {
                let yv = _mm256_loadu_si256(y_ptr.add(row));
                (
                    _mm256_loadu_si256(x_ptr.add(row)),
                    if odd_prime { _mm256_srli_epi64::<32>(yv) } else { yv },
                )
            });
            _mm256_storeu_si256(prime_outputs.as_mut_ptr().add(4 * p) as *mut __m256i, out);
        }

        for coeff in 0..4 {
            for prime in 0..4 {
                res[4 * coeff + prime] = prime_outputs[4 * prime + coeff];
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Block-pair, one column, two right-hand sides
// ─────────────────────────────────────────────────────────────────────────────

/// Applies one prepared-key column to two input digit spectra, with the operands and output of
/// [`vec_mat1col_product_blkpair_bbc_pm_avx2`].
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vec_mat1col_product_blkpair_bbc_pm_x2_avx2(
    ell: usize,
    res0: &mut [u64],
    res1: &mut [u64],
    x0_pm: &[u64],
    x1_pm: &[u64],
    y_pm: &[u64],
    y_pair_stride: usize,
) {
    unsafe {
        assert!(res0.len() >= 16);
        assert!(res1.len() >= 16);
        assert!(x0_pm.len() >= 16 * ell);
        assert!(x1_pm.len() >= 16 * ell);
        assert!(y_pm.len() >= y_pair_stride + 4 * ell);

        let x_plane_stride = 4 * ell;
        let mut prime_outputs0 = [0u64; 16];
        let mut prime_outputs1 = [0u64; 16];

        for p in 0..4usize {
            let q = _mm256_set1_epi64x(Q_VEC[p] as i64);
            let qinv = _mm256_set1_epi64x(QINV[p] as i64);
            let x0_ptr = x0_pm.as_ptr().add(p * x_plane_stride) as *const __m256i;
            let x1_ptr = x1_pm.as_ptr().add(p * x_plane_stride) as *const __m256i;
            let y_ptr = y_pm.as_ptr().add((p / 2) * y_pair_stride) as *const __m256i;
            let odd_prime = p & 1 != 0;
            let (mut value0, mut value1) = (_mm256_setzero_si256(), _mm256_setzero_si256());
            let mut row = 0;
            while row < ell {
                let end = ell.min(row + CENTERED_ROWS);
                let (mut acc0, mut acc1) = (_mm256_setzero_si256(), _mm256_setzero_si256());
                while row < end {
                    let y = _mm256_loadu_si256(y_ptr.add(row));
                    let y = if odd_prime { _mm256_srli_epi64::<32>(y) } else { y };
                    acc0 = _mm256_add_epi64(acc0, _mm256_mul_epi32(_mm256_loadu_si256(x0_ptr.add(row)), y));
                    acc1 = _mm256_add_epi64(acc1, _mm256_mul_epi32(_mm256_loadu_si256(x1_ptr.add(row)), y));
                    row += 1;
                }
                let (r0, r1) = (signed_redc_canonical(acc0, q, qinv), signed_redc_canonical(acc1, q, qinv));
                if end <= CENTERED_ROWS {
                    (value0, value1) = (r0, r1);
                } else {
                    (value0, value1) = (add_mod_canonical(value0, r0, q), add_mod_canonical(value1, r1, q));
                }
            }
            _mm256_storeu_si256(prime_outputs0.as_mut_ptr().add(4 * p) as *mut __m256i, value0);
            _mm256_storeu_si256(prime_outputs1.as_mut_ptr().add(4 * p) as *mut __m256i, value1);
        }

        for (res, prime_outputs) in [(res0, prime_outputs0), (res1, prime_outputs1)] {
            for coeff in 0..4 {
                for prime in 0..4 {
                    res[4 * coeff + prime] = prime_outputs[4 * prime + coeff];
                }
            }
        }
    }
}

// x2-block, two columns: two q120b × four q120c pairs → four q120b results

/// AVX2 x2-block inner product: two columns simultaneously.
///
/// Port of `q120x2_vec_mat2cols_product_bbc_avx2`.
///
/// Computes four q120b inner products (two x2-block rows × two matrix columns):
/// - `res[0..4]`   ← `Σᵢ x_a[i] · y_col0_a[i]`
/// - `res[4..8]`   ← `Σᵢ x_b[i] · y_col0_b[i]`
/// - `res[8..12]`  ← `Σᵢ x_a[i] · y_col1_a[i]`
/// - `res[12..16]` ← `Σᵢ x_b[i] · y_col1_b[i]`
///
/// - `x`: x2-block in u32 view — `ell` × 16 u32 (two `__m256i`s per step).
/// - `y`: two paired x2-block q120c columns — `ell` × 32 u32 (four `__m256i`s per step):
///   `[col0_a, col0_b, col1_a, col1_b]` per element.
/// - `res`: four q120b outputs — at least 16 u64.
///
/// # Safety
///
/// Caller must ensure AVX2 support. Slice lengths must satisfy
/// `x.len() >= 16 * ell`, `y.len() >= 32 * ell`, `res.len() >= 16`.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vec_mat2cols_product_x2_bbc_avx2(
    meta: &BbcMeta<Primes30>,
    ell: usize,
    res: &mut [u64],
    x: &[u32],
    y: &[u32],
) {
    unsafe {
        let mask32 = _mm256_set1_epi64x(u32::MAX as i64);

        // col 0, pair A
        let mut s0 = _mm256_setzero_si256();
        let mut s1 = _mm256_setzero_si256();
        // col 0, pair B
        let mut s2 = _mm256_setzero_si256();
        let mut s3 = _mm256_setzero_si256();
        // col 1, pair A
        let mut s4 = _mm256_setzero_si256();
        let mut s5 = _mm256_setzero_si256();
        // col 1, pair B
        let mut s6 = _mm256_setzero_si256();
        let mut s7 = _mm256_setzero_si256();

        let mut x_ptr = x.as_ptr() as *const __m256i;
        let mut y_ptr = y.as_ptr() as *const __m256i;

        for _ in 0..ell {
            // Load x pair
            let xa = _mm256_loadu_si256(x_ptr);
            let xa_hi = _mm256_srli_epi64::<32>(xa);
            let xb = _mm256_loadu_si256(x_ptr.add(1));
            let xb_hi = _mm256_srli_epi64::<32>(xb);

            // ── Column 0 ──
            // pair A: xa × y_col0_a
            let yc0a = _mm256_loadu_si256(y_ptr);
            let yc0a_hi = _mm256_srli_epi64::<32>(yc0a);
            let p0a_lo = _mm256_mul_epu32(xa, yc0a);
            let p0a_hi = _mm256_mul_epu32(xa_hi, yc0a_hi);
            s0 = _mm256_add_epi64(s0, _mm256_and_si256(p0a_lo, mask32));
            s0 = _mm256_add_epi64(s0, _mm256_and_si256(p0a_hi, mask32));
            s1 = _mm256_add_epi64(s1, _mm256_srli_epi64::<32>(p0a_lo));
            s1 = _mm256_add_epi64(s1, _mm256_srli_epi64::<32>(p0a_hi));

            // pair B: xb × y_col0_b
            let yc0b = _mm256_loadu_si256(y_ptr.add(1));
            let yc0b_hi = _mm256_srli_epi64::<32>(yc0b);
            let p0b_lo = _mm256_mul_epu32(xb, yc0b);
            let p0b_hi = _mm256_mul_epu32(xb_hi, yc0b_hi);
            s2 = _mm256_add_epi64(s2, _mm256_and_si256(p0b_lo, mask32));
            s2 = _mm256_add_epi64(s2, _mm256_and_si256(p0b_hi, mask32));
            s3 = _mm256_add_epi64(s3, _mm256_srli_epi64::<32>(p0b_lo));
            s3 = _mm256_add_epi64(s3, _mm256_srli_epi64::<32>(p0b_hi));

            // ── Column 1 ──
            // pair A: xa × y_col1_a
            let yc1a = _mm256_loadu_si256(y_ptr.add(2));
            let yc1a_hi = _mm256_srli_epi64::<32>(yc1a);
            let p1a_lo = _mm256_mul_epu32(xa, yc1a);
            let p1a_hi = _mm256_mul_epu32(xa_hi, yc1a_hi);
            s4 = _mm256_add_epi64(s4, _mm256_and_si256(p1a_lo, mask32));
            s4 = _mm256_add_epi64(s4, _mm256_and_si256(p1a_hi, mask32));
            s5 = _mm256_add_epi64(s5, _mm256_srli_epi64::<32>(p1a_lo));
            s5 = _mm256_add_epi64(s5, _mm256_srli_epi64::<32>(p1a_hi));

            // pair B: xb × y_col1_b
            let yc1b = _mm256_loadu_si256(y_ptr.add(3));
            let yc1b_hi = _mm256_srli_epi64::<32>(yc1b);
            let p1b_lo = _mm256_mul_epu32(xb, yc1b);
            let p1b_hi = _mm256_mul_epu32(xb_hi, yc1b_hi);
            s6 = _mm256_add_epi64(s6, _mm256_and_si256(p1b_lo, mask32));
            s6 = _mm256_add_epi64(s6, _mm256_and_si256(p1b_hi, mask32));
            s7 = _mm256_add_epi64(s7, _mm256_srli_epi64::<32>(p1b_lo));
            s7 = _mm256_add_epi64(s7, _mm256_srli_epi64::<32>(p1b_hi));

            x_ptr = x_ptr.add(2);
            y_ptr = y_ptr.add(4);
        }

        let mask_h2 = _mm256_set1_epi64x(((1u64 << meta.h) - 1) as i64);
        let s2l_pow_red = _mm256_loadu_si256(meta.s2l_pow_red.as_ptr() as *const __m256i);
        let s2h_pow_red = _mm256_loadu_si256(meta.s2h_pow_red.as_ptr() as *const __m256i);

        let res_ptr = res.as_mut_ptr() as *mut __m256i;
        _mm256_storeu_si256(res_ptr, reduce_bbc(s0, s1, mask_h2, meta.h, s2l_pow_red, s2h_pow_red));
        _mm256_storeu_si256(res_ptr.add(1), reduce_bbc(s2, s3, mask_h2, meta.h, s2l_pow_red, s2h_pow_red));
        _mm256_storeu_si256(res_ptr.add(2), reduce_bbc(s4, s5, mask_h2, meta.h, s2l_pow_red, s2h_pow_red));
        _mm256_storeu_si256(res_ptr.add(3), reduce_bbc(s6, s7, mask_h2, meta.h, s2l_pow_red, s2h_pow_red));
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(all(test, target_feature = "avx2"))]
mod tests {
    use super::*;
    use poulpy_cpu_portable::kernels::ntt4x30::{
        arithmetic::{b_from_znx64_portable, c_from_b_portable},
        mat_vec::{
            BbcMeta, vec_mat1col_product_bbc_portable, vec_mat1col_product_x2_bbc_portable, vec_mat2cols_product_x2_bbc_portable,
        },
        primes::Primes30,
    };

    /// Cast a q120b `[u64]` slice to `[u32]` for bbc functions.
    /// Each u64 limb fits in u32 (< 2^30), so upper 32 bits are 0.
    fn b_to_u32(b: &[u64]) -> Vec<u32> {
        b.iter().flat_map(|&v| [v as u32, (v >> 32) as u32]).collect()
    }

    /// Build a q120b slice (as u32) from small i64 coefficients.
    fn make_q120b_u32(ell: usize, n: usize, seed: i64) -> Vec<u32> {
        let coeffs: Vec<i64> = (0..ell * n).map(|i| (i as i64 * seed + 1) % 50 + 1).collect();
        let mut b = vec![0u64; 4 * ell * n];
        b_from_znx64_portable::<Primes30>(ell * n, &mut b, &coeffs);
        b_to_u32(&b)
    }

    /// Build a q120c slice (as u32) from a q120b u64 slice.
    fn make_q120c_u32(ell: usize, n: usize, seed: i64) -> Vec<u32> {
        let coeffs: Vec<i64> = (0..ell * n).map(|i| (i as i64 * seed + 2) % 50 + 1).collect();
        let mut b = vec![0u64; 4 * ell * n];
        b_from_znx64_portable::<Primes30>(ell * n, &mut b, &coeffs);
        let mut c = vec![0u32; 8 * ell * n];
        c_from_b_portable::<Primes30>(ell * n, &mut c, &b);
        c
    }

    /// AVX2 `vec_mat1col_product_bbc` matches reference (single column, single output).
    #[test]
    fn vec_mat1col_product_bbc_avx2_vs_ref() {
        let ell = 8usize;
        let n = 1usize; // one element per row
        let meta = BbcMeta::<Primes30>::new();

        let x = make_q120b_u32(ell, n, 7); // ell × 8 u32
        let y = make_q120c_u32(ell, n, 13); // ell × 8 u32

        let mut res_avx = vec![0u64; 4];
        let mut res_ref = vec![0u64; 4];

        unsafe { vec_mat1col_product_bbc_avx2(&meta, ell, &mut res_avx, &x, &y) };
        vec_mat1col_product_bbc_portable::<Primes30>(&meta, ell, &mut res_ref, &x, &y);

        assert_eq!(res_avx, res_ref, "vec_mat1col_product_bbc: AVX2 vs ref mismatch");
    }

    /// AVX2 `vec_mat1col_product_x2_bbc` matches reference (single column, two simultaneous outputs).
    #[test]
    fn vec_mat1col_product_x2_bbc_avx2_vs_ref() {
        let ell = 8usize;
        let n = 1usize;
        let meta = BbcMeta::<Primes30>::new();

        // x: 2 interleaved q120b (16 u32 per row)
        let x: Vec<u32> = {
            let a = make_q120b_u32(ell, n, 5);
            let b = make_q120b_u32(ell, n, 11);
            (0..ell)
                .flat_map(|i| a[8 * i..8 * i + 8].iter().chain(b[8 * i..8 * i + 8].iter()).cloned())
                .collect()
        };
        // y: 2 interleaved q120c (16 u32 per row)
        let y: Vec<u32> = {
            let a = make_q120c_u32(ell, n, 3);
            let b = make_q120c_u32(ell, n, 17);
            (0..ell)
                .flat_map(|i| a[8 * i..8 * i + 8].iter().chain(b[8 * i..8 * i + 8].iter()).cloned())
                .collect()
        };

        let mut res_avx = vec![0u64; 8];
        let mut res_ref = vec![0u64; 8];

        unsafe { vec_mat1col_product_x2_bbc_avx2(&meta, ell, &mut res_avx, &x, &y) };
        vec_mat1col_product_x2_bbc_portable::<Primes30>(&meta, ell, &mut res_ref, &x, &y);

        assert_eq!(res_avx, res_ref, "vec_mat1col_product_x2_bbc: AVX2 vs ref mismatch");
    }

    /// AVX2 `vec_mat2cols_product_x2_bbc` matches reference (two columns, two simultaneous outputs).
    #[test]
    fn vec_mat2cols_product_x2_bbc_avx2_vs_ref() {
        let ell = 8usize;
        let n = 1usize;
        let meta = BbcMeta::<Primes30>::new();

        // x: 2 interleaved q120b (16 u32 per row)
        let x: Vec<u32> = {
            let a = make_q120b_u32(ell, n, 7);
            let b = make_q120b_u32(ell, n, 19);
            (0..ell)
                .flat_map(|i| a[8 * i..8 * i + 8].iter().chain(b[8 * i..8 * i + 8].iter()).cloned())
                .collect()
        };
        // y: 4 interleaved q120c (32 u32 per row: col0_a, col0_b, col1_a, col1_b)
        let y: Vec<u32> = {
            let c0a = make_q120c_u32(ell, n, 2);
            let c0b = make_q120c_u32(ell, n, 9);
            let c1a = make_q120c_u32(ell, n, 23);
            let c1b = make_q120c_u32(ell, n, 31);
            (0..ell)
                .flat_map(|i| {
                    c0a[8 * i..8 * i + 8]
                        .iter()
                        .chain(c0b[8 * i..8 * i + 8].iter())
                        .chain(c1a[8 * i..8 * i + 8].iter())
                        .chain(c1b[8 * i..8 * i + 8].iter())
                        .cloned()
                })
                .collect()
        };

        let mut res_avx = vec![0u64; 16];
        let mut res_ref = vec![0u64; 16];

        unsafe { vec_mat2cols_product_x2_bbc_avx2(&meta, ell, &mut res_avx, &x, &y) };
        vec_mat2cols_product_x2_bbc_portable::<Primes30>(&meta, ell, &mut res_ref, &x, &y);

        assert_eq!(res_avx, res_ref, "vec_mat2cols_product_x2_bbc: AVX2 vs ref mismatch");
    }

    /// The prime-major kernels against `Σ x * y mod q` in exact arithmetic, on centered operands spanning
    /// `(-q/2, q/2]`, with more rows than an accumulator takes between two folds.
    #[test]
    fn vec_mat1col_product_blkpair_bbc_pm_avx2_vs_exact() {
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for ell in [1usize, 8, 16, 17, 40] {
            // Centered residue of prime p, per plane p, row and lane (lane order of the planes).
            let mut draw = |p: usize| {
                let q = Q_VEC[p] as i64;
                (next() % Q_VEC[p]) as i64 - q / 2
            };
            let x: Vec<[[i64; 4]; 4]> = (0..ell)
                .map(|_| std::array::from_fn(|p| std::array::from_fn(|_| draw(p))))
                .collect();
            let x1: Vec<[[i64; 4]; 4]> = (0..ell)
                .map(|_| std::array::from_fn(|p| std::array::from_fn(|_| draw(p))))
                .collect();
            let y: Vec<[[i64; 4]; 4]> = (0..ell)
                .map(|_| std::array::from_fn(|p| std::array::from_fn(|_| draw(p))))
                .collect();

            let plane = |v: &[[[i64; 4]; 4]]| -> Vec<u64> {
                let mut out = vec![0u64; 16 * ell];
                for (row, r) in v.iter().enumerate() {
                    for p in 0..4 {
                        for lane in 0..4 {
                            out[p * 4 * ell + row * 4 + lane] = r[p][lane] as u64;
                        }
                    }
                }
                out
            };
            let (x_pm, x1_pm) = (plane(&x), plane(&x1));
            let mut y_pm = vec![0u64; 8 * ell];
            for (row, r) in y.iter().enumerate() {
                for pair in 0..2 {
                    for lane in 0..4 {
                        // Montgomery form `y * 2^32 mod q`, centered, as the prepare kernel stores it.
                        let mont = |p: usize| {
                            let q = Q_VEC[p] as i128;
                            let m = ((r[p][lane] as i128) << 32).rem_euclid(q) as i64;
                            if m > (q / 2) as i64 { m - q as i64 } else { m }
                        };
                        y_pm[pair * 4 * ell + row * 4 + lane] =
                            mont(2 * pair) as u32 as u64 | ((mont(2 * pair + 1) as u32 as u64) << 32);
                    }
                }
            }

            // Output layout: res[4 * lane + p].
            let exact = |x: &[[[i64; 4]; 4]]| -> Vec<u64> {
                let mut out = vec![0u64; 16];
                for lane in 0..4 {
                    for p in 0..4 {
                        let sum: i128 = (0..ell).map(|row| x[row][p][lane] as i128 * y[row][p][lane] as i128).sum();
                        out[4 * lane + p] = sum.rem_euclid(Q_VEC[p] as i128) as u64;
                    }
                }
                out
            };
            let reduced = |res: &[u64]| -> Vec<u64> {
                res.iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        assert!(v < Q_VEC[i % 4] << 33, "ell={ell}: output {i} out of the q120b range");
                        v % Q_VEC[i % 4]
                    })
                    .collect()
            };

            let mut res = vec![0u64; 16];
            unsafe { vec_mat1col_product_blkpair_bbc_pm_avx2(ell, &mut res, &x_pm, &y_pm, 4 * ell) };
            assert_eq!(reduced(&res), exact(&x), "ell={ell}");

            let (mut res0, mut res1) = (vec![0u64; 16], vec![0u64; 16]);
            unsafe { vec_mat1col_product_blkpair_bbc_pm_x2_avx2(ell, &mut res0, &mut res1, &x_pm, &x1_pm, &y_pm, 4 * ell) };
            assert_eq!(reduced(&res0), exact(&x), "ell={ell}: x2, first");
            assert_eq!(reduced(&res1), exact(&x1), "ell={ell}: x2, second");
        }
    }
}
