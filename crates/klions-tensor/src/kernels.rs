//! Tensor compute kernels — FR-TEN-004 … FR-TEN-012.
//!
//! Every kernel is first-party (DC-007). Hot paths operate on contiguous
//! slices; strided inputs are materialized once at the boundary rather than
//! paying index arithmetic per element.

use crate::{
    broadcast_shapes, check_alloc, contiguous_strides, first_bad_axis, Tensor, TensorError,
    TensorResult,
};

// ============================ element-wise ============================

/// FR-TEN-004 with FR-TYP-007 broadcasting.
pub fn binary_op(
    op_name: &str,
    a: &Tensor,
    b: &Tensor,
    f: impl Fn(f32, f32) -> f32,
) -> TensorResult<Tensor> {
    let shape = broadcast_shapes(&a.shape, &b.shape).ok_or_else(|| TensorError::ShapeMismatch {
        op: op_name.to_string(),
        lhs: a.shape.clone(),
        rhs: b.shape.clone(),
        axis: first_bad_axis(&a.shape, &b.shape),
    })?;
    let n = check_alloc(&shape)?;

    // Fast path: identical shapes, both contiguous.
    if a.shape == b.shape && a.is_contiguous() && b.is_contiguous() {
        let ab = a.buf.borrow();
        let bb = b.buf.borrow();
        let av = &ab[a.offset..a.offset + n];
        let bv = &bb[b.offset..b.offset + n];
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            out.push(f(av[i], bv[i]));
        }
        return Tensor::from_vec(out, shape);
    }

    // General path: expand both operands into the broadcast frame.
    let av = a.to_vec();
    let bv = b.to_vec();
    let amap = BroadcastMap::new(&a.shape, &shape);
    let bmap = BroadcastMap::new(&b.shape, &shape);
    let mut out = Vec::with_capacity(n);
    let mut idx = vec![0usize; shape.len()];
    for _ in 0..n {
        out.push(f(av[amap.index(&idx)], bv[bmap.index(&idx)]));
        crate::bump_index(&mut idx, &shape);
    }
    Tensor::from_vec(out, shape)
}

pub fn unary_op(a: &Tensor, f: impl Fn(f32) -> f32) -> TensorResult<Tensor> {
    let v: Vec<f32> = a.to_vec().into_iter().map(f).collect();
    Tensor::from_vec(v, a.shape.clone())
}

pub fn add(a: &Tensor, b: &Tensor) -> TensorResult<Tensor> {
    binary_op("+", a, b, |x, y| x + y)
}
pub fn sub(a: &Tensor, b: &Tensor) -> TensorResult<Tensor> {
    binary_op("-", a, b, |x, y| x - y)
}
pub fn mul(a: &Tensor, b: &Tensor) -> TensorResult<Tensor> {
    binary_op("*", a, b, |x, y| x * y)
}
pub fn div(a: &Tensor, b: &Tensor) -> TensorResult<Tensor> {
    // FR-RT-006: float division by zero yields IEEE inf/NaN, not a panic.
    binary_op("/", a, b, |x, y| x / y)
}
pub fn pow(a: &Tensor, b: &Tensor) -> TensorResult<Tensor> {
    binary_op("pow", a, b, |x, y| x.powf(y))
}
pub fn maximum(a: &Tensor, b: &Tensor) -> TensorResult<Tensor> {
    binary_op("max", a, b, |x, y| x.max(y))
}
pub fn minimum(a: &Tensor, b: &Tensor) -> TensorResult<Tensor> {
    binary_op("min", a, b, |x, y| x.min(y))
}

pub fn neg(a: &Tensor) -> TensorResult<Tensor> {
    unary_op(a, |x| -x)
}
pub fn exp(a: &Tensor) -> TensorResult<Tensor> {
    unary_op(a, f32::exp)
}
pub fn ln(a: &Tensor) -> TensorResult<Tensor> {
    unary_op(a, f32::ln)
}
pub fn sqrt(a: &Tensor) -> TensorResult<Tensor> {
    unary_op(a, f32::sqrt)
}
pub fn abs(a: &Tensor) -> TensorResult<Tensor> {
    unary_op(a, f32::abs)
}
pub fn tanh(a: &Tensor) -> TensorResult<Tensor> {
    unary_op(a, f32::tanh)
}
pub fn relu(a: &Tensor) -> TensorResult<Tensor> {
    unary_op(a, |x| if x > 0.0 { x } else { 0.0 })
}
pub fn sigmoid(a: &Tensor) -> TensorResult<Tensor> {
    unary_op(a, |x| {
        // Numerically stable on both tails.
        if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let e = x.exp();
            e / (1.0 + e)
        }
    })
}
pub fn scalar_mul(a: &Tensor, k: f32) -> TensorResult<Tensor> {
    unary_op(a, |x| x * k)
}
pub fn add_scalar(a: &Tensor, k: f32) -> TensorResult<Tensor> {
    unary_op(a, |x| x + k)
}

/// Maps a logical index in the broadcast frame back to a source offset.
pub struct BroadcastMap {
    strides: Vec<usize>,
    lead: usize,
}

impl BroadcastMap {
    pub fn new(src: &[usize], dst: &[usize]) -> BroadcastMap {
        let lead = dst.len() - src.len();
        let base = contiguous_strides(src);
        let mut strides = vec![0usize; dst.len()];
        for i in 0..src.len() {
            // A source dimension of 1 contributes stride 0: it is reused.
            strides[lead + i] = if src[i] == 1 && dst[lead + i] != 1 { 0 } else { base[i] };
        }
        BroadcastMap { strides, lead }
    }
    #[inline]
    pub fn index(&self, idx: &[usize]) -> usize {
        let mut off = 0;
        for k in self.lead..idx.len() {
            off += idx[k] * self.strides[k];
        }
        off
    }
}

/// Reverse of broadcasting: sum `grad` back down to `shape`.
/// Needed by every backward rule whose forward broadcast (FR-AI-003).
pub fn unbroadcast(grad: &Tensor, shape: &[usize]) -> TensorResult<Tensor> {
    if grad.shape == shape {
        return Ok(grad.clone());
    }
    let mut g = grad.clone();
    // Collapse leading axes the target does not have.
    while g.rank() > shape.len() {
        g = sum_axis(&g, 0, false)?;
    }
    // Collapse axes the target has as 1.
    for (i, &d) in shape.iter().enumerate() {
        if d == 1 && g.dim(i) != 1 {
            g = sum_axis(&g, i, true)?;
        }
    }
    if g.shape != shape {
        g = g.reshape(&shape.iter().map(|&d| d as isize).collect::<Vec<_>>())?;
    }
    Ok(g)
}

// ============================ matmul ============================

/// FR-TEN-005 / FR-TEN-009: cache-blocked matrix multiplication for rank-2
/// operands, batched over the leading axis for rank-3.
pub fn matmul(a: &Tensor, b: &Tensor) -> TensorResult<Tensor> {
    match (a.rank(), b.rank()) {
        (2, 2) => matmul2(a, b),
        (3, 3) => {
            if a.dim(0) != b.dim(0) {
                return Err(TensorError::MatMulShape {
                    lhs: a.shape.clone(),
                    rhs: b.shape.clone(),
                });
            }
            let batch = a.dim(0);
            let mut parts = Vec::with_capacity(batch);
            for i in 0..batch {
                parts.push(matmul2(&a.row(i)?, &b.row(i)?)?);
            }
            stack(&parts)
        }
        (3, 2) => {
            // Common case: a batch of matrices times one weight matrix.
            let batch = a.dim(0);
            let m = a.dim(1);
            let flat = a.reshape(&[(batch * m) as isize, a.dim(2) as isize])?;
            let out = matmul2(&flat, b)?;
            out.reshape(&[batch as isize, m as isize, b.dim(1) as isize])
        }
        _ => Err(TensorError::MatMulShape {
            lhs: a.shape.clone(),
            rhs: b.shape.clone(),
        }),
    }
}

const BLOCK_M: usize = 64;
const BLOCK_N: usize = 64;
const BLOCK_K: usize = 128;

fn matmul2(a: &Tensor, b: &Tensor) -> TensorResult<Tensor> {
    let (m, k) = (a.dim(0), a.dim(1));
    let (k2, n) = (b.dim(0), b.dim(1));
    if k != k2 {
        return Err(TensorError::MatMulShape {
            lhs: a.shape.clone(),
            rhs: b.shape.clone(),
        });
    }
    check_alloc(&[m, n])?;
    let av = a.to_vec();
    let bv = b.to_vec();
    let mut c = vec![0.0f32; m * n];

    // Blocked i-k-j: the inner loop walks B and C contiguously, which is what
    // makes the scalar version competitive without intrinsics.
    for i0 in (0..m).step_by(BLOCK_M) {
        let i1 = (i0 + BLOCK_M).min(m);
        for k0 in (0..k).step_by(BLOCK_K) {
            let k1 = (k0 + BLOCK_K).min(k);
            for j0 in (0..n).step_by(BLOCK_N) {
                let j1 = (j0 + BLOCK_N).min(n);
                for i in i0..i1 {
                    let a_row = i * k;
                    let c_row = i * n;
                    for kk in k0..k1 {
                        let aik = av[a_row + kk];
                        if aik == 0.0 {
                            continue;
                        }
                        let b_row = kk * n;
                        let cs = &mut c[c_row + j0..c_row + j1];
                        let bs = &bv[b_row + j0..b_row + j1];
                        // Unrolled by 4: measurably faster than the naive loop
                        // and stays free of architecture-specific intrinsics
                        // so the fallback path holds on aarch64 (NFR-Q-013).
                        let chunks = cs.len() / 4;
                        for q in 0..chunks {
                            let o = q * 4;
                            cs[o] += aik * bs[o];
                            cs[o + 1] += aik * bs[o + 1];
                            cs[o + 2] += aik * bs[o + 2];
                            cs[o + 3] += aik * bs[o + 3];
                        }
                        for o in chunks * 4..cs.len() {
                            cs[o] += aik * bs[o];
                        }
                    }
                }
            }
        }
    }
    Tensor::from_vec(c, vec![m, n])
}

/// Stack tensors of identical shape along a new leading axis.
pub fn stack(parts: &[Tensor]) -> TensorResult<Tensor> {
    if parts.is_empty() {
        return Err(TensorError::BadArgument("cannot stack zero tensors".into()));
    }
    let inner = parts[0].shape.clone();
    for (i, p) in parts.iter().enumerate() {
        if p.shape != inner {
            return Err(TensorError::ShapeMismatch {
                op: format!("stack (element {})", i),
                lhs: inner.clone(),
                rhs: p.shape.clone(),
                axis: first_bad_axis(&inner, &p.shape),
            });
        }
    }
    let mut shape = vec![parts.len()];
    shape.extend_from_slice(&inner);
    check_alloc(&shape)?;
    let mut data = Vec::with_capacity(shape.iter().product());
    for p in parts {
        data.extend_from_slice(&p.to_vec());
    }
    Tensor::from_vec(data, shape)
}

/// Concatenate along an existing axis.
pub fn concat(parts: &[Tensor], axis: usize) -> TensorResult<Tensor> {
    if parts.is_empty() {
        return Err(TensorError::BadArgument("cannot concatenate zero tensors".into()));
    }
    let rank = parts[0].rank();
    if axis >= rank {
        return Err(TensorError::BadAxis { axis, rank });
    }
    let mut shape = parts[0].shape.clone();
    shape[axis] = parts.iter().map(|p| p.dim(axis)).sum();
    check_alloc(&shape)?;
    let outer: usize = shape[..axis].iter().product();
    let mut out = vec![0.0f32; shape.iter().product::<usize>()];
    let out_inner: usize = shape[axis..].iter().product();
    let mut written = 0usize;
    for p in parts {
        let pv = p.to_vec();
        let p_inner: usize = p.shape[axis..].iter().product();
        for o in 0..outer {
            let src = o * p_inner;
            let dst = o * out_inner + written;
            out[dst..dst + p_inner].copy_from_slice(&pv[src..src + p_inner]);
        }
        written += p_inner;
    }
    Tensor::from_vec(out, shape)
}

// ============================ reductions ============================

/// FR-TEN-006 with an optional axis and `keepdims`.
pub fn sum_all(a: &Tensor) -> TensorResult<Tensor> {
    // Pairwise summation: error grows as O(log n) rather than O(n).
    let v = a.to_vec();
    Ok(Tensor::scalar(pairwise_sum(&v)))
}

fn pairwise_sum(v: &[f32]) -> f32 {
    const CUTOFF: usize = 128;
    if v.len() <= CUTOFF {
        let mut s = 0.0f32;
        for &x in v {
            s += x;
        }
        return s;
    }
    let mid = v.len() / 2;
    pairwise_sum(&v[..mid]) + pairwise_sum(&v[mid..])
}

pub fn mean_all(a: &Tensor) -> TensorResult<Tensor> {
    let n = a.numel().max(1) as f32;
    Ok(Tensor::scalar(pairwise_sum(&a.to_vec()) / n))
}

pub fn max_all(a: &Tensor) -> TensorResult<Tensor> {
    let v = a.to_vec();
    Ok(Tensor::scalar(v.iter().copied().fold(f32::NEG_INFINITY, f32::max)))
}

pub fn min_all(a: &Tensor) -> TensorResult<Tensor> {
    let v = a.to_vec();
    Ok(Tensor::scalar(v.iter().copied().fold(f32::INFINITY, f32::min)))
}

pub fn argmax_all(a: &Tensor) -> TensorResult<Tensor> {
    let v = a.to_vec();
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    Ok(Tensor::scalar(best as f32))
}

fn reduce_axis(
    a: &Tensor,
    axis: usize,
    keepdims: bool,
    init: f32,
    f: impl Fn(f32, f32) -> f32,
    finish: impl Fn(f32, usize) -> f32,
) -> TensorResult<Tensor> {
    let rank = a.rank();
    if axis >= rank {
        return Err(TensorError::BadAxis { axis, rank });
    }
    let av = a.to_vec();
    let outer: usize = a.shape[..axis].iter().product();
    let len = a.shape[axis];
    let inner: usize = a.shape[axis + 1..].iter().product();

    let mut shape: Vec<usize> = a.shape.clone();
    if keepdims {
        shape[axis] = 1;
    } else {
        shape.remove(axis);
    }
    let mut out = vec![init; outer * inner];
    for o in 0..outer {
        for l in 0..len {
            let base = o * len * inner + l * inner;
            for i in 0..inner {
                let dst = o * inner + i;
                out[dst] = f(out[dst], av[base + i]);
            }
        }
    }
    for v in out.iter_mut() {
        *v = finish(*v, len);
    }
    Tensor::from_vec(out, shape)
}

pub fn sum_axis(a: &Tensor, axis: usize, keepdims: bool) -> TensorResult<Tensor> {
    reduce_axis(a, axis, keepdims, 0.0, |acc, x| acc + x, |v, _| v)
}
pub fn mean_axis(a: &Tensor, axis: usize, keepdims: bool) -> TensorResult<Tensor> {
    reduce_axis(a, axis, keepdims, 0.0, |acc, x| acc + x, |v, n| v / n as f32)
}
pub fn max_axis(a: &Tensor, axis: usize, keepdims: bool) -> TensorResult<Tensor> {
    reduce_axis(a, axis, keepdims, f32::NEG_INFINITY, f32::max, |v, _| v)
}
pub fn min_axis(a: &Tensor, axis: usize, keepdims: bool) -> TensorResult<Tensor> {
    reduce_axis(a, axis, keepdims, f32::INFINITY, f32::min, |v, _| v)
}

pub fn argmax_axis(a: &Tensor, axis: usize, keepdims: bool) -> TensorResult<Tensor> {
    let rank = a.rank();
    if axis >= rank {
        return Err(TensorError::BadAxis { axis, rank });
    }
    let av = a.to_vec();
    let outer: usize = a.shape[..axis].iter().product();
    let len = a.shape[axis];
    let inner: usize = a.shape[axis + 1..].iter().product();
    let mut shape = a.shape.clone();
    if keepdims {
        shape[axis] = 1;
    } else {
        shape.remove(axis);
    }
    let mut best = vec![f32::NEG_INFINITY; outer * inner];
    let mut arg = vec![0.0f32; outer * inner];
    for o in 0..outer {
        for l in 0..len {
            let base = o * len * inner + l * inner;
            for i in 0..inner {
                let dst = o * inner + i;
                let v = av[base + i];
                if v > best[dst] {
                    best[dst] = v;
                    arg[dst] = l as f32;
                }
            }
        }
    }
    Tensor::from_vec(arg, shape)
}

// ============================ NN primitives ============================

/// Numerically stable softmax along the last axis.
pub fn softmax_last(a: &Tensor) -> TensorResult<Tensor> {
    let rank = a.rank();
    if rank == 0 {
        return Ok(Tensor::scalar(1.0));
    }
    let last = rank - 1;
    let n = a.dim(last);
    let av = a.to_vec();
    let rows = av.len() / n;
    let mut out = vec![0.0f32; av.len()];
    for r in 0..rows {
        let s = r * n;
        let mut m = f32::NEG_INFINITY;
        for i in 0..n {
            m = m.max(av[s + i]);
        }
        let mut total = 0.0f32;
        for i in 0..n {
            let e = (av[s + i] - m).exp();
            out[s + i] = e;
            total += e;
        }
        let inv = 1.0 / total;
        for i in 0..n {
            out[s + i] *= inv;
        }
    }
    Tensor::from_vec(out, a.shape.clone())
}

/// log-softmax along the last axis, via log-sum-exp (FR-AI-009).
pub fn log_softmax_last(a: &Tensor) -> TensorResult<Tensor> {
    let rank = a.rank();
    if rank == 0 {
        return Ok(Tensor::scalar(0.0));
    }
    let n = a.dim(rank - 1);
    let av = a.to_vec();
    let rows = av.len() / n;
    let mut out = vec![0.0f32; av.len()];
    for r in 0..rows {
        let s = r * n;
        let mut m = f32::NEG_INFINITY;
        for i in 0..n {
            m = m.max(av[s + i]);
        }
        let mut total = 0.0f32;
        for i in 0..n {
            total += (av[s + i] - m).exp();
        }
        let lse = m + total.ln();
        for i in 0..n {
            out[s + i] = av[s + i] - lse;
        }
    }
    Tensor::from_vec(out, a.shape.clone())
}

/// FR-AI-012: one-hot encoding of an index tensor.
pub fn one_hot(labels: &Tensor, classes: usize) -> TensorResult<Tensor> {
    let lv = labels.to_vec();
    let n = lv.len();
    check_alloc(&[n, classes])?;
    let mut out = vec![0.0f32; n * classes];
    for (i, &l) in lv.iter().enumerate() {
        let c = l as usize;
        if c >= classes {
            return Err(TensorError::IndexOutOfBounds {
                axis: 1,
                index: c as isize,
                bound: classes,
            });
        }
        out[i * classes + c] = 1.0;
    }
    let mut shape = labels.shape.clone();
    shape.push(classes);
    Tensor::from_vec(out, shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(d: Vec<f32>, s: Vec<usize>) -> Tensor {
        Tensor::from_vec(d, s).unwrap()
    }

    #[test]
    fn elementwise_same_shape() {
        let a = t(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = t(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
        assert_eq!(add(&a, &b).unwrap().to_vec(), vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn broadcasting_row_vector() {
        let a = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = t(vec![10.0, 20.0, 30.0], vec![3]);
        assert_eq!(
            add(&a, &b).unwrap().to_vec(),
            vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]
        );
    }

    #[test]
    fn broadcast_failure_names_axis() {
        let a = t(vec![0.0; 6], vec![2, 3]);
        let b = t(vec![0.0; 8], vec![2, 4]);
        match add(&a, &b) {
            Err(TensorError::ShapeMismatch { axis, .. }) => assert_eq!(axis, Some(1)),
            other => panic!("expected a shape mismatch, got {:?}", other.map(|t| t.shape)),
        }
    }

    #[test]
    fn matmul_matches_hand_computation() {
        let a = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = t(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
        // [[58, 64], [139, 154]]
        assert_eq!(matmul(&a, &b).unwrap().to_vec(), vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn matmul_larger_agrees_with_naive() {
        let (m, k, n) = (37usize, 53usize, 41usize);
        let av: Vec<f32> = (0..m * k).map(|i| ((i * 7 % 13) as f32) - 6.0).collect();
        let bv: Vec<f32> = (0..k * n).map(|i| ((i * 5 % 11) as f32) - 5.0).collect();
        let a = t(av.clone(), vec![m, k]);
        let b = t(bv.clone(), vec![k, n]);
        let got = matmul(&a, &b).unwrap().to_vec();
        let mut want = vec![0.0f32; m * n];
        for i in 0..m {
            for p in 0..k {
                for j in 0..n {
                    want[i * n + j] += av[i * k + p] * bv[p * n + j];
                }
            }
        }
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-3, "{} vs {}", g, w);
        }
    }

    #[test]
    fn transpose_then_contiguous_round_trips() {
        let a = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let at = a.t().unwrap();
        assert_eq!(at.shape, vec![3, 2]);
        assert_eq!(at.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(at.t().unwrap().to_vec(), a.to_vec());
    }

    #[test]
    fn reshape_infers_wildcard() {
        let a = t((0..12).map(|i| i as f32).collect(), vec![12]);
        assert_eq!(a.reshape(&[3, -1]).unwrap().shape, vec![3, 4]);
        assert!(a.reshape(&[5, -1]).is_err());
    }

    #[test]
    fn reductions_over_axis() {
        let a = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        assert_eq!(sum_axis(&a, 0, false).unwrap().to_vec(), vec![5.0, 7.0, 9.0]);
        assert_eq!(sum_axis(&a, 1, false).unwrap().to_vec(), vec![6.0, 15.0]);
        assert_eq!(sum_axis(&a, 1, true).unwrap().shape, vec![2, 1]);
        assert_eq!(mean_axis(&a, 1, false).unwrap().to_vec(), vec![2.0, 5.0]);
        assert_eq!(argmax_axis(&a, 1, false).unwrap().to_vec(), vec![2.0, 2.0]);
    }

    #[test]
    fn softmax_rows_sum_to_one() {
        let a = t(vec![1.0, 2.0, 3.0, 1000.0, 1000.0, 1000.0], vec![2, 3]);
        let s = softmax_last(&a).unwrap().to_vec();
        assert!((s[0] + s[1] + s[2] - 1.0).abs() < 1e-6);
        // Stability: the large row must not produce NaN.
        assert!((s[3] + s[4] + s[5] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn unbroadcast_sums_back_down() {
        let g = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        assert_eq!(unbroadcast(&g, &[3]).unwrap().to_vec(), vec![5.0, 7.0, 9.0]);
        assert_eq!(unbroadcast(&g, &[1, 3]).unwrap().shape, vec![1, 3]);
    }

    #[test]
    fn index_out_of_bounds_is_reported() {
        let a = t(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        match a.get(&[0, 5]) {
            Err(TensorError::IndexOutOfBounds { axis, index, bound }) => {
                assert_eq!((axis, index, bound), (1, 5, 2));
            }
            _ => panic!("expected an IndexError"),
        }
    }

    #[test]
    fn slice_is_a_view_sharing_the_buffer() {
        let a = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let s = a.slice(0, 1, 3).unwrap();
        assert_eq!(s.shape, vec![2, 2]);
        assert_eq!(s.to_vec(), vec![3.0, 4.0, 5.0, 6.0]);
        s.set(&[0, 0], 99.0).unwrap();
        assert_eq!(a.get(&[1, 0]).unwrap(), 99.0);
    }
}
