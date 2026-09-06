//! Deterministic pseudo-random generation.
//!
//! FR-RT-010: one global generator seeded by `--seed`, default 42.
//! FR-AI-014: identical seed ⇒ bit-identical parameters, so the algorithm is
//! fixed here and versioned with the language, never delegated to the host.

/// PCG-XSH-RR 64/32. Chosen because it is small, has a documented stream
/// structure, and produces identical output on every platform.
#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
    inc: u64,
    /// Cached second variate from the Box-Muller pair.
    spare: Option<f32>,
}

const MULT: u64 = 6_364_136_223_846_793_005;

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut r = Rng { state: 0, inc: (seed << 1) | 1, spare: None };
        r.next_u32();
        r.state = r.state.wrapping_add(seed);
        r.next_u32();
        r
    }

    /// A named independent stream, so shuffling never perturbs init.
    pub fn stream(seed: u64, stream_id: u64) -> Rng {
        let mut r = Rng {
            state: 0,
            inc: (stream_id << 1) | 1,
            spare: None,
        };
        r.next_u32();
        r.state = r.state.wrapping_add(seed);
        r.next_u32();
        r
    }

    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(MULT).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    pub fn next_u64(&mut self) -> u64 {
        ((self.next_u32() as u64) << 32) | self.next_u32() as u64
    }

    /// Uniform in [0, 1). 24 bits of mantissa, so every value is exact.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / 16_777_216.0)
    }

    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }

    /// Box-Muller, caching the second variate so the stream stays deterministic.
    pub fn normal(&mut self, mean: f32, std: f32) -> f32 {
        if let Some(s) = self.spare.take() {
            return mean + std * s;
        }
        let mut u1 = self.next_f32();
        if u1 < 1e-7 {
            u1 = 1e-7;
        }
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        self.spare = Some(r * theta.sin());
        mean + std * r * theta.cos()
    }

    /// Unbiased integer in [0, n) by rejection sampling.
    pub fn below(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        let threshold = (u32::MAX - n + 1) % n;
        loop {
            let r = self.next_u32();
            if r >= threshold {
                return r % n;
            }
        }
    }

    /// Fisher-Yates. Deterministic given the seed (FR-AI-014).
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        if items.len() < 2 {
            return;
        }
        for i in (1..items.len()).rev() {
            let j = self.below((i + 1) as u32) as usize;
            items.swap(i, j);
        }
    }

    /// Bernoulli draw, used by Dropout (FR-AI-019).
    pub fn bernoulli(&mut self, p: f32) -> bool {
        self.next_f32() < p
    }
}

impl Default for Rng {
    fn default() -> Self {
        Rng::new(42)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn different_seed_diverges() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        let da: Vec<u32> = (0..16).map(|_| a.next_u32()).collect();
        let db: Vec<u32> = (0..16).map(|_| b.next_u32()).collect();
        assert_ne!(da, db);
    }

    #[test]
    fn uniform_in_range() {
        let mut r = Rng::new(7);
        for _ in 0..10_000 {
            let v = r.next_f32();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn below_is_bounded() {
        let mut r = Rng::new(9);
        for _ in 0..10_000 {
            assert!(r.below(10) < 10);
        }
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut r = Rng::new(3);
        let mut v: Vec<usize> = (0..100).collect();
        r.shuffle(&mut v);
        let mut s = v.clone();
        s.sort();
        assert_eq!(s, (0..100).collect::<Vec<_>>());
        assert_ne!(v, s); // vanishingly unlikely to be identity
    }
}
