use core::f64;
use std::collections::BTreeMap;

use std::fmt::Debug;
use tracing::debug;

type Vector<H> = BTreeMap<H, f64>;
type Matrix<H> = BTreeMap<H, BTreeMap<H, f64>>;

// a single trust computation
struct Trust<H: Debug + Default + Clone + Copy + Ord + PartialOrd + Eq + PartialEq> {
    // "C" matrix, peer: { what peer thinks of other peers }
    global: Matrix<H>,
    // initial local vector
    initial: Vector<H>,
    // local vector
    local: Vector<H>,
    // global max
    global_max: f64,
    // alpha value
    alpha: f64,
    // delta
    delta: f64,
    // epsilon
    epsilon: f64,
}

impl<H: Debug + Default + Clone + Copy + Ord + PartialOrd + Eq + PartialEq> Trust<H> {
    pub fn new(id: H, alpha_: f64, ptp: Vec<H>) -> Self {
        let mut t: Vector<H> = Vector::new();
        let mut t2: Vector<H> = Vector::new();
        let mut m: Matrix<H> = Matrix::new();

        // t(0) = p
        // p_i = 1/|P| if i ∈ P , and p_i = 0 otherwise.
        let inv_p = if ptp.len() != 0 {
            1.0f64 / ptp.len() as f64
        } else {
            0f64
        };

        for p in ptp.clone() {
            t.entry(p).or_insert(inv_p);
            t2.entry(p).or_insert(inv_p * alpha_);

            m.entry(id)
                .and_modify(|e| {
                    let _ = e.insert(p, inv_p);
                })
                .or_insert_with(|| {
                    let mut t: Vector<H> = Vector::new();

                    let _ = t.insert(p, inv_p);

                    t
                });
        }

        Trust {
            global: m,
            local: t.clone(),
            initial: t,
            global_max: 0f64,
            alpha: alpha_,
            delta: f64::MAX,
            epsilon: 0.0001f64,
        }
    }

    // add trust value to global matrix
    pub fn add(&mut self, i: H, j: H, score: f64) -> bool {
        // only accept normalized scores
        if score > 1.0f64 || score < 0.0f64 {
            return false;
        }

        self.global
            .entry(i)
            .and_modify(|e| {
                let _ = e.insert(j, score);
            })
            .or_insert_with(|| {
                let mut t: Vector<H> = BTreeMap::new();

                let _ = t.insert(j, score);

                t
            });

        self.global_max = self.global.iter().fold(0f64, |a, e| e.1.iter().fold(0.0f64, |a, e| a.max(*e.1)).max(a));
        
        true
    }

    fn n_score(&self, i: &H, j: &H) -> f64 {
        // if sum_j max(s_ij, 0) == 0, p_j
        if self.global_max == 0f64 {
            *self.initial.get(j).unwrap_or(&0f64)
        } else {
            // otherwise max(s_ij, 0) / `max`
            let mut acc = 0f64;
            if let Some(tvs) = self.global.get(i) {
                acc = tvs.get(j).unwrap_or(&0f64).max(0f64);
            }

            acc / self.global_max
        }
    }

    // t(k+1) = (1 − a)CT t(k) + ap
    pub fn iterate(&mut self) {
        let mut tk1: Vector<H> = Vector::new();

        // (c_i1*t_1) + (c_i2*t_2) + ... + (c_in+t_n)
        for (i, trusters) in &self.global {
            for (j, trust_value) in trusters {
                if i != j {
                    let t = self.n_score(i, j) * trust_value;
                    tk1.entry(*j).and_modify(|e| { *e += t; }).or_insert(t);
                }
            }
        }

        // (1 − a)CT t(k)
        tk1.iter_mut().for_each(|(en, e)| {
            *e += self.alpha * self.initial.get(en).unwrap_or(&0.0f64);
        });

        let mag = tk1.iter().fold(0.0f64, |a, e| a + e.1.powf(2.0f64)).sqrt();

        // normalize vector
        tk1.iter_mut().for_each(|(_, e)| {
            *e /= mag;
        });

        self.delta = tk1
            .iter()
            .map(|(row_idx, row)| {
                (row - self.local.get(row_idx).or(Some(&0f64)).unwrap_or(&0f64)).powf(2.0f64)
            })
            .sum::<f64>()
            .sqrt();

        self.local.clear();
        self.local = tk1;

        debug!("current delta: {}, epsilon: {}", self.delta, self.epsilon);
    }

    pub fn run(&mut self) {
        while self.delta > self.epsilon {
            self.iterate();
        }
    }

    pub fn get(&self, i: &H) -> f64 {
        *self.local.get(i).unwrap_or(&0.0f64)
    }
}

#[cfg(test)]
mod tests {
    use crate::U256;

    use super::*;
    use tracing_test::traced_test;

    #[traced_test]
    #[test]
    fn equal() {
        let mut trust: Trust<U256> = Trust::new(U256::from(1), 0.95, vec![]);

        trust.add(U256::from(1), U256::from(2), 1.0f64);
        trust.add(U256::from(1), U256::from(3), 1.0f64);
        trust.add(U256::from(2), U256::from(1), 1.0f64);
        trust.add(U256::from(2), U256::from(3), 1.0f64);
        trust.add(U256::from(3), U256::from(1), 1.0f64);
        trust.add(U256::from(3), U256::from(2), 1.0f64);

        trust.run();

        debug!("{:?}", trust.local);

        assert_eq!(trust.local.get(&U256::from(1)).unwrap(), trust.local.get(&U256::from(2)).unwrap());
        assert_eq!(trust.local.get(&U256::from(2)).unwrap(), trust.local.get(&U256::from(3)).unwrap());
        assert_eq!(trust.local.get(&U256::from(1)).unwrap(), trust.local.get(&U256::from(3)).unwrap());
    }

    #[traced_test]
    #[test]
    fn low_trust() {
        let mut trust: Trust<U256> = Trust::new(U256::from(1), 0.95, vec![]);

        trust.add(U256::from(1), U256::from(2), 1.0f64);
        trust.add(U256::from(1), U256::from(3), 0.1f64);
        trust.add(U256::from(2), U256::from(1), 1.0f64);
        trust.add(U256::from(2), U256::from(3), 0.1f64);
        trust.add(U256::from(3), U256::from(1), 1.0f64);
        trust.add(U256::from(3), U256::from(2), 1.0f64);

        trust.run();

        debug!("{:?}", trust.local);

        assert!(trust.local.get(&U256::from(1)).unwrap() == trust.local.get(&U256::from(2)).unwrap());
        assert!(trust.local.get(&U256::from(2)).unwrap() > trust.local.get(&U256::from(3)).unwrap());
        assert!(trust.local.get(&U256::from(1)).unwrap() > trust.local.get(&U256::from(3)).unwrap());
    }

    #[traced_test]
    #[test]
    fn single_pre_trusted() {
        let mut trust: Trust<U256> = Trust::new(U256::from(1), 0.95, vec![U256::from(1)]);

        trust.add(U256::from(1), U256::from(2), rand::random());
        trust.add(U256::from(1), U256::from(3), rand::random());
        trust.add(U256::from(2), U256::from(1), rand::random());
        trust.add(U256::from(2), U256::from(3), rand::random());
        trust.add(U256::from(3), U256::from(1), rand::random());
        trust.add(U256::from(3), U256::from(2), rand::random());

        trust.run();

        debug!("{:?}", trust.local);

        assert!(trust.local.get(&U256::from(1)).unwrap() > trust.local.get(&U256::from(2)).unwrap());
        assert!(trust.local.get(&U256::from(1)).unwrap() > trust.local.get(&U256::from(3)).unwrap());
    }
}
