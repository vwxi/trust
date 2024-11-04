use core::f64;
use std::collections::BTreeMap;

use std::fmt::Debug;
use tracing::debug;

type Vector<H> = BTreeMap<H, f64>;
type Matrix<H> = BTreeMap<H, BTreeMap<H, f64>>;

// a single trust computation
struct Trust<H: Debug + Clone + Copy + Ord + PartialOrd + Eq + PartialEq> {
    // own id
    pub own_id: H,
    // list of pre-trusted peers
    pub pre_trusted_peers: Vec<H>,
    // "C" matrix, peer: { what peer thinks of other peers }
    pub global: Matrix<H>,
    // initial local vector applied with alpha
    alpha_local: Vector<H>,
    // local vector
    pub local: Vector<H>,
    // alpha value
    pub alpha: f64,
    // delta
    delta: f64,
    // epsilon
    epsilon: f64
}

impl<H: Debug + Clone + Copy + Ord + PartialOrd + Eq + PartialEq> Trust<H> {
    pub fn new(id: H, alpha_: f64, ptp: Vec<H>) -> Self {
        let mut t: Vector<H> = Vector::new();
        let mut t2: Vector<H> = Vector::new();
        let mut m: Matrix<H> = Matrix::new();

        // t(0) = p
        // p_i = 1/|P| if i ∈ P , and p_i = 0 otherwise.
        let inv_p = 1.0f64 / ptp.len() as f64;
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
            own_id: id,
            global: m,
            local: t.clone(),
            alpha_local: t,
            pre_trusted_peers: ptp,
            alpha: alpha_,
            delta: f64::MAX,
            epsilon: 0.0001f64
        }
    }

    // add trust value to global matrix and maybe local trust vector
    pub fn add(&mut self, i: H, j: H, score: f64) -> bool {
        // only accept normalized scores
        if score > 1.0f64 || score < 0.0f64 {
            return false;
        }

        if i == self.own_id {
            debug!("adding to own entry: {:?} -> {:?} value {}", i, j, score);
            self.local.entry(j).and_modify(|e| { *e = score; }).or_insert(score);
        }

        debug!("adding to global entry: {:?} -> {:?} value {}", i, j, score);

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

        true
    }

    pub fn score(&self, i: H, j: H) -> f64 {
        // If peer i doesn’t download from anybody else,
        // or if it assigns a zero score to all other peers, cij from Equation 1
        // will be undefined. In this case, we set cij = pj .
        if self
            .global
            .iter()
            .map(|e| 0f64.max(if let Some(v) = e.1.get(&j) { *v } else { 0f64 }))
            .fold(0f64, |a, e| a + e)
            != 0f64
        {
            if let Some(ii) = self.global.get(&i) {
                // either from global or pj
                *ii.get(&j).or(self.local.get(&j)).unwrap_or(&0f64)
            } else {
                0f64
            }
        } else {
            0f64
        }
    }

    // t(k+1) = (1 − a)CT t(k) + ap 
    pub fn iterate(&mut self) {
        let tk1: Vector<H> = self.global.iter()
            .filter_map(|(row_idx, row)| {
                let sum = row
                    .iter()
                    .filter_map(|(&col_idx, _)| {
                        self.local.get(&col_idx)
                            .map(|local_score| dbg!(self.score(*row_idx, col_idx)) * dbg!(local_score))
                    })
                    .sum::<f64>() * (1.0f64 - self.alpha) + self.alpha_local.get(row_idx).or(Some(&0f64)).unwrap();

                if sum != 0.0 {
                    Some((*row_idx, sum))
                } else {
                    None
                }
            })
            .collect();

        self.delta = tk1.iter().filter_map(|(row_idx, row)| {
            Some((row - self.local.get(row_idx).or(Some(&0f64)).unwrap_or(&0f64)).powf(2.0f64))
        }).sum::<f64>().sqrt();

        self.local.clear();
        self.local = tk1;

        debug!("current delta: {}, epsilon: {}", self.delta, self.epsilon);
    }

    pub fn run(&mut self) {
        while self.delta > self.epsilon {
            self.iterate();
        }

        dbg!(&self.local);
    }
}

#[cfg(test)]
mod tests {
    use crate::U256;

    use super::*;
    use tracing_test::traced_test;

    #[traced_test]
    #[test]
    fn trust1() {
        let mut trust: Trust<U256> = Trust::new(U256::from(1), 0.001, vec![]);
        
        // A
        trust.add(U256::from(1), U256::from(2), 0.2);
        trust.add(U256::from(1), U256::from(3), 0.3);
        trust.add(U256::from(1), U256::from(4), 0.1);

        // B
        trust.add(U256::from(2), U256::from(3), 0.2);
        trust.add(U256::from(2), U256::from(4), 0.2);

        // C
        trust.add(U256::from(3), U256::from(1), 0.4);

        // D
        trust.add(U256::from(4), U256::from(2), 1.0);
    
        
        trust.run();
    }
}
