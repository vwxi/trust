use std::collections::BTreeMap;

type Vector<H> = BTreeMap<H, f64>;
type Matrix<H> = BTreeMap<H, BTreeMap<H, f64>>;

struct Trust<H: Clone + Copy + Ord + PartialOrd + Eq + PartialEq> {
    // own id
    pub own_id: H,
    // list of pre-trusted peers
    pub pre_trusted_peers: Vec<H>,
    // "C" matrix, peer: { what peer thinks of other peers }
    pub global: Matrix<H>,
    // initial local vector
    pub local: Vector<H>,
    // alpha value
    pub alpha: f64,
}

impl<H: Clone + Copy + Ord + PartialOrd + Eq + PartialEq> Trust<H> {
    pub fn new(id: H, alpha_: f64, ptp: Vec<H>) -> Self {
        let mut t: Vector<H> = Vector::new();
        let mut m: Matrix<H> = Matrix::new();

        // t(0) = p
        // p_i = 1/|P| if i ∈ P , and p_i = 0 otherwise.
        let inv_p = 1.0f64 / ptp.len() as f64;
        for p in ptp.clone() {
            t.entry(p).or_insert(inv_p);
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
            local: t,
            pre_trusted_peers: ptp,
            alpha: alpha_,
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

    pub(crate) fn dot_mat_vec(mat: &BTreeMap<H, Vector<H>>, vec: &Vector<H>) -> Vector<H> {
        mat.iter()
            .filter_map(|row| {
                let sum = row.1
                    .iter()
                    .filter_map(|(&idx, &global_score)| {
                        vec.get(&idx).map(|local_score| global_score * local_score)
                    })
                    .sum::<f64>();

                if sum != 0.0 {
                    Some((*row.0, sum))
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::U256;

    use super::*;
    use tracing_test::traced_test;

    #[traced_test]
    #[test]
    fn dot_mat_vec() {
        {
            let matrix: Matrix<U256> = Matrix::from([
                (
                    U256::from(0),
                    Vector::from([
                        (U256::from(0), 1.0),
                        (U256::from(1), 2.0),
                        (U256::from(2), 3.0),
                    ]),
                ),
                (
                    U256::from(1),
                    Vector::from([
                        (U256::from(0), 4.0),
                        (U256::from(1), 5.0),
                        (U256::from(2), 6.0),
                    ]),
                ),
                (
                    U256::from(2),
                    Vector::from([
                        (U256::from(0), 7.0),
                        (U256::from(1), 8.0),
                        (U256::from(2), 9.0),
                    ]),
                ),
            ]);

            let vector: Vector<U256> = Vector::from([
                (U256::from(0), 1.0),
                (U256::from(1), 0.5),
                (U256::from(2), -1.0),
            ]);

            let result = Trust::dot_mat_vec(&matrix, &vector);

            assert!(result
                .iter()
                .zip([-1.0f64, 0.5f64, 2.0f64].iter())
                .all(|(x, y)| x.1 == y));
        }
    }
}
