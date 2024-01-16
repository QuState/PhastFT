use std::simd::f64x4;

use rayon::prelude::*;
use spinoza::math::Float;

use crate::bit_reverse_permutation;

pub fn bravo(buf: &mut Vec<Float>) {
    const W: usize = 4;
    let n = buf.len();

    // let total_num_vecs: usize = n / W;
    let num_equiv_classes = n / W.pow(2);
    let num_data_pts_in_ec = n / num_equiv_classes;

    let all_target_vectors: Vec<_> = buf
        .par_chunks_exact(num_data_pts_in_ec)
        .flat_map(|ec| {
            // load W = 4 source vectors from the equivalence class
            let mut source_vectors: Vec<_> = ec.chunks_exact(W).map(f64x4::from_slice).collect();
            //assert_eq!(source_vectors.len(), W);

            // round 1
            let u = source_vectors[0];
            let v = source_vectors[1];
            let w = source_vectors[2];
            let x = source_vectors[3];

            let (u_i, v_i) = u.interleave(v);
            let (w_i, x_i) = w.interleave(x);

            source_vectors[0] = u_i;
            source_vectors[1] = v_i;
            source_vectors[2] = w_i;
            source_vectors[3] = x_i;

            // round 2
            let u = source_vectors[0];
            let v = source_vectors[2];
            let w = source_vectors[1];
            let x = source_vectors[3];

            let (u_i, v_i) = u.interleave(v);
            let (w_i, x_i) = w.interleave(x);

            source_vectors[0] = u_i;
            source_vectors[2] = v_i;
            source_vectors[1] = w_i;
            source_vectors[3] = x_i;

            source_vectors
        })
        .collect();

    buf.clear();

    if num_equiv_classes > 1 {
        let dist = (n >> 1) / W;
        println!("n: {n} dist: {dist}");

        let (vecs_side0, vecs_side1) = all_target_vectors.split_at(dist);
        vecs_side0
            .iter()
            .zip(vecs_side1.iter())
            .for_each(|(v0, v1)| {
                let (vi0, vi1) = v0.interleave(*v1);
                buf.extend(vi0.as_array().iter().chain(vi1.as_array().iter()));
            });
    } else {
        for target_vec in all_target_vectors {
            buf.extend(target_vec.as_array());
        }
    }
}

pub fn compare() {
    const N: usize = 16;

    let mut buf: Vec<Float> = (0..N).map(|i| i as Float).collect();
    bit_reverse_permutation(&mut buf);
    println!("expected (using naive BR): {:?}", buf);
}
