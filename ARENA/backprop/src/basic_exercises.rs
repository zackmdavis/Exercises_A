use ndarray::prelude::*;
use std::f64::consts::E;


fn log_back(grad_out: Array<f64, Ix2>, _out: Array<f64, Ix2>, x: Array<f64, Ix2>) -> Array<f64, Ix2> {
    // 1/x * grad_out
    x.mapv(|i| 1./i) * grad_out
}


// GPT-4 gave me a skeleton to work from ... and also rewrote the `axes_to_sum`
// part when it turned out that we needed to adjust the later indices when the
// axis-sum changed their interpretation. (There's no reason I couldn't have
// written that code—it's not hard—but GPT-4 did it faster, cleaner: I didn't
// know about `.skip()`.) A cool but somewhat alienating experience ... and the
// implications!
fn unbroadcast(broadcasted: &Array<f64, IxDyn>, original: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
    // Step 1: Compare shapes and determine which dimensions have been broadcasted
    let broadcast_shape = broadcasted.shape();
    let original_shape = original.shape();
    let mut axes_to_sum = Vec::new();

    let prepended_ones = broadcast_shape.len() - original_shape.len();
    for i in 0..prepended_ones {
        axes_to_sum.push(i);
    }
    for (i, &dim) in original_shape.iter().enumerate() {
        if dim == 1 {
            axes_to_sum.push(prepended_ones + i);
        }
    }

    let mut debroadcasted = broadcasted.clone();
    for i in 0..axes_to_sum.len() {
        let axis = axes_to_sum[i];
        debroadcasted = debroadcasted.sum_axis(Axis(axis));

        for a in axes_to_sum.iter_mut().skip(i + 1) {
            *a -= 1;
        }
    }

    debroadcasted.into_shape(IxDyn(&original_shape)).unwrap()
}


fn multiply_back0(grad_out: &ArrayD<f64>, _out: &ArrayD<f64>, _x: &ArrayD<f64>, y: &ArrayD<f64>) -> ArrayD<f64> {
    let raw_gradient = grad_out * y;
    unbroadcast(&raw_gradient, y)
}


fn multiply_back1(grad_out: &ArrayD<f64>, _out: &ArrayD<f64>, x: &ArrayD<f64>, _y: &ArrayD<f64>) -> ArrayD<f64> {
    let raw_gradient = grad_out * x;
    unbroadcast(&raw_gradient, x)
}



#[test]
fn test_log_back() {
    let a = array![[1., E, E.powf(E)]];
    let b = a.mapv(|x| x.log(E));
    let grad_out = array![[2., 2., 2.]];
    let actual = log_back(grad_out, b, a);
    let expected = array![[2., 2./E, 2./E.powf(E)]];
    assert_eq!(expected, actual);
}


// Tests translated to Rust by GPT-4 from the course's test cases.

#[test]
fn test_unbroadcast() {
    // Test case 1
    let small = Array::ones(IxDyn(&[2, 1, 3]));
    let large = small.broadcast(IxDyn(&[5, 1, 2, 4, 3])).unwrap().to_owned();
    let out = unbroadcast(&large, &small);
    assert_eq!(out.shape(), small.shape());
    assert!(out.iter().all(|&x| (x - 20.0).abs() < 1e-6));

    // Test case 2
    let small = Array::ones(IxDyn(&[2, 1, 3]));
    let large = small.broadcast(IxDyn(&[5, 1, 2, 1, 3])).unwrap().to_owned();
    let out = unbroadcast(&large, &small);
    assert_eq!(out.shape(), small.shape());
    assert!(out.iter().all(|&x| (x - 5.0).abs() < 1e-6));

    // Test case 3
    let small = Array::ones(IxDyn(&[2, 1, 3]));
    let large = small.broadcast(IxDyn(&[2, 4, 3])).unwrap().to_owned();
    let out = unbroadcast(&large, &small);
    assert_eq!(out.shape(), small.shape());
    assert!(out.iter().all(|&x| (x - 4.0).abs() < 1e-6));
}

#[test]
fn test_multiply_back() {
    // Test case 1
    let a = arr1(&[1.0, 2.0, 3.0]).into_dyn();
    let b = arr1(&[2.0]).into_dyn();
    let c = &a * &b;
    let grad_out = arr1(&[2.0, 2.0, 2.0]).into_dyn();
    let actual = multiply_back0(&grad_out, &c, &a, &b);
    let expected = arr1(&[4.0, 4.0, 4.0]).into_dyn();
    assert_eq!(actual.shape(), expected.shape());
    assert!(actual.iter().zip(expected.iter()).all(|(&a, &e)| (a - e).abs() < 1e-6));

    let actual = multiply_back1(&grad_out, &c, &a, &b);
    let expected = arr1(&[12.0]).into_dyn();
    println!("{:?}\n{:?}", actual, expected);
    assert_eq!(actual.shape(), expected.shape());
    assert!(actual.iter().zip(expected.iter()).all(|(&a, &e)| (a - e).abs() < 1e-6));

    // Test case 2
    let a = arr1(&[1.0, 2.0]).into_dyn();
    let b = arr2(&[[2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]).into_dyn();
    let c = &a * &b;
    let grad_out = arr2(&[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]).into_dyn();
    let actual = multiply_back0(&grad_out, &c, &a, &b);
    let expected = arr1(&[2.0 * 1.0 + 3.0 * 2.0 + 4.0 * 3.0, 3.0 * 1.0 + 4.0 * 2.0 + 5.0 * 3.0]).into_dyn();
    assert_eq!(actual.shape(), expected.shape());
    assert!(actual.iter().zip(expected.iter()).all(|(&a, &e)| (a - e).abs() < 1e-6));

    let actual = multiply_back1(&grad_out, &c, &a, &b);
    let expected = arr2(&[[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]).into_dyn();
    assert_eq!(actual.shape(), expected.shape());
    assert!(actual.iter().zip(expected.iter()).all(|(&a, &e)| (a - e).abs() < 1e-6));
}
