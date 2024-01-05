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


// Tests on multiply_back are failing where I get [12.0] but the solutions
// expect [4.0, 4.0, 4.0]. Looking at the instructor's solution for
// `unbroadcast`, it looks like we're not supposed to remove dimensions that
// were originally 1 (as contasted to the prepended 1 dimensions). Now that
// it's been pointed out, that makes sense!
//
// Let's add a test to `test_unbroadcast`. (And then we could PR that against
// the course later tonight?!—for the glory.) Except, wait—the test already
// covers a shape [2, 1, 3] vs. shape [5, 1, 2, 4, 3] case! That should be
// sufficient.

// Think step-by-step.
//
// let actual = multiply_back0(&grad_out, &c, &a, &b);
// let expected = arr1(&[4.0, 4.0, 4.0]).into_dyn();
//
// `multiply_back0` first calls `grad_out * y`.
// `grad_out` is 1×3; `y` is `b` which is 1×1 ... which makes it seem like
// `unbroadcast` is locally doing the right thing here?!
//
// The canonical tests expect `multiply_back0(grad_out, c, a, b)` to give [4.0,
// 4.0, 4.0], but `multiply_back1(grad_out, c, a, b)` (same args, same order)
// to give [12.0].
//
// Wait ... the instructor's solution actually gives `multiply_back0` as
// `unbroadcast(y * grad_out, x)`; I have x totally unused. That should have
// been a red flag (the unused output)—and then the tests pass. OK.

fn multiply_back0(grad_out: &ArrayD<f64>, _out: &ArrayD<f64>, x: &ArrayD<f64>, y: &ArrayD<f64>) -> ArrayD<f64> {
    let raw_gradient = y * grad_out;
    unbroadcast(&raw_gradient, x)
}

fn multiply_back1(grad_out: &ArrayD<f64>, _out: &ArrayD<f64>, x: &ArrayD<f64>, y: &ArrayD<f64>) -> ArrayD<f64> {
    let raw_gradient = x * grad_out;
    unbroadcast(&raw_gradient, y)
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
    let small = Array::ones(IxDyn(&[2, 1, 3]));
    let large = small.broadcast(IxDyn(&[5, 1, 2, 4, 3])).unwrap().to_owned();
    let out = unbroadcast(&large, &small);
    assert_eq!(out.shape(), small.shape());
    assert!(out.iter().all(|&x| (x - 20.0).abs() < 1e-6));

    let small = Array::ones(IxDyn(&[2, 1, 3]));
    let large = small.broadcast(IxDyn(&[5, 1, 2, 1, 3])).unwrap().to_owned();
    let out = unbroadcast(&large, &small);
    assert_eq!(out.shape(), small.shape());
    assert!(out.iter().all(|&x| (x - 5.0).abs() < 1e-6));

    let small = Array::ones(IxDyn(&[2, 1, 3]));
    let large = small.broadcast(IxDyn(&[2, 4, 3])).unwrap().to_owned();
    let out = unbroadcast(&large, &small);
    assert_eq!(out.shape(), small.shape());
    assert!(out.iter().all(|&x| (x - 4.0).abs() < 1e-6));
}

#[test]
fn test_multiply_back() {
    let a = arr1(&[1.0, 2.0, 3.0]).into_dyn();
    let b = arr1(&[2.0]).into_dyn();
    let c = &a * &b;

    let expected_c: Array<f64, _> = arr1(&[2., 4., 6.]).into_dyn();
    assert!(c.iter().zip(expected_c.iter()).all(|(&a, &e)| (a - e).abs() < 1e-6));

    let grad_out = arr1(&[2.0, 2.0, 2.0]).into_dyn();

    let actual = multiply_back0(&grad_out, &c, &a, &b);
    let expected = arr1(&[4.0, 4.0, 4.0]).into_dyn();
    println!("{:?}\n{:?}", actual, expected);
    assert_eq!(actual.shape(), expected.shape());
    assert!(actual.iter().zip(expected.iter()).all(|(&a, &e)| (a - e).abs() < 1e-6));

    let actual = multiply_back1(&grad_out, &c, &a, &b);
    let expected = arr1(&[12.0]).into_dyn();
    println!("{:?}\n{:?}", actual, expected);
    assert_eq!(actual.shape(), expected.shape());
    assert!(actual.iter().zip(expected.iter()).all(|(&a, &e)| (a - e).abs() < 1e-6));

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
