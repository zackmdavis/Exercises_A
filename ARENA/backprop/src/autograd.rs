use std::cell::RefCell;
use std::collections::HashMap;
use std::f32::consts::E;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::sync::Mutex;

use approx::assert_relative_eq;
use lazy_static::lazy_static;
use ndarray::prelude::*;
use topological_sort::TopologicalSort;

lazy_static! {
    static ref COUNTER: Mutex<u64> = Mutex::new(0);
}

fn generate_sequential_tensor_id() -> String {
    let mut num = COUNTER.lock().unwrap();
    *num += 1;
    format!("Tensor{}", num)
}

trait ForwardFunction {
    fn call(&self, inputs: Vec<Rc<Tensor>>) -> Rc<Tensor>;
    fn backward_function(&self, arg_number: usize) -> Box<dyn BackwardFunction>;
}

trait BackwardFunction {
    fn call(&self, out_gradient: ArrayD<f32>, value: Rc<Tensor>, args: Vec<Rc<Tensor>>) -> ArrayD<f32>;
}

#[derive(Clone)]
struct Tensor {
    identifier: String,
    array: ArrayD<f32>,
    requires_gradient: bool,
    gradient: RefCell<Option<ArrayD<f32>>>,
    recipe: Option<Rc<Recipe>>,
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("identifier", &self.identifier)
            .field("array", &self.array)
            .field("requires_gradient", &self.requires_gradient)
            .field("gradient", &self.gradient)
            .finish()
    }
}

impl Hash for Tensor {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.identifier.hash(state);
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.identifier == other.identifier
    }
}

impl Eq for Tensor {}

impl Tensor {
    fn is_leaf(&self) -> bool {
        !(self.requires_gradient
            && self.recipe.is_some()
            && !self
                .recipe
                .as_ref()
                .expect("`recipe.is_some()` is known")
                .parents
                .is_empty())
    }
}

struct TensorBuilder {
    array: ArrayD<f32>,
    identifier: Option<String>,
    requires_gradient: bool,
    gradient: Option<ArrayD<f32>>,
    recipe: Option<Rc<Recipe>>,
}

#[allow(dead_code)]
impl TensorBuilder {
    fn new(array: ArrayD<f32>) -> TensorBuilder {
        TensorBuilder {
            array,
            identifier: None,
            requires_gradient: true,
            gradient: None,
            recipe: None,
        }
    }

    fn identifier(mut self, identifier: String) -> TensorBuilder {
        self.identifier = Some(identifier);
        self
    }

    fn requires_gradient(mut self, requires: bool) -> TensorBuilder {
        self.requires_gradient = requires;
        self
    }

    fn gradient(mut self, gradient: ArrayD<f32>) -> TensorBuilder {
        self.gradient = Some(gradient);
        self
    }

    fn recipe(mut self, recipe: Recipe) -> TensorBuilder {
        self.recipe = Some(Rc::new(recipe));
        self
    }

    fn build(self) -> Tensor {
        Tensor {
            array: self.array,
            identifier: match self.identifier {
                Some(identifier) => identifier,
                None => generate_sequential_tensor_id(),
            },
            requires_gradient: self.requires_gradient,
            gradient: RefCell::new(self.gradient),
            recipe: self.recipe,
        }
    }
}

struct Recipe {
    forward_function: Box<dyn ForwardFunction>,
    parents: Vec<Rc<Tensor>>,
}

#[derive(Debug, Clone, Copy)]
struct LogForward {}

#[derive(Debug, Clone, Copy)]
struct LogBackward {}

impl BackwardFunction for LogBackward {
    fn call(
        &self,
        out_gradient: ArrayD<f32>,
        _value: Rc<Tensor>,
        args: Vec<Rc<Tensor>>,
    ) -> ArrayD<f32> {
        let x = &args[0];
        x.array.mapv(|i| 1. / i) * out_gradient
    }
}

impl ForwardFunction for LogForward {
    fn call(&self, inputs: Vec<Rc<Tensor>>) -> Rc<Tensor> {
        let input = &inputs[0];
        let output = TensorBuilder::new(input.array.map(|e| e.ln()))
            .recipe(Recipe {
                forward_function: Box::new(LogForward {}),
                parents: inputs.clone(),
            })
            .build();
        Rc::new(output)
    }

    fn backward_function(&self, arg_number: usize) -> Box<dyn BackwardFunction> {
        assert!(arg_number == 0);
        Box::new(LogBackward {})
    }
}

#[derive(Debug, Clone, Copy)]
struct MultiplyForward {}

#[derive(Debug, Clone, Copy)]
struct MultiplyBackward0 {}

#[derive(Debug, Clone, Copy)]
struct MultiplyBackward1 {}

impl BackwardFunction for MultiplyBackward0 {
    fn call(
        &self,
        out_gradient: ArrayD<f32>,
        _product: Rc<Tensor>,
        args: Vec<Rc<Tensor>>,
    ) -> ArrayD<f32> {
        args[1].array.clone() * out_gradient
        // Ultimately, there should be an `unbroadcast` here, but let's not
        // worry about that now.
    }
}

impl BackwardFunction for MultiplyBackward1 {
    fn call(
        &self,
        out_gradient: ArrayD<f32>,
        _product: Rc<Tensor>,
        args: Vec<Rc<Tensor>>,
    ) -> ArrayD<f32> {
        args[0].array.clone() * out_gradient
        // `unbroadcast` ditto
    }
}

impl ForwardFunction for MultiplyForward {
    fn call(&self, inputs: Vec<Rc<Tensor>>) -> Rc<Tensor> {
        let output = TensorBuilder::new(&inputs[0].array * &inputs[1].array)
            .recipe(Recipe {
                forward_function: Box::new(MultiplyForward {}),
                parents: inputs.clone(),
            })
            .build();
        Rc::new(output)
    }

    fn backward_function(&self, arg_number: usize) -> Box<dyn BackwardFunction> {
        match arg_number {
            0 => Box::new(MultiplyBackward0 {}),
            1 => Box::new(MultiplyBackward1 {}),
            _ => panic!("multiplication is a binary operation as far as this program is concerned"),
        }
    }
}

// The curriculum asked us to write a wrapper for generic forward functions,
// but I'm not sure how we're supposed to get the corresponding backward
// functions.
//
// struct GenericForward<F>
// where
//     F: Fn(Vec<ArrayD<f32>>) -> ArrayD<f32> + Clone + 'static,
// {
//     forward: F,
// }

// impl<F> ForwardFunction for GenericForward<F>
// where
//     F: Fn(Vec<ArrayD<f32>>) -> ArrayD<f32> + Clone + 'static,
// {
//     fn call(&self, inputs: Vec<Rc<Tensor>>) -> Rc<Tensor> {
//         let raw_inputs = inputs.iter().map(|i| i.array.clone()).collect();
//         let output = TensorBuilder::new((self.forward)(raw_inputs))
//             .recipe(Recipe {
//                 forward_function: Box::new(GenericForward {
//                     forward: self.forward.clone(),
//                 }),
//                 parents: inputs.clone(),
//             })
//             .build();
//         Rc::new(output)
//     }
// }

#[allow(dead_code)]
fn array_sum(inputs: Vec<ArrayD<f32>>) -> ArrayD<f32> {
    inputs.into_iter().reduce(|a, b| a + b).unwrap()
}

fn register_parents(sorter: &mut TopologicalSort<Rc<Tensor>>, child: Rc<Tensor>) {
    if let Some(recipe) = &child.recipe {
        for parent in &recipe.parents {
            sorter.add_dependency(parent.clone(), child.clone());
            register_parents(sorter, parent.clone());
        }
    }
}

fn sorted_computation_graph(end: Rc<Tensor>) -> Vec<Rc<Tensor>> {
    let mut sorter = TopologicalSort::new();
    register_parents(&mut sorter, end);
    let mut sorted = sorter.collect::<Vec<_>>();
    // We actually want reverse-topological order
    sorted.reverse();
    sorted
}

fn backprop(end: Rc<Tensor>) {
    // We can use Tensors as map keys after implementing Hash
    let mut gradients = HashMap::<Rc<Tensor>, ArrayD<f32>>::new();
    gradients.insert(end.clone(), array![1.].into_dyn());
    for node in sorted_computation_graph(end) {
        let out_gradient = gradients
            .remove(&node)
            .expect("expected gradient to have been stored");
        if node.is_leaf() && node.requires_gradient {
            let mut needs_init = true;
            if let Some(gradient) = node.gradient.borrow_mut().as_mut() {
                needs_init = false;
                *gradient = &*gradient + out_gradient.clone();
            }
            if needs_init {
                *node.gradient.borrow_mut() = Some(out_gradient.clone());
            }
        }

        if node.recipe.is_none() || node.recipe.as_ref().expect("previous disjunct covered `None` case").parents.is_empty() {
            continue;
        }

        for (i, parent) in node.recipe.as_ref().expect("recipe known to exist").parents.iter().enumerate() {
            let in_gradient = node.recipe.as_ref().expect("recipe known to exist").forward_function.backward_function(i).call(
                out_gradient.clone(),
                node.clone(),
                node.recipe.as_ref().expect("recipe known to exist").parents.clone(),
            );

            match gradients.get_mut(parent) {
                Some(gradient) => {
                    *gradient = gradient.clone() + in_gradient;
                },
                None => {
                    gradients.insert(parent.clone(), in_gradient);
                }
            }
        }
    }
}

#[test]
fn test_log() {
    let a = Rc::new(TensorBuilder::new(array![E, E.powf(E)].into_dyn()).build());
    let b = LogForward {}.call(vec![a]);
    assert!(b.array[0] - 1. < 0.000001);
    assert!(b.array[1] - E < 0.000001);
}

#[test]
fn test_multiply() {
    let a = Rc::new(TensorBuilder::new(array![[1., 2., 3.], [4., 5., 6.]].into_dyn()).build());
    let b = Rc::new(TensorBuilder::new(array![2., 2., 2.].into_dyn()).build());
    let c = MultiplyForward {}.call(vec![a, b]);
    let expected = array![[2., 4., 6.], [8., 10., 12.]].into_dyn();
    assert_eq!(c.array, expected);
}

// #[test]
// fn test_sum() {
//     let sum_forward = GenericForward { forward: array_sum };
//     let a = Rc::new(TensorBuilder::new(array![[1., 2., 3.], [4., 5., 6.]].into_dyn()).build());
//     let b = Rc::new(TensorBuilder::new(array![[2., 2., 2.], [2., 2., 2.]].into_dyn()).build());
//     let c = sum_forward.call(vec![a, b]);
//     let expected = array![[3., 4., 5.], [6., 7., 8.]].into_dyn();
//     assert_eq!(c.array, expected);
// }


#[test]
fn test_backprop() {
    let a = Rc::new(TensorBuilder::new(array![E, E.powf(E)].into_dyn()).build());
    let b = LogForward {}.call(vec![a.clone()]);
    let c = LogForward {}.call(vec![b.clone()]);
    backprop(c);
    let expected = 1. / b.array.clone() / a.array.clone();
    assert_relative_eq!(a.gradient.borrow().clone().unwrap(), expected);
}

#[test]
fn test_backprop_branching() {
    let a = Rc::new(TensorBuilder::new(array![1., 2., 3.].into_dyn()).build());
    let b = Rc::new(TensorBuilder::new(array![1., 2., 3.].into_dyn()).build());
    let c = MultiplyForward {}.call(vec![a.clone(), b.clone()]);
    backprop(c);
    assert_relative_eq!(a.gradient.borrow().clone().unwrap(), b.array);
    assert_relative_eq!(b.gradient.borrow().clone().unwrap(), a.array);
}
