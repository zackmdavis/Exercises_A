use std::cell::RefCell;
use std::collections::HashMap;
use std::f32::consts::E;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::rc::Rc;
use std::sync::Mutex;

use lazy_static::lazy_static;
use ndarray::prelude::*;
use num_traits::pow::Pow;
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
        self.requires_gradient
            && self.recipe.is_some()
            && !self
                .recipe
                .as_ref()
                .expect("`recipe.is_some()` is known")
                .parents
                .is_empty()
    }
}

struct TensorBuilder {
    array: ArrayD<f32>,
    identifier: Option<String>,
    requires_gradient: bool,
    gradient: Option<ArrayD<f32>>,
    recipe: Option<Rc<Recipe>>,
}

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

// The `dyn ForwardFunction` here isn't right—we're storing backwards functions
//
// struct BackwardFunctionLookup {
//     storage: HashMap<String, Box<dyn ForwardFunction>>
// }

// impl BackwardFunctionLookup {
//     fn new() -> BackwardFunctionLookup {
//         BackwardFunctionLookup {
//             storage: HashMap::new(),
//         }
//     }

//     fn add(&mut self, name: String, forward_function: Box<dyn ForwardFunction>) {
//         self.storage.insert(name, forward_function);
//     }

//     fn get(&self, name: &str) -> Option<&Box<dyn ForwardFunction>> {
//         self.storage.get(name)
//     }
// }

#[derive(Debug, Clone, Copy)]
struct LogForward {}

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
}

#[derive(Debug, Clone, Copy)]
struct MultiplyForward {}

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
}

struct GenericForward<F>
where
    F: Fn(Vec<ArrayD<f32>>) -> ArrayD<f32> + Clone + 'static,
{
    forward: F,
}

impl<F> ForwardFunction for GenericForward<F>
where
    F: Fn(Vec<ArrayD<f32>>) -> ArrayD<f32> + Clone + 'static,
{
    fn call(&self, inputs: Vec<Rc<Tensor>>) -> Rc<Tensor> {
        let raw_inputs = inputs.iter().map(|i| i.array.clone()).collect();
        let output = TensorBuilder::new((self.forward)(raw_inputs))
            .recipe(Recipe {
                forward_function: Box::new(GenericForward {
                    forward: self.forward.clone(),
                }),
                parents: inputs.clone(),
            })
            .build();
        Rc::new(output)
    }
}

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
    let mut gradients = HashMap::<String, ArrayD<f32>>::new();
    gradients.insert(end.identifier.clone(), array![1.].into_dyn());
    for node in sorted_computation_graph(end) {
        // TODO: continue ...
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

#[test]
fn test_sum() {
    let sum_forward = GenericForward { forward: array_sum };
    let a = Rc::new(TensorBuilder::new(array![[1., 2., 3.], [4., 5., 6.]].into_dyn()).build());
    let b = Rc::new(TensorBuilder::new(array![[2., 2., 2.], [2., 2., 2.]].into_dyn()).build());
    let c = sum_forward.call(vec![a, b]);
    let expected = array![[3., 4., 5.], [6., 7., 8.]].into_dyn();
    assert_eq!(c.array, expected);
}
