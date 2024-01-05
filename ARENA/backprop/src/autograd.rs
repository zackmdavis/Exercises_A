use std::collections::HashMap;
use std::rc::Rc;


trait ForwardFunction {
    fn call(&self, inputs: Vec<Tensor>) -> Tensor;
}

struct Tensor {
    array: ArrayD<f32>,
    requires_gradient: bool,
    gradient: Option<ArrayD<f32>>,
    recipe: Option<Recipe>,
}

struct TensorBuilder {
    array: ArrayD<f32>,
    requires_gradient: bool,
    gradient: Option<ArrayD<f32>>,
    recipe: Option<Recipe>,
}

impl TensorBuilder {
    fn new(array: ArrayD<f32>) -> TensorBuilder {
        TensorBuilder {
            array,
            requires_gradient: false,
            gradient: None,
            recipe: None,
        }
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
        self.recipe = Some(recipe);
        self
    }

    fn build(self) -> Tensor {
        Tensor {
            array: self.array,
            requires_gradient: self.requires_gradient,
            gradient: self.gradient,
            recipe: self.recipe,
        }
    }
}

struct Recipe {
    forward_function: Box<dyn ForwardFunction>,
    parents: HashMap<usize, Rc<Tensor>>,
}


struct BackwardFunctionLookup {
    storage: HashMap<String, Box<dyn ForwardFunction>>
}


impl BackwardFunctionLookup {
    fn new() -> BackwardFunctionLookup {
        BackwardFunctionLookup {
            storage: HashMap::new(),
        }
    }

    fn add(&mut self, name: String, forward_function: Box<dyn ForwardFunction>) {
        self.storage.insert(name, function);
    }

    fn get(&self, name: &str) -> Option<&Box<dyn ForwardFunction>> {
        self.storage.get(name)
    }
}


fn log_forward(x: Tensor) -> Tensor {
    // TensorBuilder::new(x.array.map(|e| e.lg()))
    //     .requires_gradient(true)
    //     .recipe(Recipe{

    //     })
    //     .build()
}
