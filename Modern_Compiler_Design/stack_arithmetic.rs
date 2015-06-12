// Here are our stack machine instructions, based on the stack machine
// model described in §4.2.4 "Simple Code Generation".

fn push_constant(stack: &mut Vec<isize>, stack_pointer: &mut usize, c: isize) {
    stack.push(c);
    *stack_pointer += 1;
}

fn push_local(stack: &mut Vec<isize>, stack_pointer: &mut usize, i: usize) {
    let local: isize = stack[i];
    stack.push(local);
    *stack_pointer += 1;
}

fn store_local(stack: &mut Vec<isize>, stack_pointer: &mut usize, i: usize) {
    stack[i] = stack.pop().unwrap();
    *stack_pointer -= 1;
}

fn add_top_two(stack: &mut Vec<isize>, stack_pointer: &mut usize) {
    let addend = stack.pop().unwrap();
    let addend_redux = stack.pop().unwrap();
    stack.push(addend + addend_redux);
    *stack_pointer -= 1;
}

fn subtract_top_two(stack: &mut Vec<isize>, stack_pointer: &mut usize) {
    let addend = stack.pop().unwrap();
    let addend_redux = stack.pop().unwrap();
    stack.push(addend - addend_redux);
    *stack_pointer -= 1;
}

fn multiply_top_two(stack: &mut Vec<isize>, stack_pointer: &mut usize) {
    let addend = stack.pop().unwrap();
    let addend_redux = stack.pop().unwrap();
    stack.push(addend * addend_redux);
    *stack_pointer -= 1;
}


// Let's suppose we've been given an AST that has been parsed from an
// arithmetic expression. An AST with nodes described like this—

enum Payload {
    Value(isize),
    Position(usize),
    None
}

enum Operation {
    Constant,
    StoreLocal,
    PushLocal,
    Add,
    Subtract,
    Multiply
}

struct ArithmeticTreeNode {
    operation: Operation,
    payload: Payload,
    left: Option<Box<ArithmeticTreeNode>>,
    right: Option<Box<ArithmeticTreeNode>>
}

fn my_example_ast() -> ArithmeticTreeNode {
    let a = Payload::Position(1);
    let b = Payload::Position(2);
    let c = Payload::Position(3);

    let my_first_leaf = ArithmeticTreeNode{
        operation: Operation::Constant,
        payload: Payload::Value(1),
        left: None, right: None
    };
    let my_second_leaf = ArithmeticTreeNode{
        operation: Operation::Constant,
        payload: Payload::Value(1),
        left: None, right: None
    };

    let root = ArithmeticTreeNode{
        operation: Operation::Add,
        payload: Payload::None,
        left: Some(Box::new(my_first_leaf)),
        right: Some(Box::new(my_second_leaf)),
    };
    root
}

// Now we can implement the `Generate code` procedure described in Figure 4.26.

fn generate_code(node: ArithmeticTreeNode) {
    // TODO: finish

    // match node.operation {
    //     Operation::Constant => println!("push_constant(&mut my_stack, &mut my_stack_pointer, {});", node.),
    //     Operation::StoreLocal =>,
    //     Operation::PushLocal,
    //     Operation::Add,
    //     Operation::Subtract,
    //     Operation::Multiply
    // }
}

fn main() {
    let mut my_stack: Vec<isize> = Vec::new();
    let mut my_stack_pointer: usize = 0;
    push_constant(&mut my_stack, &mut my_stack_pointer, 1);
    push_constant(&mut my_stack, &mut my_stack_pointer, 2);
    add_top_two(&mut my_stack, &mut my_stack_pointer);
    println!("{:?} {}", my_stack, my_stack_pointer);
}
