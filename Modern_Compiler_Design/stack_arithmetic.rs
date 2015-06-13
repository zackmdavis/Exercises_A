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

// But there are a few helper functions involved.

fn push_all<T: Clone>(ourself: &mut Vec<T>, other: &Vec<T>) {
    // There actually is a `push_all` implemented for Vecs, but
    // it's marked "unstable", so I can't use it with the stable-like
    // rustc release channel that I'm currently using (I actually
    // haven't upgraded since beta 9854143cb).
    //
    // https://doc.rust-lang.org/std/vec/struct.Vec.html#method.push_all
    for thing in other.iter() {
        let copy_of_thing = thing.clone();
        ourself.push(copy_of_thing);
    }
}

fn emit(instruction_list: &mut Vec<String>,
        instruction: &str,
        argument: Payload) {
    let rendered_argument: String = match argument {
        Payload::Value(c) => format!(", {}", c),
        Payload::Position(i) => format!(", {}", i),
        Payload::None => "".to_string()
    };
    let rendered_instruction = format!(
        "{}(&mut my_stack, &mut my_stack_pointer{});",
        instruction, rendered_argument
    );
    instruction_list.push(rendered_instruction);
}



fn generate_code(node: ArithmeticTreeNode) -> Vec<String> {
    let mut instruction_list = Vec::new();
    match node.operation {
        Operation::Constant => emit(&mut instruction_list,
                                    "push_constant", node.payload),
        Operation::StoreLocal => emit(&mut instruction_list,
                                    "store_local", node.payload),
        Operation::PushLocal => emit(&mut instruction_list,
                                    "push_local", node.payload),
        Operation::Add => {
            if let Some(child_box) = node.left {
                let child_code = generate_code(*child_box);
                push_all(&mut instruction_list, &child_code);
            }
            if let Some(child_box) = node.right {
                let child_code = generate_code(*child_box);
                push_all(&mut instruction_list, &child_code);
            }
            emit(&mut instruction_list,
                 "add_top_two", node.payload)
        },
        Operation::Subtract => {
            if let Some(child_box) = node.left {
                let child_code = generate_code(*child_box);
                push_all(&mut instruction_list, &child_code);
            }
            if let Some(child_box) = node.right {
                let child_code = generate_code(*child_box);
                push_all(&mut instruction_list, &child_code);
            }
            emit(&mut instruction_list,
                 "subtract_top_two", node.payload)
        },
        Operation::Multiply => {
            if let Some(child_box) = node.left {
                let child_code = generate_code(*child_box);
                push_all(&mut instruction_list, &child_code);
            }
            if let Some(child_box) = node.right {
                let child_code = generate_code(*child_box);
                push_all(&mut instruction_list, &child_code);
            }
            emit(&mut instruction_list,
                 "multiply_top_two", node.payload)
        }
    }
    instruction_list
}


fn generated_code_example_experiment_test() {
    let mut my_stack: Vec<isize> = Vec::new();
    let mut my_stack_pointer: usize = 0;

    // copy-pasted from output of actual codegen
    push_constant(&mut my_stack, &mut my_stack_pointer, 1);
    push_constant(&mut my_stack, &mut my_stack_pointer, 1);
    add_top_two(&mut my_stack, &mut my_stack_pointer);

    println!("{:?} {}", my_stack, my_stack_pointer);
}

fn main() {
    let my_ast: ArithmeticTreeNode = my_example_ast();
    let teh_codes = generate_code(my_ast);
    for codes in teh_codes.iter() {
        println!("{}", codes);
    }
    // generated_code_example_experiment_test()
}
