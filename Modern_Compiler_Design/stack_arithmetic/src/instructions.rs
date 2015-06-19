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
