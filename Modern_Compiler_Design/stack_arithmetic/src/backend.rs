use std::fs::File;
use std::io;
use std::io::{Read, Write};

use frontend::{Payload, Operation, ArithmeticTreeNode};

// Now we can implement the `Generate code` procedure described in Figure 4.26.

// There are a few helper functions involved.

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

fn push_all_from_box_option(instruction_list_ref: &mut Vec<String>,
                            child_box_maybe: Option<Box<ArithmeticTreeNode>>) {
    if let Some(child_box) = child_box_maybe {
        let child_code = generate_codes(*child_box);
        push_all(instruction_list_ref, &child_code);
    }
}

pub fn generate_codes(node: ArithmeticTreeNode) -> Vec<String> {
    let mut instruction_list = Vec::new();
    match node.operation {
        Operation::Constant => emit(&mut instruction_list,
                                    "push_constant", node.payload),
        Operation::StoreLocal => emit(&mut instruction_list,
                                    "store_local", node.payload),
        Operation::PushLocal => emit(&mut instruction_list,
                                    "push_local", node.payload),
        Operation::Add => {
            push_all_from_box_option(&mut instruction_list, node.left);
            push_all_from_box_option(&mut instruction_list, node.right);
            emit(&mut instruction_list,
                 "add_top_two", node.payload)
        },
        Operation::Subtract => {
            push_all_from_box_option(&mut instruction_list, node.left);
            push_all_from_box_option(&mut instruction_list, node.right);
            emit(&mut instruction_list,
                 "subtract_top_two", node.payload)
        },
        Operation::Multiply => {
            push_all_from_box_option(&mut instruction_list, node.left);
            push_all_from_box_option(&mut instruction_list, node.right);
            emit(&mut instruction_list,
                 "multiply_top_two", node.payload)
        }
    }
    instruction_list
}

fn try_to_write_it(w: &mut io::BufWriter<&File>, bs: &[u8]) {
    w.write_all(bs).ok().unwrap();
}

pub fn write_codes(path: &str, codes: Vec<String>, debug: bool) {
    // writing set up
    let out_file = match File::create(path) {
        Ok(file) => file,
        Err(_) => panic!(format!("couldn't create {}", path))
    };
    let mut writer = io::BufWriter::new(&out_file);

    // load instruction bits
    let inst_file = match File::open("src/instructions.rs") {
        Ok(file) => file,
        Err(_) => panic!("couldn't load instructions")
    };
    let mut reader = io::BufReader::new(&inst_file);
    let inst_buffer = &mut String::new();
    reader.read_to_string(inst_buffer).ok();

    try_to_write_it(&mut writer, inst_buffer.as_bytes());
    try_to_write_it(&mut writer, b"fn main() {\n");
    try_to_write_it(&mut writer, b"let mut my_stack: Vec<isize> = Vec::new();");
    try_to_write_it(&mut writer, b"let mut my_stack_pointer: usize = 0;\n");
    for code in codes.iter() {
        try_to_write_it(&mut writer, code.as_bytes());
        try_to_write_it(&mut writer, &[10]);  // \n
        if debug {
            try_to_write_it(
                &mut writer,
                b"println!(\"{} {:?}\", my_stack_pointer,  my_stack);\n"
            );
        }
    }
    try_to_write_it(&mut writer, b"}\n");
}
