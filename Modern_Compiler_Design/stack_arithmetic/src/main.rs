// Let's suppose we've been given an AST that has been parsed from an
// arithmetic expression. An AST with nodes described like this—

mod frontend;
mod backend;

use frontend::{Payload, Operation, ArithmeticTreeNode};


#[allow(unused_variables)]
fn my_example_ast() -> ArithmeticTreeNode {
    let a = Payload::Position(1);
    let b = Payload::Position(2);
    let c = Payload::Position(3);

    let my_first_leaf = ArithmeticTreeNode{
        operation: Operation::Constant,
        payload: Payload::Value(3),
        left: None, right: None
    };
    let my_second_leaf = ArithmeticTreeNode{
        operation: Operation::Constant,
        payload: Payload::Value(5),
        left: None, right: None
    };

    let root = ArithmeticTreeNode{
        operation: Operation::Subtract,
        payload: Payload::None,
        left: Some(Box::new(my_first_leaf)),
        right: Some(Box::new(my_second_leaf)),
    };
    root
}

fn main() {
    let mut index = 0usize;
    let my_ast: ArithmeticTreeNode = frontend::parse_expression(
        &frontend::lex(
            "(2 + 2)".to_string()
        ),
        &mut index
    );
    // let my_ast: ArithmeticTreeNode = my_example_ast();

    let teh_codes = backend::generate_codes(my_ast);
    backend::write_codes("out.rs", teh_codes,
                         // TODO: get debug arg from CLI argv
                         true);
}
