#[derive(PartialEq,Eq,Debug)]
enum Tokenclass {
    Value,
    Position,
    Operator,
    Delimiter,
    Separator
}

#[derive(PartialEq,Eq,Debug)]
pub struct Token {
    class: Tokenclass,
    representation: char
}

pub fn lex(source: String) -> Vec<Token> {
    let mut tokens: Vec<Token> = Vec::new();
    let mut charstream = source.chars();
    loop {
        if let Some(c) = charstream.next() {
            match c {
                // TODO: constants above 9? (if we assume that you
                // have to have a space afterwards, then we can get away
                // with the zero-token lookahead seemingly inherent to
                // this basic iterator/match approach)
                '0'...'9' => {
                    tokens.push(
                        Token{ class: Tokenclass::Value,
                               representation: c }
                        );
                },
                '#' => {
                    if let Some(d) = charstream.next() {
                        match d {
                            // TODO variables above 9 as above
                            '0'...'9' => {
                                tokens.push(
                                    Token{ class: Tokenclass::Position,
                                           representation: d }
                                    );
                            },
                            _ => { panic!("expected a digit following '#'"); }
                        }
                    }

                },
                '+' | '-' | '*' => {
                    tokens.push(
                        Token{ class: Tokenclass::Operator,
                               representation: c }
                        );
                },
                '(' | ')' => {
                    tokens.push(
                        Token{ class: Tokenclass::Delimiter,
                               representation: c }
                        );
                },
                ';' => {
                    tokens.push(
                        Token{ class: Tokenclass::Separator,
                               representation: c }
                        );
                },
                ' ' | '\n' => { continue; },
                _ => { panic!("Unexpected input character {}", c); }
            }
        } else {
            break;
        }
    }
    tokens
}

#[test]
fn test_lexing() {
    let source: String = "(2 + 2);".to_string();
    let lexed = lex(source);
    assert_eq!(6, lexed.len());
    assert_eq!(
        lexed,
        vec![
            Token{ class: Tokenclass::Delimiter, representation: '(' },
            Token{ class: Tokenclass::Value, representation: '2' },
            Token{ class: Tokenclass::Operator, representation: '+' },
            Token{ class: Tokenclass::Value, representation: '2' },
            Token{ class: Tokenclass::Delimiter, representation: ')' },
            Token{ class: Tokenclass::Separator, representation: ';' },
        ]
    );
}


#[derive(PartialEq,Eq,Debug)]
#[allow(dead_code)]
pub enum Payload {
    Value(isize),
    Position(usize),
    None
}

#[derive(PartialEq,Eq,Debug)]
#[allow(dead_code)]
pub enum Operation {
    Constant,
    StoreLocal,
    PushLocal,
    Add,
    Subtract,
    Multiply
}

#[derive(PartialEq,Eq,Debug)]
pub struct ArithmeticTreeNode {
    pub operation: Operation,
    pub payload: Payload,
    pub left: Option<Box<ArithmeticTreeNode>>,
    pub right: Option<Box<ArithmeticTreeNode>>
}

fn parse_value(tokenvec: &Vec<Token>,
               from: &mut usize) -> ArithmeticTreeNode {
    let integer_token: &Token = &tokenvec[*from];
    *from += 1;
    ArithmeticTreeNode {
        operation: Operation::Constant,
        payload: Payload::Value(
            integer_token.representation.to_string().parse().ok().unwrap()),
        left: None,
        right: None
    }
}

fn parse_parenthesized(tokenvec: &Vec<Token>,
                       from: &mut usize) -> ArithmeticTreeNode {
    *from += 1;  // open paren

    let left: ArithmeticTreeNode = parse_value(tokenvec, from);

    let operator_token: &Token = &tokenvec[*from];
    let operation = match operator_token.representation {
        '+' => Operation::Add,
        '-' => Operation::Subtract,
        '*' => Operation::Multiply,
        ref t @ _ => panic!("panic in parse_parenthesized on seeing {:?}!!", t)
    };
    *from += 1;

    let right: ArithmeticTreeNode = parse_value(tokenvec, from);

    *from += 1; // close paren

    ArithmeticTreeNode {
        operation: operation,
        payload: Payload::None,
        left: Some(Box::new(left)),
        right: Some(Box::new(right))
    }
}

pub fn parse_expression(tokenvec: &Vec<Token>,
                        from: &mut usize) -> ArithmeticTreeNode {
    let mut treenodes: Vec<ArithmeticTreeNode> = Vec::new();
    while *from < tokenvec.len() {
        let next_expression = match tokenvec[*from].class {
            Tokenclass::Value => parse_value(tokenvec, from),
            Tokenclass::Delimiter => parse_parenthesized(tokenvec, from),
            ref t @ _ => panic!("panic in parse_expression on seeing {:?}!", t)
        };
        treenodes.push(next_expression);
    }
    println!("{:?}", treenodes);
    treenodes.swap_remove(0)
}

#[test]
fn test_parse_value() {
    let tokenvec = vec![Token{ class: Tokenclass::Value, representation: '2' }];
    let mut index = 0;
    assert_eq!(
        ArithmeticTreeNode{
            operation: Operation::Constant,
            payload: Payload::Value(2),
            left: None, right: None
        },
        parse_value(tokenvec, &mut index)
    );
}
