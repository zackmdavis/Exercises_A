// reddit.com
// /r/dailyprogrammer/comments
// /11par4/10182012_challenge_104_intermediate_bracket_racket/

fn is_open_delimiter(delimiter: char) -> bool {
    let OPEN_DELIMITERS: [char; 4] = [
        40 as char,  // (
        60 as char,  // <
        91 as char,  // [
        123 as char,  // {
    ];
    for true_open_delimiter in OPEN_DELIMITERS.iter() {
        if delimiter == *true_open_delimiter {
            return true;
        }
    }
    false
}

fn is_close_delimiter(delimiter: char) -> bool {
    let CLOSE_DELIMITERS: [char; 4] = [
        41 as char,  // )
        62 as char,  // >
        93 as char,  // ]
        125 as char,  // }
    ];
    for true_close_delimiter in CLOSE_DELIMITERS.iter() {
        if delimiter == *true_close_delimiter {
            return true;
        }
    }
    false
}

fn is_paired_correctly(expression: String) -> bool {
    let mut delimiter_stack: Vec<char> = Vec::new();
    for character in expression.chars() {
        if is_open_delimiter(character) {
            delimiter_stack.push(character);
        }
        if is_close_delimiter(character) {
            let popped = delimiter_stack.pop();
            if popped == None {
                return false
            };
            let opener_to_match = match popped {
                Some(opener) => opener,
                _ => panic!("Haven't you people ever heard of \
                             closing a goddam door?")
            };
            let okay: bool = match opener_to_match {
                // XXX doesn't compile, can't find documentation about
                // character literals even though they must exist because
                // Mozilla wouldn't want to be a laughingstock for
                // letting a language get to 1.0 alpha without them,
                // world is mad
                40 as char => character == 41 as char,
                60 as char => character == 62 as char,
                91 as char => character == 93 as char,
                123 as char => character == 125 as char,
                _ => panic!("\"No! This can't be happening!\" screamed the \
                             Baron as the walls of the matching construct \
                             came tumbling down and the waves crashed in on \
                             its inhabitants.")
            };
            if !okay {
                return false
            }
        }
    }
}


#[test]
fn test_is_open_delimiter() {
    let positive_cases = ["(", "[", "{", "<"];
    for case in positive_cases.iter() {
        match case.chars().nth(0) {
            Some(character) => assert!(is_open_delimiter(character)),
            None => assert!(false)
        }
    }
    let negative_cases = ["|", "!", "a"];
    for case in negative_cases.iter() {
        match case.chars().nth(0) {
            Some(character) => assert!(!is_open_delimiter(character)),
            None => assert!(false)
        }
    }
}

#[test]
fn test_is_close_delimiter() {
    let positive_cases = [")", "]", "}", ">"];
    for case in positive_cases.iter() {
        match case.chars().nth(0) {
            Some(character) => assert!(is_close_delimiter(character)),
            None => assert!(false)
        }
    }
    let negative_cases = ["|", "!", "a"];
    for case in negative_cases.iter() {
        match case.chars().nth(0) {
            Some(character) => assert!(!is_close_delimiter(character)),
            None => assert!(false)
        }
    }
}

#[cfg(do_not_compile_this_yet)]
#[test]
fn test_is_paired_correctly() {
    assert!(is_paired_correctly(".format(\"{foo}\", {'foo': \"bar\"})"));
    assert!(is_paired_correctly("([{<()>}()])"));

    assert!(!is_paired_correctly("foo(''.format(\"[no pun intended]\",)"));
    assert!(!is_paired_correctly("foo(''.format(\"[no pun intended]\",)))"));
    assert!(!is_paired_correctly("([{<()>}()))"));
    assert!(!is_paired_correctly("((("));
}
