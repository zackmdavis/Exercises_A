// reddit.com
// /r/dailyprogrammer/comments
// /11par4/10182012_challenge_104_intermediate_bracket_racket/

fn is_open_delimiter(delimiter: char) -> bool {
    let open_delimiters: [char; 4] = [
        40 as char,  // (
        60 as char,  // <
        91 as char,  // [
        123 as char,  // {
    ];
    for true_open_delimiter in open_delimiters.iter() {
        if delimiter == *true_open_delimiter {
            return true;
        }
    }
    false
}

fn is_close_delimiter(delimiter: char) -> bool {
    let close_delimiters: [char; 4] = [
        41 as char,  // )
        62 as char,  // >
        93 as char,  // ]
        125 as char,  // }
    ];
    for true_close_delimiter in close_delimiters.iter() {
        if delimiter == *true_close_delimiter {
            return true;
        }
    }
    false
}

fn are_matching_delimiters(first: char, second: char) -> bool {
    ((first == (40 as char) && second == (41 as char)) ||
     (first == (60 as char) && second == (62 as char)) ||
     (first == (91 as char) && second == (93 as char)) ||
     (first == (123 as char) && second == (125 as char)))
}

fn is_paired_correctly(expression: String) -> bool {
    let mut unmatched: Vec<char> = Vec::new();
    for character in expression.chars() {
        if is_open_delimiter(character) {
            unmatched.push(character);
        }
        if is_close_delimiter(character) {
            let popped: Option<char> = unmatched.pop();
            let correctly_matches: bool = match popped {
                Some(opener) => are_matching_delimiters(opener, character),
                None => false
            };
            if !correctly_matches {
                return false;
            }
        }
    }
    unmatched.len() == 0
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

#[test]
fn test_is_paired_correctly() {
    assert!(is_paired_correctly(
        ".format(\"{foo}\", {'foo': \"bar\"})".to_string()));
    assert!(is_paired_correctly("([{<()>}()])".to_string()));

    assert!(!is_paired_correctly(
        "foo(''.format(\"[no pun intended]\",)".to_string()));
    assert!(!is_paired_correctly(
        "foo(''.format(\"[no pun intended]\",)))".to_string()));
    assert!(!is_paired_correctly(
        "([{<()>}()))".to_string()));
    assert!(!is_paired_correctly("(((".to_string()));
}
