// reddit.com
// /r/dailyprogrammer/comments
// /11par4/10182012_challenge_104_intermediate_bracket_racket/

fn character_is_in_length_four_array(character: char, array: [char; 4]) -> bool {
    for &item in array.iter() {
        if item == character {
            return true
        }
    }
    false
}

fn are_matching_delimiters(first: char, second: char) -> bool {
    ((first == ('(') && second == (')')) ||
     (first == ('<') && second == ('>')) ||
     (first == ('[') && second == (']')) ||
     (first == ('{') && second == ('}')))
}

fn is_paired_correctly(expression: String) -> bool {
    let open_delimiters: [char; 4] = ['(', '<', '[', '{'];
    let close_delimiters: [char; 4] = [')', '>', ']', '}'];
    let mut unmatched: Vec<char> = Vec::new();
    for character in expression.chars() {
        if character_is_in_length_four_array(character, open_delimiters) {
            unmatched.push(character);
        }
        if character_is_in_length_four_array(character, close_delimiters) {
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
