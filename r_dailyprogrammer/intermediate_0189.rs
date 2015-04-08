// www.reddit.com
// /r/dailyprogrammer/comments
// /2ms946/20141119_challenge_189_intermediate_roman_numeral/


fn roman_to_integer(roman: String) -> usize {
    // TODO
    10usize
}

fn integer_to_roman(integer: usize) -> String {
    let pseudo_digits: [char; 5] = ['C', 'L', 'X', 'V', 'I'];
    let pseudo_place_values: [usize; 5] = [100, 50, 10, 5, 1];

    let mut remaining = integer;
    let mut bildungsroman = String::new();
    // get it?? It sounds like _building Roman_ (numerals), but it's
    // also part of the story about me growing up as a programmer by
    // learning this dreadful language
    //
    // XXX http://tvtropes.org/pmwiki/pmwiki.php/Main/DontExplainTheJoke
    for (value, &figure) in pseudo_place_values.iter().zip(pseudo_digits.iter())
    {
        let factor = remaining / value;
        remaining = remaining % value;
        // TODO: support substractive cases like four and nine
        for _ in 0..factor {
            bildungsroman.push(figure);
        }
    }
    bildungsroman
}


#[test]
fn test_roman_to_integer() {
    // TODO
    assert_eq!(1, roman_to_integer("I".to_string()));
    assert_eq!(12, roman_to_integer("XII".to_string()));
    assert_eq!(9, roman_to_integer("IX".to_string()));
    assert_eq!(98, roman_to_integer("XCVIII".to_string()));

}

#[test]
fn test_roman_to_integer_nonsubtractive_cases() {
    assert_eq!(integer_to_roman(1), "I".to_string());
    assert_eq!(integer_to_roman(12), "XII".to_string());
}

#[test]
fn test_integer_to_roman_with_subtractive_cases() {
    // TODO
    assert_eq!(integer_to_roman(9), "IX".to_string());
    assert_eq!(integer_to_roman(98), "XCVIII".to_string());
}
