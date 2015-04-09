// www.reddit.com
// /r/dailyprogrammer/comments
// /2ms946/20141119_challenge_189_intermediate_roman_numeral/

use std::cmp::Ordering;
use std::cmp::Ordering::{Greater, Equal, Less};

fn index_of_char_in_seven_element_array(array: [char; 7], target: char) ->
    // it's a best practice to use a ridiculous name when you write a
    // ridiculous function, to make sure that you don't get used to it
    Option<usize>
{
    for (index, &item) in array.iter().enumerate() {
        if item == target {
            return Some(index)
        }
    }
    return None;
}

fn roman_pseudo_digit_comparator(
    first_pseudo_digit: char, second_pseudo_digit: char
        ) -> Ordering {
    let pseudo_digits: [char; 7] = ['M', 'D', 'C', 'L', 'X', 'V', 'I'];
    let a = index_of_char_in_seven_element_array(pseudo_digits,
                                                 first_pseudo_digit);
    let b = index_of_char_in_seven_element_array(pseudo_digits,
                                                 second_pseudo_digit);
    if a > b {
        Greater
    } else if a == b {
        Equal
    } else if a < b {
        Less
    } else {
        panic!("This can't be happening")
    }
}


fn roman_to_integer(roman: String) -> usize {
    let mut total = 0;
    let mut previous_figure: Option<char> = None;
    for figure in roman.chars() {
        // TODO
    }
    // TODO
    10usize
}

fn integer_to_roman(integer: usize) -> String {
    let pseudo_digits: [char; 7] = ['M', 'D', 'C', 'L', 'X', 'V', 'I'];
    let pseudo_place_values: [usize; 7] = [1000, 500, 100, 50, 10, 5, 1];

    let mut remaining = integer;
    let mut bildungsroman = String::new();
    // get it?? It sounds like _building Roman_ (numerals), but it's
    // also part of the story about me growing up as a programmer by
    // learning this dreadful language
    //
    // XXX http://tvtropes.org/pmwiki/pmwiki.php/Main/DontExplainTheJoke
    for ((index, value), &figure) in pseudo_place_values.iter()
        .enumerate().zip(pseudo_digits.iter())
    {
        let factor = remaining / value;
        remaining = remaining % value;
        if factor < 4 {
            for _ in 0..factor {
                bildungsroman.push(figure);
            }
        } else {
            // XXX WRONG: `'assertion failed [...] (left: `"VIV"`, right: `"IX"`)'`
            bildungsroman.push(figure);
            bildungsroman.push(pseudo_digits[index-1]);
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
