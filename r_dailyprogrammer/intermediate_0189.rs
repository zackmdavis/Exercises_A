// www.reddit.com
// /r/dailyprogrammer/comments
// /2ms946/20141119_challenge_189_intermediate_roman_numeral/

// This is my first Rust program that is not unredeemably laughably
// terrible!!

use std::collections::HashMap;
use std::env;

const PSEUDO_DIGITS: [char; 7] = ['M', 'D', 'C', 'L', 'X', 'V', 'I'];
const PSEUDO_PLACE_VALUES: [usize; 7] = [1000, 500, 100, 50, 10, 5, 1];

#[allow(unused_parens)]
fn roman_to_integer(roman: String) -> usize {
    let mut figure_to_value: HashMap<char, usize> = HashMap::new();
    for (&figure, &value) in PSEUDO_DIGITS.iter().zip(PSEUDO_PLACE_VALUES.iter())
    {
        figure_to_value.insert(figure, value);
    }
    let mut total_value_so_far: usize = 0;

    let mut value_stack: Vec<usize> = Vec::new();
    for figure in roman.chars() {
        let value: usize = *(figure_to_value.get(&figure).unwrap());
        let previously: Option<usize> = value_stack.pop();
        // this may not be entirely idiomatic and I may not have truly
        // seen the light of pattern-matching as yet
        if previously.is_none() {
            value_stack.push(value)
        } else {
            let previous_value: usize = previously.unwrap();
            if previous_value >= value {
                total_value_so_far += previous_value;
                value_stack.push(value);
            } else if previous_value < value {
                total_value_so_far += (value - previous_value);
            }
        }
    }
    let remaining_value = value_stack.pop();

    if remaining_value.is_some() {
        total_value_so_far += remaining_value.unwrap();
    }

    total_value_so_far
}

#[allow(unused_parens)]
fn integer_to_roman(integer: usize) -> String {
    let mut remaining = integer;
    let mut bildungsroman = String::new();
    // get it?? It sounds like _building Roman_ (numerals), but it's
    // also part of the story about me coming into my own as a
    // programmer by learning a grown-up language
    //
    // XXX http://tvtropes.org/pmwiki/pmwiki.php/Main/DontExplainTheJoke
    for ((index, value), &figure) in PSEUDO_PLACE_VALUES.iter()
        .enumerate().zip(PSEUDO_DIGITS.iter())
    {
        let factor = remaining / value;
        remaining = remaining % value;

        if figure == 'M' || factor < 4 {
            for _ in 0..factor {
                bildungsroman.push(figure);
            }
        }

        // IV, IX, XL, &c.
        let smaller_unit_index = index + 2 - (index % 2);
        if smaller_unit_index < PSEUDO_PLACE_VALUES.len() {
            let smaller_unit_value = PSEUDO_PLACE_VALUES[smaller_unit_index];
            let smaller_unit_figure = PSEUDO_DIGITS[smaller_unit_index];

            if value - remaining <= smaller_unit_value {
                bildungsroman.push(smaller_unit_figure);
                bildungsroman.push(figure);
                remaining -= (value - smaller_unit_value);
            }
        }
    }
    bildungsroman
}

fn main() {
    let mut from_integer: bool = true;
    let mut input: String = "".to_string();
    for (index, argument) in env::args().enumerate() {
        match index {
            0 => { },  // we don't care what the program is named
            1 => { input = argument; },  // we care about what you have to say
            2 => { if argument == "--from-roman" {
                // and how you want us to interpret it
                from_integer = false;
              }
            },
            _ => { }  // but not about anything that comes after
        }
    }
    if from_integer {
        let parsed: usize = input.parse().ok().expect("invalid input");
        println!("{}", integer_to_roman(parsed));
    } else {
        println!("{}", roman_to_integer(input));
    }
}


#[test]
fn test_roman_to_integer() {
    assert_eq!(1, roman_to_integer("I".to_string()));
    assert_eq!(12, roman_to_integer("XII".to_string()));
    assert_eq!(9, roman_to_integer("IX".to_string()));
    assert_eq!(98, roman_to_integer("XCVIII".to_string()));
    assert_eq!(3998, roman_to_integer("MMMCMXCVIII".to_string()));
}

#[test]
fn test_roman_to_integer_with_some_nonsubtractive_cases() {
    assert_eq!(integer_to_roman(1), "I".to_string());
    assert_eq!(integer_to_roman(12), "XII".to_string());
}

#[test]
fn test_integer_to_roman_with_some_subtractive_cases() {
    assert_eq!(integer_to_roman(9), "IX".to_string());
    assert_eq!(integer_to_roman(48), "XLVIII".to_string());
}

#[test]
fn integer_to_roman_grand_test() {
    let seventy_six_to_one_hundred_twelve_inclusive = [
        "LXXVI", "LXXVII", "LXXVIII", "LXXIX", "LXXX", "LXXXI", "LXXXII",
        "LXXXIII", "LXXXIV", "LXXXV", "LXXXVI", "LXXXVII", "LXXXVIII",
        "LXXXIX", "XC", "XCI", "XCII", "XCIII", "XCIV", "XCV", "XCVI",
        "XCVII", "XCVIII", "XCIX", "C", "CI", "CII", "CIII", "CIV", "CV",
        "CVI", "CVII", "CVIII", "CIX", "CX", "CXI", "CXII"];

    for (integer, expected_roman) in (76..113).zip(
        seventy_six_to_one_hundred_twelve_inclusive.iter())
    {
        assert_eq!(expected_roman.to_string(), integer_to_roman(integer));
    }
}

#[test]
fn integer_to_roman_stress_test() {
    assert_eq!("".to_string(), integer_to_roman(0));

    let tail_to_four_thousand = ["MMMCMXCVIII", "MMMCMXCIX", "MMMM"];
    for (integer, expected_roman) in (3998..4001).zip(
        tail_to_four_thousand.iter())
    {
        assert_eq!(expected_roman.to_string(), integer_to_roman(integer));
    }

    assert_eq!("MMMMMMXVI".to_string(), integer_to_roman(6016));
}
