//www.reddit.com/r/dailyprogrammer/comments
//3bi5na/20150629_challenge_221_easy_word_snake/

// Word snake!! (a trivial exercise from the subreddit to get the blood
// flowing into my fingers on what had been altogether too lazy of a
// Sunday)

fn align_horizontally(alignment: &usize) {
    for _ in 0..*alignment {
        print!(" ");
    }
}

fn print_simple_right_down_word_snake(words: Vec<&str>) {
    let mut horizontal_displacement = 0;

    // special-case the first word, which, unlike the rest, won't
    // already have had its first letter printed (as the last letter of
    // the previous word)
    let the_first_word = words[0];
    println!("{}", the_first_word);
    horizontal_displacement += the_first_word.len() - 1;

    // loop through the other words in the obvious way
    let mut horizontal = false;
    for word in words.iter().skip(1) {
        if horizontal {
            for character in word.chars().skip(1) {
                print!("{}", character);
            }
            println!("");
            horizontal_displacement += word.len() - 1;
        } else {
            for (index, character) in word.chars().skip(1).enumerate() {
                align_horizontally(&horizontal_displacement);
                print!("{}", character);
                if index != word.len()-2 {
                    println!("");
                }
            }
        }
        horizontal = !horizontal;
    }
    if horizontal {
        println!(""); // all outputs should have trailing newline
        // (because you don't want to misalign someone's terminal prompt)
    }
}

fn main() {
    print_simple_right_down_word_snake(vec![
        "SHENANIGANS", "SALTY", "YOUNGSTER", "ROUND",
        "DOUBLET", "TERABYTE", "ESSENCE"
    ]);
}

// even this very simple exercise had a few unanticipated subtleties

// zmd@ExpectedReturn:~/Code/Exercises_A/r_dailyprogrammer$ ./easy_0221
// SHENANIGANS
//           A
//           L
//           T
//           YOUNGSTER
//                   O
//                   U
//                   N
//                   DOUBLET
//                         E
//                         R
//                         A
//                         B
//                         Y
//                         T
//                         ESSENCE
