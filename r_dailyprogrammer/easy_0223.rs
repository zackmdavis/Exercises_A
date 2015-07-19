// garland words

//www.reddit.com/r/dailyprogrammer/comments
//3d4fwj/20150713_challenge_223_easy_garland_words/

use std::ascii::AsciiExt;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

fn garland_degree(word: &str) -> usize {
    // Unicode is weird and I don't feel like thinking about it right now
    if !word.is_ascii() {
        // so in a completely unprincipled exception, I'm going to be an
        // ASCIIhole and arbitrarily declare that words with letters
        // above 128 do not have the garland-nature
        return 0;
    }

    let wordlength = word.len();
    let mut max_garland_degree = 0;

    for i in 1..word.len() {
        if word[0..i] == word[wordlength-i..wordlength] {
            max_garland_degree = i;
        }
    }

    max_garland_degree
}

#[test]
fn test_garland_degree() {
    assert_eq!(0, garland_degree("programmer"));
    assert_eq!(1, garland_degree("ceramic"));
    assert_eq!(2, garland_degree("onion"));
    assert_eq!(4, garland_degree("alfalfa"));

    assert_eq!(0, garland_degree("álfalfa"));
}

fn main() {
    // let's find the longest garland word in /usr/share/dict/words
    let path = Path::new("/usr/share/dict/words");
    let usr_share_dict_words = File::open(path).ok()
        .expect("we couldn't open the file");
    let wordreader = BufReader::new(usr_share_dict_words);

    let mut best_garlands = vec!["".to_string()];
    let mut best_garland_degree = 0;
    for line in wordreader.lines() {
        let word = line.ok().expect("we couldn't read the line");
        let this_degree = garland_degree(&word);
        if this_degree > best_garland_degree {
            best_garland_degree = this_degree;
            best_garlands = vec![word];
        }
        else if this_degree == best_garland_degree {
            best_garlands.push(word);
        }
    }
    println!("best garland words: {:?}", best_garlands);
    println!("garland degree of the best words: {}", best_garland_degree);
}

// zmd@ExpectedReturn:~/Code/Exercises_A/r_dailyprogrammer$ ./easy_0223
// best garland words: ["abracadabra", "alfalfa", "beriberi", "entente", "hotshots"]
// garland degree of the best words: 4
