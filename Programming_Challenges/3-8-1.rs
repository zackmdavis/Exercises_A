use std::hashmap::HashMap;

fn erroneous_typing(intended: ~str) -> ~str {
    let top_row = ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', '[', ']'];
    let home_row = ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';', '\''];
    let bottom_row = ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/'];

    let mut keymap = HashMap::new();

    // this is awful but I don't really understand Rust's type
    // system and don't know how to iterate over a vector of vectors
    for i in range(0, top_row.len()-1) {
        keymap.insert(top_row[i+1], top_row[i]);
    }
    for i in range(0, home_row.len()-1) {
        keymap.insert(home_row[i+1], home_row[i]);
    }
    for i in range(0, bottom_row.len()-1) {
        keymap.insert(bottom_row[i+1], bottom_row[i]);
    }

    keymap.insert(' ', ' ');

    let mut written = ~"";
    for c in intended.chars() {
        written.push_char(*keymap.get(&c));
    }
    written
}


#[test]
fn test_sample_output() {
    let result: ~str = erroneous_typing(~"O S, GOMR YPFSU/");
    println!("{}", result);
    assert!(result == ~"I AM FINE TODAY.");
}
