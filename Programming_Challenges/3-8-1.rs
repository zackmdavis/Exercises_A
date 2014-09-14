use std::collections::hashmap::HashMap;

fn erroneous_typing(intended: String) -> String {
    let top_row = vec!['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', '[', ']'];
    let home_row = vec!['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';', '\''];
    let bottom_row = vec!['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', '/'];
    let spacebar = vec![' ', ' '];

    let mut keymap = HashMap::new();

    let rows = [&top_row, &home_row, &bottom_row, &spacebar];
    for row in rows.iter() {
        for i in range(0, row.len()-1) {
            keymap.insert(*row.get(i+1), *row.get(i));
        }
    }

    let mut written: String = "".to_string();
    for c in intended.as_slice().chars() {
        written.push_char(*keymap.get(&c));
    }
    written
}


#[test]
fn test_sample_output() {
    let result: String = erroneous_typing("O S, GOMR YPFSU/".to_string());
    println!("{}", result);
    assert!(result == "I AM FINE TODAY.".to_string());
}
