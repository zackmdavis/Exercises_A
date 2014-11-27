// "Write an algorithm such that if an element in an MxN matrix is 0,
// its entire row and column are set to 0.

// ... except I'm retarded and not used to programming lanugages where
// you're actually expected to understand how computers work, so what
// happens if I restrict it to "4 x 4 matrix"? Can I fucking do
// something in that case?

fn zero_row(matrix: &mut [[int, ..4], ..4], row: uint) {
    let row_length = matrix[0].len();
    for j in range(0u, row_length) {
        matrix[row][j] = 0;
    }
}

fn zero_col(matrix: &mut [[int, ..4], ..4], col: uint) {
    let col_length = matrix.len();
    for i in range(0u, col_length) {
        matrix[i][col] = 0;
    }
}

#[test]
fn test_zero_row_and_col() {
    let mut my_matrix: [[int, ..4], ..4] = [[1i, 2, 3, 4],
                                            [1, 2, 3, 4],
                                            [1, 2, 3, 4],
                                            [1, 2, 3, 4]];
    // ffffff "mismatched types: expected `&mut [[int, .. 4], .. 4]`
    // but found `&[[int, .. 4], .. 4]` (values differ in mutability)"
    // but I _said_ `my_matrix` is mutable, and I ... need to do more
    // reading before trying to write the simplest of programs??
    zero_row(&my_matrix, 1);
    zero_col(&my_matrix, 2);
    let mut expectation = [[1i, 2, 0, 4],
                           [0, 0, 0, 0],
                           [1, 2, 0, 4],
                           [1, 2, 0, 4]];
    for i in range(0u, 3) {
        for j in range(0u, 3) {
            assert!(expectation[i][j] == my_matrix[i][j]);
        }
    }
}
