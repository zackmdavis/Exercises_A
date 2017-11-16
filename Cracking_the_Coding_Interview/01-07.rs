// "Write an algorithm such that if an element in an MxN matrix is 0,
// its entire row and column are set to 0.

// TODO: finish

fn zero_row(matrix: &mut [[isize; 4]; 4], row: usize) {
    let row_length = matrix[0].len();
    for j in 0..row_length {
        matrix[row][j] = 0;
    }
}

fn zero_col(matrix: &mut [[isize; 4]; 4], col: usize) {
    let col_length = matrix.len();
    for i in 0..col_length {
        matrix[i][col] = 0;
    }
}

#[test]
fn test_zero_row_and_col() {
    let mut my_matrix: [[isize; 4]; 4] = [[1, 2, 3, 4],
                                          [1, 2, 3, 4],
                                          [1, 2, 3, 4],
                                          [1, 2, 3, 4]];
    zero_row(&mut my_matrix, 1);
    zero_col(&mut my_matrix, 2);
    let expectation = [[1, 2, 0, 4],
                       [0, 0, 0, 0],
                       [1, 2, 0, 4],
                       [1, 2, 0, 4]];
    for i in 0..3 {
        for j in 0..3 {
            assert!(expectation[i][j] == my_matrix[i][j]);
        }
    }
}
