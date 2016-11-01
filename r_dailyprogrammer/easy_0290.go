// https://www.reddit.com/r/dailyprogrammer/comments/5aemnn/20161031_challenge_290_easy_kaprekar_numbers/

// "a Kaprekar number for a given base is a non-negative integer, the
// representation of whose square in that base can be split into two parts that
// add up to the original number again. For instance, 45 is a Kaprekar number,
// because 452 = 2025 and 20+25 = 45. [...] For the main challenge we'll only
// focus on base 10 numbers [...] Your program will receive two integers per line
// telling you the start and end of the range to scan, inclusively [...]  Your
// program should emit the Kaprekar numbers in that range

package main

import (
	"fmt"
	"strconv"
)

type twoStringTuple struct { // this programming language sucks (I just wanted a fucking tuple)
	left  string
	right string
}

func nonemptyStringPartitions(s string) (partitions []twoStringTuple) {
	partitions = make([]twoStringTuple, 0, len(s)-1)
	for i := 1; i < len(s); i++ {
		partitions = append(partitions, twoStringTuple{s[:i], s[i:]})
	}
	return partitions
}

func isKaprekar(n int) bool {
	represeNTation := fmt.Sprintf("%v", n*n)
	for _, partition := range nonemptyStringPartitions(represeNTation) {
		left, leftErr := strconv.ParseInt(partition.left, 10, 0)
		right, rightErr := strconv.ParseInt(partition.right, 10, 0)
		if leftErr != nil || rightErr != nil {
			panic("this should not happen")
		}
		if left+right == int64(n) { // this programming language sucks (ParseInt returns int64)
			return true
		}
	}
	return false
}

func main() {
	// exercise prompt says answer should be "9 45 55 99"
	// actually prints "9 10 45 55 99" (unexpected 10)
	for i := 2; i < 100; i++ {
		if isKaprekar(i) {
			fmt.Printf("%v ", i)
		}
	}

	fmt.Println()

	// exercise prompt says answer should be "297 703 999 2223 2728 4879 5050 5292 7272 7777"
	// actually prints "297 703 999 1000 2223 2728 4879 4950 5050 5292 7272 7777" (unexpected 1000)
	for i := 101; i < 9000; i++ {
		if isKaprekar(i) {
			fmt.Printf("%v ", i)
		}
	}

	// whoopse

	fmt.Println()
}
