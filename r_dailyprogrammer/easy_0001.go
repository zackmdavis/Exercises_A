package main

// might be writing Go at my dayjob soon, better do some simple
// exercises now to get oriented with the basics

// http://www.reddit.com/r/dailyprogrammer/comments/pih8x/easy_challenge_1/
// got started with help from http://stackoverflow.com/q/20895552

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

func main() {
	reader := bufio.NewReader(os.Stdin)
	askables := [3]string{"name", "age", "username"}
	answers := make([]string, 3)
	for index, askable := range askables {
		fmt.Printf("What is your %v? >> ", askable)
		answer, _ := reader.ReadString('\n')
		answer = strings.TrimSpace(answer)
		answers[index] = answer
	}
	for index, askable := range(askables) {
		fmt.Printf("You claimed that your %v is %v.\n", askable, answers[index])
	}
}
