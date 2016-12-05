// https://www.reddit.com/r/dailyprogrammer/comments/4fc896/20160418_challenge_263_easy_calculating_shannon/

// Shannon entropy of a string; easy

package main

import (
	"fmt"
	"log"
	"math"
)

func tallyRunes(ourString string) map[rune]int {
	runeMap := make(map[rune]int)
	for _, c := range ourString {
		runeMap[c]++
	}
	return runeMap
}

func entropy(ourString string) float64 {
	runeMap := tallyRunes(ourString)
	theEntropy := 0.0
	for _, count := range runeMap {
		p := float64(count) / float64(len(ourString))
		theEntropy += -p * math.Log2(p)
	}
	return theEntropy
}

func main() {
	expectations := map[string]float64{
		"122333444455555666666777777788888888":     2.794208683,
		"563881467447538846567288767728553786":     2.794208683,
		"https://www.reddit.com/r/dailyprogrammer": 4.056198332,
		"int main(int argc, char *argv[])":         3.866729296,
	}
	ε := 0.0001
	for theString, theExpectedEntropy := range expectations {
		theTrueEntropy := entropy(theString)
		if math.Abs(theTrueEntropy-theExpectedEntropy) > ε {
			log.Fatalf("expected actual entropy %v bits should be about equal to expected %v", theTrueEntropy, theExpectedEntropy)
		}
	}
	fmt.Println("WINNING")
}
