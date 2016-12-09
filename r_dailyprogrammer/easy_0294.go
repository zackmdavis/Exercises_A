package main

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
)

// https://www.reddit.com/r/dailyprogrammer/comments/5go843/20161205_challenge_294_easy_rack_management_1/

// Today's challenge is inspired by the board game Scrabble. Given a set of 7 letter tiles and a word, determine whether you can make the given word using the given tiles.

// Optional Bonus 1: Handle blank tiles (represented by "?"). These are "wild card" tiles that can stand in for any single letter.

// Optional Bonus 2: Given a set of up to 20 letter tiles, determine the longest word from [the enable1 English word list](https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/dotnetperls-controls/enable1.txt) that can be formed using the tiles.

type rack map[rune]uint

func buildRack(rackview string) rack {
	ourRack := make(rack)
	for _, r := range rackview {
		ourRack[r]++
	}
	return ourRack
}

func spellable(rackview string, target string) bool {
	ourRack := buildRack(rackview)
	for _, r := range target {
		supply, ok := ourRack[r]
		if !ok || supply == 0 {
			blankSupply, blankOk := ourRack['?']
			if !blankOk || blankSupply == 0 {
				return false
			} else {
				ourRack['?']--
			}
		} else {
			ourRack[r]--
		}
	}
	return true
}

var wordlistPath string = "/tmp/wordlist.txt"

// just doing what the prompt says, even though I usually just reach for
// /usr/share/dict/words (although this is admittedly more expansive)
var wordlistSourceURL string = "https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/dotnetperls-controls/enable1.txt"

func enableWordlistAvailability() {
	// Thanks to http://stackoverflow.com/a/12518877 and
	// http://stackoverflow.com/a/33853856 for quick-reference on how to do obvious
	// things in this shitty language
	if _, err := os.Stat(wordlistPath); os.IsNotExist(err) {
		wordlistFile, err := os.Create(wordlistPath)
		if err != nil {
			panic(fmt.Sprintf("couldn't enable wordlist: %v", err))
		}
		defer wordlistFile.Close()

		response, err := http.Get(wordlistSourceURL)
		if err != nil {
			panic(fmt.Sprintf("couldn't enable wordlist: %v", err))
		}
		defer response.Body.Close()

		_, err = io.Copy(wordlistFile, response.Body)
		if err != nil {
			panic(fmt.Sprintf("couldn't enable wordlist: %v", err))
		}
	}
}

func compileWordlist() []string {
	wordlist := make([]string, 0, 172819)
	wordlistFile, err := os.Open(wordlistPath)
	if err != nil {
		panic(fmt.Sprintf("couldn't compile wordlist: %v", err))
	}
	wordlistReader := bufio.NewReader(wordlistFile)
	for {
		var (
			isPrefix  bool = true
			nextWord  string
			nextChunk []byte
			err       error
		)
		for isPrefix {
			nextChunk, isPrefix, err = wordlistReader.ReadLine()
			nextWord += string(nextChunk)
			if err != nil {
				goto DONE
			}
		}
		wordlist = append(wordlist, nextWord)
	}
DONE:
	return wordlist
}

func longestSpellable(wordlist []string, rackview string) string {
	var longest string
	for _, candidate := range wordlist {
		if len(candidate) > len(longest) && spellable(rackview, candidate) {
			longest = candidate
		}
	}
	return longest
}

type expectation struct {
	rackview  string
	target    string
	spellable bool
}

func main() {
	expectations := []expectation{
		expectation{rackview: "ladilmy", target: "daily", spellable: true},
		expectation{rackview: "eerriin", target: "eerie", spellable: false},
		expectation{rackview: "orrpgma", target: "program", spellable: true},
		expectation{rackview: "orppgma", target: "program", spellable: false},
		expectation{rackview: "piizza?", target: "pizzazz", spellable: false},
		expectation{rackview: "pizza??", target: "pizzazz", spellable: true},
		expectation{rackview: "a??????", target: "program", spellable: true},
		expectation{rackview: "b??????", target: "program", spellable: false},
	}
	for _, e := range expectations {
		if spellable(e.rackview, e.target) != e.spellable {
			var negatoryMaybe string
			if !e.spellable {
				negatoryMaybe = "not "
			}
			log.Fatalf("expected %vto be able to spell %q with %q", negatoryMaybe, e.target, e.rackview)
		}
	}
	log.Println("WINNING (through basic challenge and Optional Bonus 1)")

	enableWordlistAvailability()
	wordlist := compileWordlist()

	lengthExtensionExpectations := []expectation{
		expectation{rackview: "dcthoyueorza", target: "coauthored"},
		expectation{rackview: "uruqrnytrois", target: "turquois"},
		expectation{rackview: "rryqeiaegicgeo??", target: "greengrocery"},
		expectation{rackview: "udosjanyuiuebr??", target: "subordinately"},
		expectation{rackview: "vaakojeaietg????????", target: "ovolactovegetarian"},
	}
	for _, e := range lengthExtensionExpectations {
		if theLongest := longestSpellable(wordlist, e.rackview); theLongest != e.target {
			log.Fatalf("expected longest spellable with %q to be %q, got %q", e.rackview, e.target, theLongest)
		}
	}
	log.Println("WINNING (through Optional Bonus 2)")
}
