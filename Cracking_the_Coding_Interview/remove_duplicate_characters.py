
# imitating the text solution to a problem in an earlier edition of
# the text

# pretend it's a C-style string, okay??

# no extra array allocation is the interesting part obvs

def remove_duplicates(our_str):
    if our_str is None:
        return
    length = len(our_str)
    if length < 2:
        return

    tail = 1

    # We trivially know the first character is unique. We use the
    # beginning of the array up to `tail` to store characters that we
    # know are unique; as we find more yet-unseen characters, we write
    # them into the "unique zone" delimited by `tail` (and increment
    # `tail` to "make room" in the "unique zone").

    for i in range(1, length-1):  # exclude the trailing null
        for j in range(tail+1):
            if our_str[j] == our_str[i]:
                break
        if j == tail:
            our_str[tail] = our_str[i]
            tail += 1

    our_str[tail] = 0
    print(our_str)
