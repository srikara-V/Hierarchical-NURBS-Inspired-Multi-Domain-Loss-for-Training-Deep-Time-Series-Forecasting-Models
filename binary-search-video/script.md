# Binary Search: One Formula, Every Flavor

Narration script (Khan-academy-style video). Voice: ElevenLabs 'Brian' (male), model eleven_turbo_v2_5.

## The universal template

```python
lo, hi = 0, n              # answer trapped in [lo, hi]
while lo < hi:
    mid = (lo + hi) // 2   # floor -> mid < hi, always
    if ask(mid):           # monotone yes/no question
        hi = mid           # mid might BE the first True: keep it
    else:
        lo = mid + 1       # mid is provably dead
return lo                  # the first True (n = 'no True exists')
```

Every flavor = four decisions: **SPACE** (candidates), **QUESTION** (monotone F->T), **BOUNDARY** (always first True; last True = first False - 1), **RETURN** (transform lo).

## The grid

| FLAVOR | SPACE | QUESTION ask(x) | RETURN |
|---|---|---|---|
| 1. sorted: find / insert | index 0..n | a[x] >= T   (> T right) | lo (+ check) |
| 2. rotated: min / find | index 0..n-1 | a[x] <= a[last] ? | a[lo]; then row 1 |
| 3. peak | index 0..n-1 | a[x] > a[x+1] ? | lo |
| 4. answer space: MIN | values lo..hi | feasible(x) ? | lo |
| 5. answer space: MAX | values lo..hi | can(x)?  T,T,T,F,F | first False - 1 |
| 6. real numbers | interval [lo, hi] | same, on reals | lo, after ~60 halvings |
| 7. 2D matrix | flat 0..m*n | A[x//n][x%n] >= T | row x//n, col x%n |

## Scene 1 - The reframe: binary search finds boundaries

**[s1b1]** Binary search. Most of us learn it as one narrow trick: find a number in a sorted array. But that picture is too small, and it's exactly why all the variants, rotated arrays, peaks, bananas, feel like separate pieces of magic. Today we rebuild binary search as one single formula, and then fill out a grid that covers pretty much every flavor you will ever meet.

**[s1b2]** Quick warm up. I'm thinking of a number from one to one hundred. You guess fifty, and I say: too low. With one question, you just deleted half the world. You guess seventy five. Too high. Half of what's left is gone again. Seven questions of this, and I am cornered. That feeling, cutting the world in half with one yes-or-no question, is the whole engine.

**[s1b3]** So here's the real picture. Forget arrays for a second. Picture a row of answers to some yes-or-no question, and suppose the row is sorted: all the No's come first, then all the Yes's. False, false, false, then true, true, true. We call a question like that monotone. It flips once, and it never flips back.

**[s1b4]** Binary search does exactly one job, ever. It finds that flip point, the first true, in about log n questions. That's it. That is the entire algorithm.

**[s1b5]** And every famous variant is just a different way of manufacturing this row. So here's our mission: one grid. For each flavor, we fill in the same cells. What is the search space? What is the yes-or-no question? And what do we return at the end? Let's fill it out, row by row.

## Scene 2 - THE FORMULA: four decisions, one template

**[s2b1]** Here is the formula, as four decisions. One: the space. The candidates, laid out in a row. Two: the question. A yes-or-no probe that goes false, false, false, true, true, true across that space. Three: the boundary. We always, always aim at the first true. And four: the return. What we hand back once we've caught it.

**[s2b2]** And one template executes all four. Two fingers, low and high, and one promise: the first true is always trapped between them. While they haven't met, probe the middle.

**[s2b3]** Say the probe comes back false. Then mid itself is a no, and everything left of mid is even deeper in no territory. That whole side is dead, including mid. So low jumps to mid plus one.

**[s2b4]** Now say a probe comes back true. Careful. Mid might be the very first true, so we must not skip past it. High lands right on mid, and mid stays alive inside the window.

**[s2b5]** False? Low hops past mid. True? High lands on mid. The window only shrinks, the flip point can never escape it, and when the fingers meet, they are standing on the first true. Return low.

**[s2b6]** Two tiny details make this bulletproof, and they are the details people get wrong. The middle uses floor division, so mid always lands strictly below high. And we only ever add one on the false side, because false cells are provably dead. Together: the window must shrink every round. No infinite loops, no off-by-one, ever.

**[s2b7]** One last guard. What if the row is all false, and no true exists? That's why high starts one past the end, at n, on an imaginary cell we simply declare to be true. If the search returns n, it's telling you: no real true exists. Remember that dashed cell. It does quiet work in every row of our grid.

## Scene 3 - Row 1: sorted arrays (LC 704, 35, 34)

**[s3b1]** Row one of the grid: the plain sorted array. The classic: find target seven, LeetCode seven-oh-four. So, where is our monotone question hiding? Here it is: is the value at index x, at least seven? Down in the small values: no, no, no. From the first seven onward: yes, yes, yes, forever. Sorted order is exactly what makes that question monotone. That is the real reason binary search works on sorted arrays at all.

**[s3b2]** Run the exact template from before. Probe the middle: at least seven? True. High lands on it. Probe again: three? False. Low hops past it. One more probe, and the fingers meet at index three, the first true. One final check: is the value there actually seven? It is. Found, at index three. If it weren't, we'd return minus one.

**[s3b3]** Now watch how cheap the next problem becomes. Search insert position, LeetCode thirty-five: where would five go? Change almost nothing. Ask: at least five? The first true is index three, again. And that IS the insert slot. Same space, same question shape, same boundary. Only the return story changed.

**[s3b4]** Duplicates. LeetCode thirty-four: first and last position of seven. Well, the first seven is just our first true for: at least seven. Index three. Already done.

**[s3b5]** For the LAST seven, do not write some new backwards loop. Flip the question instead: strictly greater than seven? That flips later, at index six. So the last seven sits one step before it: index five. First: at least. Last: strictly greater, minus one. That minus one trick is worth its own little cell in the grid.

**[s3b6]** And by the way, you already know these two searches by name. They are lower bound and upper bound. Bisect left and bisect right, in Python. Two predicates, one template, zero special cases.

**[s3b7]** Row one of the grid: space, indices zero to n. Question: value at x, at least the target. Or strictly greater, for right edges. Return: low, with a found-check, or minus one for last positions.

## Scene 4 - Row 2: rotated arrays (LC 153, 33, 154)

**[s4b1]** Row two: the rotated sorted array. Someone took a sorted array and spun it. Two sorted runs now, glued at a cliff. First job, LeetCode one fifty-three: find the minimum. The bottom of that cliff.

**[s4b2]** Where's the monotone question? Look at the very last element. Three. Now ask every index: is your value at most that last element? The high run, before the cliff: no, no, no. The low run, after it: yes, yes, yes. The rotation point is literally a false-to-true boundary. We just manufactured our row.

**[s4b3]** Same template, not a character changed. Probe the middle: twelve, at most three? False. Low hops right. Probe: zero? True. High lands on it. Probe: minus four? True again. The fingers meet at index three. The minimum.

**[s4b4]** And now, searching a rotated array, LeetCode thirty-three, is just composition. Step one: find the cliff, exactly like we just did. Step two: the cliff splits the array into two perfectly sorted halves, and comparing your target to the last element tells you which half it lives in. Step three: run row one's classic search inside that half. Two binary searches, both from the same template. Still logarithmic.

**[s4b5]** Two footnotes for this row. Always compare against the LAST element; comparing mid to low breaks on arrays that were never rotated. And if duplicates are allowed, LeetCode one fifty-four, equal values can hide the cliff, and the worst case genuinely degrades to linear. Know that trade before an interview.

**[s4b6]** Row two, into the grid.

## Scene 5 - Row 3: peak (LC 162)

**[s5b1]** Row three: the peak, LeetCode one sixty-two. Find any element bigger than both neighbors. And look at this array. Nothing is sorted. Surely binary search is off the table. Right?

**[s5b2]** Ask this at every index: are we going downhill at x? Is the value at x greater than the value right after it? While we climb: no, no. Past the summit: yes, yes. And off the right edge we pretend the array falls to minus infinity, so the last index is always a yes. There's our dashed sentinel cell again, doing real work.

**[s5b3]** Now, one honest subtlety, because this is where the deep idea lives. With many hills, that downhill question can flicker. Yes, no, yes. Not monotone! But watch what the template actually needs. Probe mid. Going uphill? Then the next step is higher than mid, and somewhere to the right a summit must exist. Slide low past mid. Going downhill? Then a summit exists at mid or to its left. Pull high onto mid. Either way, the window still traps a peak. That is the real contract of binary search: never lose the answer. Monotone rows are just its most common costume.

**[s5b4]** Run it. Middle of the row: five against four. Downhill. True. High lands there. One more probe, uphill, low slides past. The fingers meet at index two. Value five: a peak, found in log time, in an unsorted array.

**[s5b5]** Row three of the grid. Space: indices. Question: downhill here? Return: low. Plus one idea worth its own subcell: the invariant is the law. Monotonicity is just the costume.

**[s5b6]** Into the grid.

## Scene 6 - Rows 4-5: binary search on the ANSWER (LC 875, 1011, 410, 1482, 1552)

**[s6b1]** Row four. And this is the flavor that unlocks hundreds of hard problems. So far, the space was indices into an array somebody handed us. Now the space becomes the possible ANSWERS themselves. Koko loves bananas, LeetCode eight seventy-five. Four piles. Eight hours before the guards come back. Koko picks one eating speed, k bananas per hour, and each hour she eats from a single pile.

**[s6b2]** Here's the key move: fix a speed, and just check it. Try speed four. The pile of three takes one hour. Six takes two. Seven takes two. Eleven takes three. Total: eight hours. She makes it. Exactly.

**[s6b3]** Now sweep every candidate speed, one through eleven, and ask each one: can Koko finish in time at this speed? The slow speeds: no, no, no. Then from four onward: yes, yes, yes. Of course it's monotone. Eating faster never hurts. We just built our row out of thin air. THAT is binary search on the answer.

**[s6b4]** So the formula instantiates like this. Space: speeds one up to the biggest pile. Question: does the total time at speed x fit in eight hours? That check is a five-line helper. Boundary: first yes. Return: low. The template does not change by one character. Only the question is new.

**[s6b5]** Run it on the speeds. Probe six: fits. Probe three: too slow. Probe five: fits. Probe four: fits. Low meets high at four. Koko eats at speed four.

**[s6b6]** And here is why this row is worth hundreds of problems. Watch how little changes. Ship packages within D days, LeetCode ten-eleven. The answer is a ship capacity. The space runs from the heaviest single package up to everything in one trip. The question: at capacity x, do the days fit? One greedy pass counts the trips. Same row.

**[s6b7]** Split an array into k chunks, minimizing the largest chunk sum. LeetCode four-ten. The answer is that largest sum. The question: capped at x, do we need at most k chunks? Same row. Minimum days for m bouquets, LeetCode fourteen eighty-two: the space is days, the question: enough bouquets by day x? Same row. Different story, different feasibility check. Identical search.

**[s6b8]** One twist left: maximize instead of minimize. Magnetic force, LeetCode fifteen fifty-two. Place three balls in baskets, and make the smallest gap as LARGE as possible. Ask: can we place them with gap at least x? Small gaps: yes, yes, yes. Then it breaks: no, no, no. The row flipped! True first, then false. And still, no new loop. Search for the first no, and step back one. Maximize equals first false, minus one.

**[s6b9]** Rows four and five, into the grid. Minimize: first yes. Maximize: first no, minus one. If you keep only one row of this whole grid, keep this one. Binary search the answer, feasibility check inside.

## Scene 7 - Rows 6-7: reals (sqrt) and 2D matrix (LC 74)

**[s7b1]** Two quick rows finish the grid. First: continuous answers. Square root of two. The space is the interval from one to two, and the question, is x squared at least two, is as monotone as ever. Only one thing dies: mid plus one. There is no next number on a continuous line. So low simply becomes mid, and instead of waiting for fingers to meet, we just halve a fixed number of times. Sixty halvings shrinks the interval below ten to the minus eighteenth. Sharper than any double. Run sixty, return low.

**[s7b2]** Second: LeetCode seventy-four. A matrix where each row is sorted, and each row starts after the last one ends. Read it row by row: it is one long sorted array wearing a costume. Index x maps to row x divided by n, column x mod n. Flatten the picture, and it is literally row one of our grid again.

**[s7b3]** Rows six and seven. The grid is full.

## Scene 8 - The Grid: the full map + the 4-question checklist

**[s8b1]** And there it is. The whole map. Seven rows, one template. Read down the columns. The space is always one of three things: indices, candidate answers, or an interval. The question is always engineered to split no from yes. The boundary is always the first true; even last-true is just first-false minus one. And the return is low, wearing three different costumes.

**[s8b2]** So the next time a problem smells like binary search, don't start typing a while loop. Fill in a row of this grid. What is my space? What is my yes-or-no question, and does it flip exactly once? First true, or last true? And what do I return? Four cells, and the code writes itself. The same six lines. Every single time.

**[s8b3]** That's binary search. Not a trick for sorted arrays. A formula for finding boundaries. Happy searching.
