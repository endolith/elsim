This branch is just for the core collapse finder animations and images

A lot of it is vibe-coded with Cursor and I haven't checked that it's actually correct.

Eventually I'll clean it up and merge back into `master`.

# Core collapse

The same randomly-generated election is used for all 4 animations, with voters and candidates both normally distributed, but candidate distribution 1/2 the width of the voter distribution.  Ran a bunch of random elections until it found an example of this "core collapse" worst-case center-squeeze scenario:

- Under IRV, vote-splitting causes the most-representative candidate to be eliminated first, then the second-best eliminated second, and so on until only the worst two are left in the final round and the second-worst wins.  (This scenario is [described by Nanson in 1882](https://archive.org/details/transactionsproc1719roya/page/207/mode/1up).  This exact "core collapse" successive-elimination scenario becomes increasingly unlikely as the number of candidates increases, of course, but Condorcet failures in general become more likely, as measured by [Condorcet efficiency](https://en.wikipedia.org/wiki/Condorcet_efficiency).)

- Under TVR, on the other hand, all voter preferences are included, so the least-representative candidates are eliminated first, transferring their ballots inwards such that support converges to the consensus candidate in the middle of the electorate (who has the highest approval rating and is the Condorcet winner who beats all other candidates head-to-head).


## Instant-Runoff Voting = Ranked Choice Voting, dark background

![collapse_2d_irv dark](./results/collapse_2d_irv%20dark.gif)

## Baldwin's method = Total Vote Runoff, dark background

![collapse_2d_tvr dark](./results/collapse_2d_tvr%20dark.gif)

## Instant-Runoff Voting = Ranked Choice Voting, light background

![collapse_2d_irv light](./results/collapse_2d_irv%20light.gif)

## Baldwin's method = Total Vote Runoff, light background

![collapse_2d_tvr light](./results/collapse_2d_tvr%20light.gif)

Nanson, E. J. (1882). ["Methods of election"](https://archive.org/details/transactionsproc1719roya/page/197). *Transactions and Proceedings of the Royal Society of Victoria*. **19**: 197–240.

> To illustrate fully the difference between the two methods and the defects of each, suppose that there are several candidates, A, B, C, D, . . P, Q, R, and that in the opinion of the electors each candidate is better than each of the candidates who follow him in the above list,so that A is clearly the best, B the second best, and so on, R being the worst. Then on the single vote method R may win; on Ware’s method A, B,C, D, . . P, may be excluded one after another on the successive scrutinies, and at the final scrutiny the contest will be between Q and R, and Q, of course, wins, since we have supposed him better than R in the opinion of the electors, Thus the single vote method may return the worst of all the candidates ; and although Ware’s method cannot return the worst, it may return the next worst.

