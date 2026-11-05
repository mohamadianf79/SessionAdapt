# Architecture spec (fill in; every answer becomes a function signature)

Goal: the model is ONE function:  score(prefix) -> vector of n_items scores.
`prefix` = the items (and timestamps) the user has clicked so far in the CURRENT session.

## A. Inputs available at inference time
- [ ] item ids of the prefix            (always)
- [ ] timestamps of the prefix          (needed for gaps / intensity)
- [ ] user id / earlier sessions        (Last.fm: yes; Yoochoose, Diginetica, RetailRocket: NO)
- [ ] clock time (hour, weekday)        (KEEP OUT of the architecture; it is the switch for the time hypothesis)

## B. Behaviour vector  b(window)   (layers 3-4)
Computed from a window of the prefix. Proposal: [log(len), unique_items/len, mean log-gap, last log-gap].
Your choice: ______________________   Dimension: ____   (use only item ids + inter-click gaps)

## C. Change detection  d(prefix)   (layer 5, theta)
Proposal: d = 1 - cos( b(older part of prefix), b(last w items) ).  window w = ____ ,  theta = ____ (tuned on VAL)
(Without user ids the "previous session" does not exist, so "old vs new" is inside the session.)

## D. Base next-item model  P(next | prefix)
Proposal: lag-Markov / SKNN first, then a small GRU/SASRec backbone.

## E. What happens when d > theta ?
Old design: fall back to popularity (measured: harmful).  Options to TEST as ablations:
 (a) do nothing, (b) use only the last item, (c) shrink Markov depth L, (d) fall back to cluster-level scores.

## F. Cluster layer (layers 6-7)
Cluster id c = KMeans(b(prefix)).  score += mu * P(item | cluster c).   mu tuned on VAL; mu=0 is the ablation.

## G. Final formula
score(j) = P_base(j | prefix) * [ rule from E if d>theta ]  +  mu * P(j | c)

Ablation table to report: base | +cluster | +change-rule | full | (time on/off in the time-hypothesis step).
