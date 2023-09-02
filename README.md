# LexDE (DE with Lexicase Selection)

This repository implements the LexDE algorithm from the paper
"**Optimization of subsurface models with multiple criteria using
Lexicase Selection**", Yifan He, Claus Aranha, Antony Hallam, Romain
Chassagne, Operation Research Perspectives, [DOI:
10.1016/j.orp.2022.100237](https://doi.org/10.1016/j.orp.2022.100237), 2022.05.

LexDE is a Differential Evolution for multi-objective (or rather,
multi-task) problems, where the parent selection mechanism was
replaced with Lexicase selection. 

The basic idea is that by using Lexicase selection, we try to obtain
solutions that are good in all functions of the multi-task problem,
while still having a degree of diversity.

## How to use

Please see an usage example in the "example.py" file.

For now, each fitness must be produced by a different function. We
will improve this in the future.

Requires numpy, csv and tdqm. Tested on python 3.9.6, but any version
should do in principle.

## Files

- README.md: This file
- TODO.md: Outstanding tasks
- lexde.py: lexde implementation
- example.py: example usage
- lexde_orig.py: original implementation by Yifan He
