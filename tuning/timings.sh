#!/bin/bash

# Prints a single number: the RHS solve time for the tuned simulation.
# This is the exact value the tuner optimizes for (the second column of the
# 'ODESolvers::Solve::rhs' row that the old timings.sh table reported).

test=CottonmouthZ4c4m

grep 'ODESolvers::Solve::rhs' /work/$USER/simulations/$test/output-0000/$test.out | tail -1 | awk '{print $2}'
