#!/bin/sh

for i in {1..2500};
do
    echo $i;
    iter=$((100*$i));
    python test.py --iter=$iter  >> evaluate_results.log;
done
