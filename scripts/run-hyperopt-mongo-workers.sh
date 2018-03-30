#!/bin/sh

for ((i=1;i<=$1;i++)); do
    hyperopt-mongo-worker --mongo=$2 --poll-interval=0.1 >> log${i}.log 2>&1 &
done

wait
echo "All workers complete!!!"
