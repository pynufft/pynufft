#!/bin/bash
for((i=8;i<=8;i++))
do sudo duo $i >> test.log
python ../pynufft_hsa.py > ./log/test_core_$i.txt
done

