#!/bin/bash
# for loop to register all the sources in the testdata directory
for file in testdata/*; do
  if [ -d "$file" ]; then
    pz reg --path $file --name $(basename $file)
  fi
done