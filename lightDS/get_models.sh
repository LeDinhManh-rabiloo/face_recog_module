#!/usr/bin/env bash

PTH_FILENAME=light_DSFD.pth

if [ ! -d "./weights" ]; then
    mkdir -p ./weights;
fi

cd weights

# Download yoloface models
echo "*** Downloading the trained models..."

wget --load-cookies /tmp/cookies.txt -r "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1wZkVZBr6JphK86dnHWI7F93xFQ8EvUiq' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1wZkVZBr6JphK86dnHWI7F93xFQ8EvUiq" -O $PTH_FILENAME && rm -rf /tmp/cookies.txt


echo "*** All done!!!"
