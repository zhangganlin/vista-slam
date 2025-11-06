#!/bin/bash

dest="datasets/7scenes"
mkdir -p "$dest"

urls=(
    "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/chess.zip"
    "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/fire.zip"
    "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/heads.zip"
    "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/office.zip"
    "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/pumpkin.zip"
    "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/redkitchen.zip"
    "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/stairs.zip"
)

for url in "${urls[@]}"; do
    file_name=$(basename "$url")
    echo "Downloading $file_name..."
    wget "$url" -O "$dest/$file_name"
    echo "Unzipping $file_name..."
    unzip "$dest/$file_name" -d "$dest"
    unzip "$dest/${file_name%.*}/seq-01" -d "$dest/${file_name%.*}"
done