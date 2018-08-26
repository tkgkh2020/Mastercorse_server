#!/bin/sh
<<EOF
VOBファイルからデインターレースしつつmpegファイルを作るスクリプト
DiscNameを変更して使用してください
EOF

DiscName="disc10"

files='/opt/pfw/dragon_ball_DVD/'"$DiscName"'_mp4/VOB/*VOB'

mkdir /opt/pfw/dragon_ball_DVD/"$DiscName"_mp4/deinterlaced

i=1
for file in $files; do
    echo "$i $file"
    ffmpeg -i "$file" -target pal-dvd -vcodec copy -acodec copy /opt/pfw/dragon_ball_DVD/"$DiscName"_mp4/deinterlaced/"$DiscName"_"$i"de.mpeg
    i=$((i+1))
done
