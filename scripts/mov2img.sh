#!/bin/sh
<<EOF
mpegなどの動画ファイルから全フレームを切りだしてjpgで保存
DiscNameを書き換えて使ってください
EOF

DiscName="disc10"

files="/opt/pfw/dragon_ball_DVD/""$DiscName""_mp4/deinterlaced/*mpeg"

i=1
for file in $files; do
    echo "$i $file"
    #mkdir /opt/pfw/dragon_ball_img/all_ndarray/"$DiscName"/story"$i"
    #ffmpeg -i "$file" -s 360:256 -r 25 -f image2 -qscale 1 /opt/pfw/dragon_ball_img/all_ndarray/"$DiscName"/story"$i"/img_%05d.jpg
    mkdir /opt/pfw/dragon_ball_img/"$DiscName"/story"$i"
    ffmpeg -i "$file" -s 360:256 -r 25 -f image2 -qscale 1 /opt/pfw/dragon_ball_img/"$DiscName"/story"$i"/img_%05d.jpg
    i=$((i+1))
done
