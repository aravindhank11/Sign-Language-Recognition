#!/bin/bash

echo "User ID: "
read user
echo "Sign No.: "
read sign
echo "Trial No.: "
read trial
gwenview "<PATH>/DATASET/USER-"$user"/"$user-$sign-$trial".jpg"
python3 <PATH>/CODES/hand_sign.py $user $sign $trial
