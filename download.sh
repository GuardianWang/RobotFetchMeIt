gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

echo "Downloading text features"
gdrive_download 1WPgFuQ4W4ejW5ttG6P0vsvGADfnmAp_U subdataset.zip

echo "Downloading weight..."
gdrive_download 1cEWTQo8jXz5krJ3clViH70rDFnR4-8Sh checkpoint.pth