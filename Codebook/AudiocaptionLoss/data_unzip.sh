cd data
mkdir waveforms
mv train* waveforms
mv val.zip waveforms
mv test.zip waveforms
cd waveforms
zip -s 0 train.zip --out new_train.zip
unzip val.zip
unzip test.zip
unzip new_train.zip
rm train.z*
rm test.zip
rm val.zip
rm new_train.zip