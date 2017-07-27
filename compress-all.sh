echo -n 'Started: ' && date
rm ../lstm-wsd.tar*
tar --exclude nohup.out --exclude '*gigaword*.txt*' --exclude '*Worker*.txt.gz' \
		--exclude 'slurm-*.out' --exclude '*.data-00000-of-00001' \
		-cvf ../lstm-wsd.tar . | tee ../lstm-wsd.log
echo -n 'Finished: ' && date