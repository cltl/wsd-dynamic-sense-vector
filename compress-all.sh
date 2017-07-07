rm ../lstm-wsd.tar*
tar --exclude nohup.out --exclude '*gigaword.txt*' --exclude '*Worker*.txt.gz' \
		--exclude 'slurm-*.out' -cvf ../lstm-wsd.tar . | tee ../lstm-wsd.log