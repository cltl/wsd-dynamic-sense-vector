inp_path=output/gigaword.2018-05-04-149cde7.txt.gz
out_path=output/gigaword-deduplicated.`python3 version.py`.txt.gz

#num_lines=`zcat $inp_path | wc -l`
num_lines=175771837

zcat $inp_path | tqdm --total $num_lines --unit line --miniters 100000 | \
	python3 filter-dumb-sentences.py |
	sort -T output/ | uniq | gzip > $out_path && \
	echo "Result written to $out_path"
