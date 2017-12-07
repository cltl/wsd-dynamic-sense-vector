
rm -rf higher_level_annotations && mkdir higher_level_annotations

sbatch convert_sense_annotations.job semcor synset
sbatch convert_sense_annotations.job mun synset
sbatch convert_sense_annotations.job semcor_mun synset

sbatch convert_sense_annotations_171.job semcor synset
sbatch convert_sense_annotations_171.job mun synset
sbatch convert_sense_annotations_171.job semcor_mun synset
