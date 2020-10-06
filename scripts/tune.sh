BASE_PATH="/Users/samir/Dev/projects/MIMIC/MIMIC"
# BASE_PATH="/media/data_1/samir/projects/MIMIC"
INPUT=$BASE_PATH"/DATA/input/"
FEATURES=$BASE_PATH"/DATA/features/"
OUTPUT=$BASE_PATH"/DATA/sensitivity_results/"
CACHE=$BASE_PATH"/DATA/sensitivity_processed/"
DATA=$1
MODEL=$2

python src/main.py -input_path $INPUT -dataset $DATA -model $MODEL -feats_path $FEATURES \
                   -output_path $OUTPUT \
                   -cache_path $CACHE -mini_tasks -tune "sensitivity"
                   
                    

