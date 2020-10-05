BASE_PATH="/Users/samir/Dev/projects/MIMIC/"
INPUT=$BASE_PATH"MIMIC/DATA/input/"
OUTPUT=$BASE_PATH"MIMIC/DATA/results/"
CACHE=$BASE_PATH"MIMIC/DATA/processed/"
DATA=$1
MODEL=$2

python src/main.py -input_path $INPUT -dataset $DATA -model $MODEL -output_path $OUTPUT \
                    -cache_path $CACHE
                   
                    

