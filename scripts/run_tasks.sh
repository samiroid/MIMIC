BASE_PATH="/Users/samir/Dev/projects/MIMIC_random_seeds/MIMIC"
# BASE_PATH="/media/data_1/samir/projects/MIMIC"
INPUT=$BASE_PATH"/DATA/input/"
FEATURES=$BASE_PATH"/DATA/features/"
OUTPUT=$BASE_PATH"/DATA/test_new/"
# CACHE=$BASE_PATH"/DATA/aftermath/"
DATA="tasks"
MODEL="CLINICALBERT-POOL"
MODEL="BERT-POOL"

python src/main.py -input_path $INPUT -dataset $DATA -feature_type $MODEL -feats_path $FEATURES \
                   -output_path $OUTPUT \
                   -metric "auroc" \
                   -mini_tasks
                   
                    

