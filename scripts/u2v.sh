U2V_PATH="/Users/samir/Dev/projects/user2vec_torch/user2vec"
BASE_PATH="/Users/samir/Dev/projects/MIMIC/MIMIC"
WORD_EMBEDDINGS="/Users/samir/Dev/resources/embeddings/word_embeddings.txt"

# U2V_PATH="/Users/samir/Dev/projects/user2vec_torch/user2vec"
# BASE_PATH="/media/data_1/samir/projects/MIMIC"
# WORD_EMBEDDINGS="/Users/samir/Dev/resources/embeddings/word_embeddings.txt"

INPUT=$BASE_PATH"/DATA/input/patients.csv"
OUTPUT_PATH=$BASE_PATH"/DATA/features/u2v/"

python src/get_u2v_data.py -input_file $INPUT -output_folder $OUTPUT_PATH 

# python $U2V_PATH/build.py -input $CORPUS -emb $WORD_EMBEDDINGS -output $PKL_PATH

# python $U2V_PATH/train.py -input $PKL_PATH  -output $OUTPUT_PATH
                   
                   
                    

