U2V_PATH="/Users/samir/Dev/projects/user2vec_torch/user2vec"
BASE_PATH="/Users/samir/Dev/projects/MIMIC/MIMIC"
WORD_EMBEDDINGS="/Users/samir/Dev/resources/embeddings/word_embeddings.txt"

U2V_PATH="/home/silvio/home/projects/user2vec_torch/user2vec/"
BASE_PATH="/media/data_1/samir/projects/MIMIC"
WORD_EMBEDDINGS="/media/data_1/samir/resources/word_embeddings.txt"

INPUT=$BASE_PATH"/DATA/input/patients.csv"
OUTPUT_PATH=$BASE_PATH"/DATA/features/u2v/"

python src/get_u2v_data.py -input_file $INPUT -output_folder $OUTPUT_PATH 
CORPUS=$OUTPUT_PATH"/u2v/users.txt"
python $U2V_PATH"run.py" -input $CORPUS -emb $WORD_EMBEDDINGS -output $OUTPUT_PATH \
                        -lr 0.001 \
                        -epochs 20 \
                        -neg_samples 2 \
                        -margin 5 \
			-device cpu 
			