import transformers

#this is the max number of tokens in the sentence
MAX_LEN = 512

#batch size is small because model is huge
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4

#let's train for a maximun of 10 epochs
EPOCHS = 10

#this is where we want to save the model
MODEL_PATH = 'model.bin'

## training fi;e
TRAINING_FILE = 'IMDB Dataset.csv'

#define the tokenizer we use for the model from huggingface's transformers
TOKENIZER = transformers.BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case = True)
