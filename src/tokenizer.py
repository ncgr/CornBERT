# created following to the Hugging Face Tokenizers Quicktour:
# https://huggingface.co/docs/tokenizers/quicktour

from itertools import chain

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from transformers import RobertaTokenizerFast
from transformers import PreTrainedTokenizerFast

from fasta import fastaToSequences


TOKENIZER_CONFIG_FILE = "cornbert-tokenizer.json"
VOCAB_SIZE = 5000
PAD_TO_LENGTH = 256

# Trains a (byte-level?) BPE tokenizer on an iterable of sequences using the following special tokens:
# "[UNK]" - unknown character
# "[CLS]" - classifier token
# "[SEP]" - separator
# "[PAD]" - adds padding to token vectors?
# "[MASK]" - used to mask tokens during training?
def trainTokenizerOnSequences(sequence_iterable, vocab_size=VOCAB_SIZE, pad_to_length=PAD_TO_LENGTH, output_config_file=TOKENIZER_CONFIG_FILE):
    unknown_token="[UNK]"
    pad_token = "[PAD]"
    special_tokens = [unknown_token, "[CLS]", "[SEP]", pad_token, "[MASK]"]

    # instantiate a BPE tokeinzer model
    tokenizer = Tokenizer(BPE(unk_token=unknown_token))
    # use a pre-tokenizer to split inputs into words using whitespace as the word separator
    # TODO: do separate string in iterator still need whitespace to separate?
    tokenizer.pre_tokenizer = Whitespace()
    # pad the generated outputs
    if pad_to_length:
        pad_id = special_tokens.index(pad_token)
        tokenizer.enable_padding(direction='left', pad_id=pad_id, pad_token=pad_token, length=pad_to_length)

    # instantiate a trainer for our BPE tokenizer
    trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=vocab_size)

    # train the tokenizer on the sequences
    tokenizer.train_from_iterator(sequence_iterable, trainer)

    # save the tokenizer to a single file that contains its configuration and the vocabulary
    tokenizer.save(output_config_file)


# Same as trainTokenizerOnSequences but loads sequences from the given FASTA files.
def trainTokenizerOnFastas(fasta_files, *args, **kwargs):
    # create an iterable of sequences from the input FASTAs
    sequence_generators = [fastaToSequences(fasta) for fasta in fasta_files]
    sequence_iterator = chain.from_iterable(sequence_generators)
    # train the tokenizer
    trainTokenizerOnSequences(sequence_iterator, *args, **kwargs)


def loadCornbertTokenizer(config_file=TOKENIZER_CONFIG_FILE):
    # reload the tokenizer from its tokenizer file
    return RobertaTokenizerFast(tokenizer_file=config_file)


if __name__ == '__main__':
    import sys

    # get the list of FASTA files
    if len(sys.argv) == 1:
        exit(sys.argv[0])
    fasta_files = sys.argv[1:]

    # create the tokenizer
    trainTokenizerOnFastas(fasta_files)

    # test the tokenizer using the input sequences
    #shortest = sys.maxsize
    #longest = 0
    #average = 0
    #num_sequences = 0
    #tokenizer = loadCornbertTokenizer()
    #for fasta in fasta_files:
    #    for sequence in fastaToSequences(fasta):
    #        output = tokenizer.encode(sequence)
    #        #print(output.tokens, file=sys.stderr)
    #        num_tokens = len(output)
    #        shortest = min(shortest, num_tokens)
    #        longest = max(longest, num_tokens)
    #        average += num_tokens
    #        num_sequences += 1
    #average /= num_sequences
    #print(f'number of sequences: {num_sequences}')
    #print(f'least tokens: {shortest}')
    #print(f'most tokens: {longest}')
    #print(f'average tokens: {average}')
