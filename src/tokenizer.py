# created following to the Hugging Face Tokenizers Quicktour:
# https://huggingface.co/docs/tokenizers/quicktour

import sys
from itertools import chain

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from fasta import fastaToSequences


TOKENIZER_CONFIG_FILE = "cornbert-tokenizer.json"


def trainTokenizerOnSequences(sequence_iterable, vocab_size=5000, output_config_file=TOKENIZER_CONFIG_FILE):
    unknown_token="[UNK]"
    special_tokens=[unknown_token, "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

    # instantiate a BPE tokeinzer model
    tokenizer = Tokenizer(BPE(unk_token=unknown_token))
    # instantiate a trainer for our BPE tokenizer
    trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=vocab_size)
    # use a pre-tokenizer to split inputs into words using whitespace as the word separator
    # TODO: do separate string in iterator still need whitespace to separate?
    tokenizer.pre_tokenizer = Whitespace()

    # train the tokenizer on the sequences
    #tokenizer.train(fasta_files, trainer)
    tokenizer.train_from_iterator(sequence_iterable, trainer)

    # post-process the tokens
    # TODO: can this be used to pad the tokens so the vectors are all the same length?
    #from tokenizers.processors import TemplateProcessing
    #tokenizer.post_processor = TemplateProcessing(
    #    single="[CLS] $A [SEP]",
    #    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    #    special_tokens=[
    #        ("[CLS]", tokenizer.token_to_id("[CLS]")),
    #        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    #    ],
    #)

    # save the tokenizer to a single file that contains its configuration and the vocabulary
    tokenizer.save(output_config_file)


# Trains a byte-level(?) BPE tokenizer from the given FASTA files using the following special tokens:
# "[UNK]" - unknown character
# "[CLS]" - classifier token
# "[SEP]" - separator
# "[PAD]" - adds padding to token vectors?
# "[MASK]" - used to mask tokens during training?
def trainTokenizerOnFastas(fasta_files, vocab_size=5000, output_config_file=TOKENIZER_CONFIG_FILE):
    # create an iterable of sequences from the input FASTAs
    sequence_generators = [fastaToSequences(fasta) for fasta in fasta_files]
    sequence_iterator = chain.from_iterable(sequence_generators)
    # train the tokenizer
    trainTokenizerOnSequences(sequence_iterator, vocab_size, output_config_file)


def loadCornbertTokenizer(config_file=TOKENIZER_CONFIG_FILE):
    # reload the tokenizer from its tokenizer file
    return Tokenizer.from_file(config_file)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        exit(sys.argv[0])
    trainTokenizerOnFastas(sys.argv[1:])
    # use the tokenizer
    #output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
