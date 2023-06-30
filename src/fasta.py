# a generator that yields the sequences from a FASTA file with the newlines stripped
def fastaToSequences(fastafile):
    fragments = []
    with open(fastafile, 'r') as fasta:
        for line in fasta:
            if line.startswith('>'):
                if fragments:
                    yield ''.join(fragments)
                    fragments = []
            else:
                fragment = line.strip()
                fragments.append(fragment)
        if fragments:
            yield ''.join(fragments)
