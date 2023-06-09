import numpy as np
from Bio import SeqIO
import os


class kmer_featurization:

    def __init__(self, k):
        """
    seqs: a list of DNA sequences
    k: the "k" in k-mer
    """
        self.k = k
        self.letters = ['A', 'T', 'C', 'G']
        self.multiplyBy = 4 ** np.arange(k - 1, -1,
                                         -1)  # the multiplying number for each digit position in the k-number system
        self.n = 4 ** k  # number of possible k-mers

    def obtain_kmer_feature_for_a_list_of_sequences(self, seqs, write_number_of_occurrences=False):
        """
    Given a list of m DNA sequences, return a 2-d array with shape (m, 4**k) for the 1-hot representation of the kmer features.
    Args:
      write_number_of_occurrences:
        a boolean. If False, then in the 1-hot representation, the percentage of the occurrence of a kmer will be recorded; otherwise the number of occurrences will be recorded. Default False.    
    """
        kmer_features = []
        for seq in seqs:
            this_kmer_feature = self.obtain_kmer_feature_for_one_sequence(seq.upper(),
                                                                          write_number_of_occurrences=write_number_of_occurrences)
            kmer_features.append(this_kmer_feature)

        kmer_features = np.array(kmer_features)

        return kmer_features

    def obtain_kmer_feature_for_one_sequence(self, seq, write_number_of_occurrences=False):
        """
    Given a DNA sequence, return the 1-hot representation of its kmer feature.
    Args:
      seq: 
        a string, a DNA sequence
      write_number_of_occurrences:
        a boolean. If False, then in the 1-hot representation, the percentage of the occurrence of a kmer will be recorded; otherwise the number of occurrences will be recorded. Default False.
    """
        number_of_kmers = len(seq) - self.k + 1

        kmer_feature = np.zeros(self.n)

        for i in range(number_of_kmers):
            this_kmer = seq[i:(i + self.k)]
            this_numbering = self.kmer_numbering_for_one_kmer(this_kmer)
            kmer_feature[this_numbering] += 1

        # if not write_number_of_occurrences:
        #   kmer_feature = kmer_feature / number_of_kmers

        return kmer_feature

    def kmer_numbering_for_one_kmer(self, kmer):
        """
    Given a k-mer, return its numbering (the 0-based position in 1-hot representation)
    """
        digits = []
        for letter in kmer:
            digits.append(self.letters.index(letter))

        digits = np.array(digits)

        numbering = (digits * self.multiplyBy).sum()

        return numbering


def get_kmer():
    print(os.getcwd())
    k = 6
    kmer = kmer_featurization(k)
    seqlist = SeqIO.parse(r'../../data/dataset/input.fa', 'fasta')  # 相对运行该文件的位置
    L = ['A', 'C', 'T', 'G']
    res = []
    i = 0
    for each in seqlist:
        print(i, '条正在处理')
        i += 1
        seq = str(each.seq)
        print('序列信息为', seq)
        cseq = ''
        for char in seq:
            if char in L:
                cseq += char
        data = kmer.obtain_kmer_feature_for_one_sequence(cseq)
        res.append(data)
    # seq = "AAACCCTTTGGGCACAGTCAGCTGATCGATCGATCGATCGATCGTACGATGCTAGCATCGATCGATCGATGCTAGCTAGCTACGATGCATGCATCGATCGATCGATCGATCGA"
    kmernor = []
    for i in range(len(res)):
        kmersum = np.sum(res[i])
        W = []
        for j in range(4 ** k):
            f = res[i][j] / kmersum
            W.append(f)
        kmernor.append(W)
    print(np.sum(res[0]),np.sum(kmernor[0]))
    #np.save(r'../../data/8mer', arr=res)
    np.save(f'../../data/{k}mernor', arr=kmernor)
    print('完成kmer特征提取')


if __name__ == '__main__':
    get_kmer()
