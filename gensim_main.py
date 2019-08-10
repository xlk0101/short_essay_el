import sys
import config

from gensim.models import fasttext
from gensim.models import word2vec

# init params
fileConfig = config.FileConfig()


def train_unsup():
    print("start train gensim fasttext unsup model")
    model = fasttext.FastText(size=256, window=5, min_count=1, word_ngrams=0, workers=8)
    # scan over corpus to build the vocabulary
    model.build_vocab(
        corpus_file=fileConfig.dir_fasttext + fileConfig.file_fasttext_unsup_train_data)
    total_words = model.corpus_total_words  # number of words in the corpus
    print('train...')
    model.train(corpus_file=fileConfig.dir_fasttext + fileConfig.file_fasttext_unsup_train_data,
                total_words=total_words, epochs=3)
    model.save(fileConfig.dir_fasttext + fileConfig.file_fasttext_gensim_unsup_model)
    print("success train gensim fasttext unsup model")


def test():
    print("start test gensim fasttext unsup model")
    model = word2vec.Word2VecKeyedVectors.load(fileConfig.dir_fasttext + fileConfig.file_gensim_tencent_unsup_model)
    print(model.most_similar('蛋糕', topn=5))
    print("success save tencent word vectors")


if __name__ == '__main__':
    if len(sys.argv) == 1 or not sys.argv[1] in ['train_unsup', 'test']:
        print("should input param [train_unsup/test]")
    if sys.argv[1] == 'train_unsup':
        train_unsup()
    elif sys.argv[1] == 'test':
        test()
    else:
        pass
