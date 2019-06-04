import sys
import config
from torch import optim
from model.nel_entity_link import EntityLink
from entity.entity import Vocab
from utils import data_utils
from evaluate import nel_evaluate

# init params
comConfig = config.ComConfig()
fileConfig = config.FileConfig()
nelConfig = config.NELConfig()


def train():
    entity_context_vocab = Vocab(fileConfig.dir_nel + fileConfig.file_nel_entity_context_vocab)
    mention_context_vocab = Vocab(fileConfig.dir_nel + fileConfig.file_nel_mention_context_vocab)
    entity_vocab = Vocab(fileConfig.dir_nel + fileConfig.file_nel_entity_vocab)
    print('entity_context_vocab size : {}'.format(len(entity_context_vocab)))
    print('mention_context_vocab size : {}'.format(len(mention_context_vocab)))
    print('entity_vocab size : {}'.format(len(entity_vocab)))
    train_iter, dev_iter = data_utils.create_nel_batch_iter(nelConfig.mode_train, entity_context_vocab,
                                                            mention_context_vocab, entity_vocab)
    model = EntityLink(len(mention_context_vocab), len(entity_context_vocab), len(entity_vocab),
                       emb_dim=nelConfig.emb_dim, hid_dim=nelConfig.hid_dim)
    model.to(comConfig.device)
    opt = optim.Adam(model.parameters())
    nel_evaluate.train_eval(model, train_iter, dev_iter, opt)


if __name__ == '__main__':
    if len(sys.argv) == 1 or not sys.argv[1] in ['train', 'test']:
        print("should input param [train/test/predict]")
    if sys.argv[1] == 'train':
        train()
    else:
        pass
