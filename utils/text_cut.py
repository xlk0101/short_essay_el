from abc import abstractmethod, ABCMeta
import pkuseg
import jieba_fast as jieba
import config

# init params
comConfig = config.ComConfig()
fileConfig = config.FileConfig()


def get_client():
    if comConfig.cut_client == comConfig.jieba_client:
        return JieBaCut()
    elif comConfig.cut_client == comConfig.pkuseg_client:
        return PKUsegCut()


class BaseCut(metaclass=ABCMeta):

    @abstractmethod
    def cut_text(self, text):
        pass


class PKUsegCut(BaseCut):

    def __init__(self):
        if comConfig.use_dict:
            self.client = pkuseg.pkuseg(user_dict=fileConfig.dir_jieba + fileConfig.file_jieba_dict)
        else:
            self.client = pkuseg.pkuseg()

    def cut_text(self, text):
        return self.client.cut(text)


class JieBaCut(BaseCut):
    def __init__(self):
        if comConfig.use_dict:
            jieba.load_userdict(fileConfig.dir_jieba + fileConfig.file_jieba_dict)

    def cut_text(self, text):
        return jieba.lcut(text)
