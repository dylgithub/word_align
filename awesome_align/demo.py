# -*- coding: utf-8 -*-
import itertools
from torch.nn.utils.rnn import pad_sequence
from awesome_align.tokenization_bert import BertTokenizer
from torch.utils.data import DataLoader, Dataset
import os


class Demo:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('../rbt3')

    def process_line(self, worker_id, line):
        if len(line) == 0 or line.isspace() or not len(line.split(' ||| ')) == 2:
            return None

        src, tgt = line.split(' ||| ')
        if src.rstrip() == '' or tgt.rstrip() == '':
            return None

        sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
        for word in sent_src:
            print(self.tokenizer.tokenize(word))
        token_src, token_tgt = [self.tokenizer.tokenize(word) for word in sent_src], [self.tokenizer.tokenize(word) for
                                                                                      word
                                                                                      in sent_tgt]
        print("token_src......", token_src)
        wid_src, wid_tgt = [self.tokenizer.convert_tokens_to_ids(x) for x in token_src], [
            self.tokenizer.convert_tokens_to_ids(x) for x in token_tgt]
        print(wid_src)
        ids_src, ids_tgt = self.tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt',
                                                            max_length=self.tokenizer.max_len)['input_ids'], \
                           self.tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt',
                                                            max_length=self.tokenizer.max_len)['input_ids']
        print(ids_src)
        if len(ids_src[0]) == 2 or len(ids_tgt[0]) == 2:
            return None

        bpe2word_map_src = []
        for i, word_list in enumerate(token_src):
            bpe2word_map_src += [i for x in word_list]
        bpe2word_map_tgt = []
        for i, word_list in enumerate(token_tgt):
            bpe2word_map_tgt += [i for x in word_list]
        return (worker_id, ids_src[0], ids_tgt[0], bpe2word_map_src, bpe2word_map_tgt, sent_src, sent_tgt)


class LineByLineTextDataset(Dataset):
    def __init__(self, file_path):
        assert os.path.isfile(file_path)
        print('Loading the dataset...')
        self.tokenizer = BertTokenizer.from_pretrained('../rbt3')

        self.ids_src = []
        self.ids_tgt = []
        self.bpe2word_map_src = []
        self.bpe2word_map_tgt = []
        self.sent_src = []
        self.sent_tgt = []
        self.load_data(file_path)

    def load_data(self, filename):
        # 加载数据
        with open(filename, "r", encoding="utf-8") as fr:
            for line in fr.readlines():
                line = line.strip()
                if len(line) == 0 or line.isspace() or not len(line.split(' ||| ')) == 2:
                    return None

                src, tgt = line.split(' ||| ')
                if src.rstrip() == '' or tgt.rstrip() == '':
                    return None

                sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
                token_src, token_tgt = [self.tokenizer.tokenize(word) for word in sent_src], [
                    self.tokenizer.tokenize(word) for word in sent_tgt]
                wid_src, wid_tgt = [self.tokenizer.convert_tokens_to_ids(x) for x in token_src], [
                    self.tokenizer.convert_tokens_to_ids(x) for x in token_tgt]

                ids_src, ids_tgt = \
                    self.tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt',
                                                     max_length=self.tokenizer.max_len)['input_ids'], \
                    self.tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt',
                                                     max_length=self.tokenizer.max_len)['input_ids']
                if len(ids_src[0]) == 2 or len(ids_tgt[0]) == 2:
                    return None
                bpe2word_map_src = []
                for i, word_list in enumerate(token_src):
                    bpe2word_map_src += [i for x in word_list]
                bpe2word_map_tgt = []
                for i, word_list in enumerate(token_tgt):
                    bpe2word_map_tgt += [i for x in word_list]
                # print(ids_src[0], len(ids_src[0]))
                # print(ids_tgt[0], len(ids_tgt[0]))
                # print(bpe2word_map_src)
                # print(sent_src)
                self.ids_src.append(ids_src[0])
                self.ids_tgt.append(ids_tgt[0])
                self.bpe2word_map_src.append(bpe2word_map_src)
                self.bpe2word_map_tgt.append(bpe2word_map_tgt)
                self.sent_src.append(sent_src)
                self.sent_tgt.append(sent_tgt)

    def __getitem__(self, index):
        return self.ids_src[index], self.ids_tgt[index], self.bpe2word_map_src[index], self.bpe2word_map_tgt[index], \
               self.sent_src[index], self.sent_tgt[index]

    def __len__(self):
        return len(self.ids_src)


if __name__ == '__main__':
    # demo = Demo()
    # print(demo.process_line(1, "测试 数据1 ||| 测试 数据2"))
    train_dataset = LineByLineTextDataset("../data/word_align_use_final.txt")
    tokenizer = BertTokenizer.from_pretrained('../rbt3')


    def collate(examples):
        ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt = zip(*examples)
        ids_src = pad_sequence(ids_src, batch_first=True, padding_value=tokenizer.pad_token_id)
        ids_tgt = pad_sequence(ids_tgt, batch_first=True, padding_value=tokenizer.pad_token_id)
        return ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt


    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate)

    for index, (ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt) in enumerate(train_dataloader):
        print(ids_src.size(0))
        print(len(bpe2word_map_src))
        print(len(sents_src))
