import _locale
import jieba
from subword_nmt import subword_nmt
from tqdm import tqdm


class STACLTokenizer:

    def __init__(self, bpe_dict, is_chinese):
        # By default, the Windows system opens the file with GBK code,
        # and the subword_nmt package does not support setting open encoding,
        # so it is set to UTF-8 uniformly.
        _locale._getdefaultlocale = (lambda *args: ['en_US', 'utf8'])

        bpe_parser = subword_nmt.create_apply_bpe_parser()
        bpe_args = bpe_parser.parse_args(args=['-c', bpe_dict])
        self.bpe = subword_nmt.BPE(bpe_args.codes, bpe_args.merges,
                                   bpe_args.separator, None,
                                   bpe_args.glossaries)
        self.is_chinese = is_chinese

    def tokenize(self, raw_string):
        """
        Tokenize string(BPE/jieba+BPE)
        """
        raw_string = raw_string.strip('\n')
        if not raw_string:
            return raw_string
        if self.is_chinese:
            raw_string = ' '.join(jieba.cut(raw_string))
        bpe_str = self.bpe.process_line(raw_string)
        return ' '.join(bpe_str.split())


def get_full_data():
    zh_path = "../data/nist2m/train.zh.utf8"
    en_path = "../data/nist2m/train.en.utf8"

    with open(zh_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        zh_data = lines[1:]

    with open(en_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        en_data = lines[1:]

    res = []
    for pair in tqdm(zip(zh_data, en_data), desc="Tokenizing"):
        zh = tokenizer_zh.tokenize(pair[0])
        en = tokenizer_en.tokenize(pair[1])

        res.append((zh, en))

    with open('../data/train/train.zh.bpe', 'w', encoding='utf-8') as f1, \
            open('../data/train/train.en.bpe', 'w', encoding='utf-8') as f2:
        for data in tqdm(res, desc="Write train set"):
            f1.write(data[0] + "\n")
            f2.write(data[1] + "\n")


def get_min_data():
    zh_path = "../data/nist2m/train.zh.utf8"
    en_path = "../data/nist2m/train.en.utf8"

    with open(zh_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        zh_data = lines[1:1000]

    with open(en_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        en_data = lines[1:1000]

    tokenizer_zh = STACLTokenizer('../data/nist2m/2M.zh2en.dict4bpe.zh', is_chinese=True)

    tokenizer_en = STACLTokenizer('../data/nist2m/2M.zh2en.dict4bpe.en', is_chinese=False)

    res = []
    for pair in tqdm(zip(zh_data, en_data), desc="Tokenizing"):
        zh = tokenizer_zh.tokenize(pair[0])
        en = tokenizer_en.tokenize(pair[1])

        res.append((zh, en))

    with open('../data/train/train.zh.bpe.min', 'w', encoding='utf-8') as f1, \
            open('../data/train/train.en.bpe.min', 'w', encoding='utf-8') as f2:
        for data in tqdm(res, desc="Write train set"):
            f1.write(data[0] + "\n")
            f2.write(data[1] + "\n")


def process(input_path, output_path, tokenizer, is_min=False, desc="", drop_first=False):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if drop_first:
            lines = lines[1:]
        if is_min:
            lines = lines[:1000]

    res = []
    for line in tqdm(lines,desc=desc):
        res.append(tokenizer.tokenize(line))

    with open(output_path, 'w', encoding='utf-8') as f:
        for line in tqdm(res,desc="writing file"):
            f.write(line + "\n")


if __name__ == '__main__':
    tokenizer_zh = STACLTokenizer('../data/2M.zh2en.dict4bpe.zh', is_chinese=True)
    tokenizer_en = STACLTokenizer('../data/2M.zh2en.dict4bpe.en', is_chinese=False)

    # # 训练集
    # process("../data/train.zh.utf8", "../data/train/train.zh.bpe", tokenizer_zh, drop_first=True,
    #         desc="tokenizing full train zh")
    # process("../data/train.en.utf8", "../data/train/train.en.bpe", tokenizer_en, drop_first=True,
    #         desc="tokenizing full train en")
    #
    # # 最小训练集
    # process("../data/train.zh.utf8", "../data/train/train.zh.bpe.min", tokenizer_zh, drop_first=True,is_min=True,
    #         desc="tokenizing full train zh")
    # process("../data/train.en.utf8", "../data/train/train.en.bpe.min", tokenizer_en, drop_first=True,is_min=True,
    #         desc="tokenizing full train en")
    #
    # # 验证集
    # process("../data/nist06/nist06_src.plain.utf8", "../data/val/val.zh.bpe", tokenizer_zh,
    #         desc="tokenizing val zh")
    #
    # # 测试集
    # process("../data/nist08/nist08_src.plain.utf8", "../data/test/val.zh.bpe", tokenizer_zh,
    #         desc="tokenizing test zh")

    # 验证集
    process("../data/nist06/nist06.tok.utf8.ref0", "../data/val/val.en.bpe.ref0", tokenizer_en,
        desc="tokenizing val en0")
    process("../data/nist06/nist06.tok.utf8.ref1", "../data/val/val.en.bpe.ref1", tokenizer_en,
        desc="tokenizing val en1")
    process("../data/nist06/nist06.tok.utf8.ref2", "../data/val/val.en.bpe.ref2", tokenizer_en,
        desc="tokenizing val en2")
    process("../data/nist06/nist06.tok.utf8.ref3", "../data/val/val.en.bpe.ref3", tokenizer_en,
        desc="tokenizing val en3")

