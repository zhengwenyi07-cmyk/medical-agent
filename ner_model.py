import random
import torch
from torch import nn
import os
import pickle
import onnxruntime as ort
import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
from seqeval.metrics import f1_score
import ahocorasick
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

cache_model = 'best_roberta_rnn_model_ent_aug'


def export_model_to_onnx(model, tokenizer, max_len=50, save_path='model/ner_model.onnx'):
    model.eval()
    dummy_text = "这是一个用于导出ONNX的测试文本"
    dummy_input = tokenizer.encode(dummy_text, add_special_tokens=True, return_tensors='pt').to(next(model.parameters()).device)
    
    # 补齐到 max_len，模拟真实输入形状
    pad_len = max_len - dummy_input.shape[1]
    if pad_len > 0:
        dummy_input = torch.cat([dummy_input, torch.zeros((1, pad_len), dtype=torch.long).to(dummy_input.device)], dim=1)

    torch.onnx.export(
        model, 
        dummy_input, 
        save_path, 
        export_params=True, 
        opset_version=14, 
        do_constant_folding=True, 
        input_names=['input_ids'], 
        output_names=['output'], 
        dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence_length'}, 
                      'output': {0: 'batch_size', 1: 'sequence_length'}}
    )
    print(f"✅ 模型已成功导出为 ONNX 格式: {save_path}")

def get_data(path, max_len=None):
    all_text, all_tag = [], []
    with open(path, 'r', encoding='utf8') as f:
        all_data = f.read().split('\n')

    sen, tag = [], []
    for data in all_data:
        data = data.split(' ')
        if len(data) != 2:
            if len(sen) > 2:
                all_text.append(sen)
                all_tag.append(tag)
            sen, tag = [], []
            continue
        te, ta = data
        sen.append(te)
        tag.append(ta)
    if max_len is not None:
        return all_text[:max_len], all_tag[:max_len]
    return all_text, all_tag


class rule_find:
    def __init__(self):
        self.idx2type = idx2type = ["食物", "药品商", "治疗方法", "药品", "检查项目", "疾病", "疾病症状", "科目"]
        self.type2idx = type2idx = {"食物": 0, "药品商": 1, "治疗方法": 2, "药品": 3, "检查项目": 4, "疾病": 5, "疾病症状": 6,
                                    "科目": 7}
        self.ahos = [ahocorasick.Automaton() for i in range(len(self.type2idx))]

        for type in idx2type:
            with open(os.path.join('data', 'ent_aug', f'{type}.txt'), encoding='utf-8') as f:
                all_en = f.read().split('\n')
            for en in all_en:
                en = en.split(' ')[0]
                if len(en) >= 2:
                    self.ahos[type2idx[type]].add_word(en, en)
        for i in range(len(self.ahos)):
            self.ahos[i].make_automaton()

    def find(self, sen):
        rule_result = []
        mp = {}
        all_res = []
        all_ty = []
        for i in range(len(self.ahos)):
            now = list(self.ahos[i].iter(sen))
            all_res.extend(now)
            for j in range(len(now)):
                all_ty.append(self.idx2type[i])
        if len(all_res) != 0:
            all_res = sorted(all_res, key=lambda x: len(x[1]), reverse=True)
            for i, res in enumerate(all_res):
                be = res[0] - len(res[1]) + 1
                ed = res[0]
                if be in mp or ed in mp:
                    continue
                rule_result.append((be, ed, all_ty[i], res[1]))
                for t in range(be, ed + 1):
                    mp[t] = 1
        return rule_result


def find_entities(tag):
    result = []  # [(2,3,'药品'),(7,10,'药品商')]
    label_len = len(tag)
    i = 0
    while (i < label_len):
        if (tag[i][0] == 'B'):
            type = tag[i].strip('B-')
            j = i + 1
            while (j < label_len and tag[j][0] == 'I'):
                j += 1
            result.append((i, j - 1, type))
            i = j
        else:
            i = i + 1
    return result


class tfidf_alignment():
    def __init__(self):
        eneities_path = os.path.join('data', 'ent_aug')
        files = os.listdir(eneities_path)
        files = [docu for docu in files if '.py' not in docu]

        self.tag_2_embs = {}
        self.tag_2_tfidf_model = {}
        self.tag_2_entity = {}
        for ty in files:
            with open(os.path.join(eneities_path, ty), 'r', encoding='utf-8') as f:
                entities = f.read().split('\n')
                entities = [ent for ent in entities if len(ent.split(' ')[0]) <= 15 and len(ent.split(' ')[0]) >= 1]
                en_name = [ent.split(' ')[0] for ent in entities]
                ty = ty.strip('.txt')
                self.tag_2_entity[ty] = en_name
                tfidf_model = TfidfVectorizer(analyzer="char")
                embs = tfidf_model.fit_transform(en_name).toarray()
                self.tag_2_embs[ty] = embs
                self.tag_2_tfidf_model[ty] = tfidf_model

    def align(self, ent_list, bge_model=None):
        """
        对每一个实体类型，收集所有相似度达标的实体。

        主策略：TF-IDF 字符 n-gram 匹配（快，适合字符重叠高的短实体名）。
        兜底策略：当 TF-IDF 分数不足时，尝试 BGE 语义嵌入匹配（处理
        "阿莫仙"→"阿莫西林"这类字符不重叠但语义相近的 case）。

        Args:
            ent_list: 待对齐的实体列表 [(start, end, type, raw_name), ...]
            bge_model: 可选，已加载的 SentenceTransformer BGE 模型实例。

        Returns:
            {type: [entity_name1, entity_name2, ...]}
        """
        new_result = {}   # {type: [entity1, entity2, ...]}
        seen = {}         # {type: set()}  用于同类型内去重

        for s, e, cls, ent in ent_list:
            if cls not in self.tag_2_entity:
                continue
            standard_names = self.tag_2_entity[cls]

            # 主策略：TF-IDF 字符级匹配
            ent_emb = self.tag_2_tfidf_model[cls].transform([ent])
            sim_score = cosine_similarity(ent_emb, self.tag_2_embs[cls])
            max_idx = sim_score[0].argmax()
            max_score = sim_score[0][max_idx]

            matched = None
            match_method = None

            if max_score >= 0.5:
                matched = standard_names[max_idx]
                match_method = "TF-IDF"
            elif bge_model is not None and len(standard_names) > 0:
                # 兜底策略：BGE 语义匹配（处理字符不重叠但语义相近的 case）
                try:
                    query_emb = bge_model.encode(ent, normalize_embeddings=True)
                    std_embs = bge_model.encode(standard_names, normalize_embeddings=True)
                    bge_scores = cosine_similarity([query_emb], std_embs)[0]
                    bge_max_idx = bge_scores.argmax()
                    bge_max_score = bge_scores[bge_max_idx]

                    if bge_max_score >= 0.65:  # BGE 阈值略高于 TF-IDF（嵌入空间更稠密）
                        matched = standard_names[bge_max_idx]
                        match_method = f"BGE({bge_max_score:.2f})"
                except Exception:
                    pass  # BGE 兜底失败，不影响主流程

            if matched:
                if cls not in new_result:
                    new_result[cls] = []
                    seen[cls] = set()
                if matched not in seen[cls]:
                    seen[cls].add(matched)
                    new_result[cls].append(matched)

        return new_result


class Entity_Extend:
    def __init__(self):
        eneities_path = os.path.join('data', 'ent_aug')
        files = os.listdir(eneities_path)
        files = [docu for docu in files if '.py' not in docu]

        self.type2entity = {}
        self.type2weight = {}
        for type in files:
            with open(os.path.join(eneities_path, type), 'r', encoding='utf-8') as f:
                entities = f.read().split('\n')
                en_name = [ent for ent in entities if len(ent.split(' ')[0]) <= 15 and len(ent.split(' ')[0]) >= 1]
                en_weight = [1] * len(en_name)
                type = type.strip('.txt')
                self.type2entity[type] = en_name
                self.type2weight[type] = en_weight

    def no_work(self, te, tag, type):
        return te, tag

    def entity_replace(self, te, ta, type):
        choice_ent = random.choices(self.type2entity[type], weights=self.type2weight[type], k=1)[0]
        ta = ["B-" + type] + ["I-" + type] * (len(choice_ent) - 1)
        return list(choice_ent), ta

    def entity_mask(self, te, ta, type):
        if (len(te) <= 3):
            return te, ta
        elif (len(te) <= 5):
            te.pop(random.randint(0, len(te) - 1))
        else:
            te.pop(random.randint(0, len(te) - 1))
            te.pop(random.randint(0, len(te) - 1))
        ta = ["B-" + type] + ["I-" + type] * (len(te) - 1)
        return te, ta

    def entity_union(self, te, ta, type):
        words = ['和', '与', '以及']
        wor = random.choice(words)
        choice_ent = random.choices(self.type2entity[type], weights=self.type2weight[type], k=1)[0]
        te = te + list(wor) + list(choice_ent)
        ta = ta + ['O'] * len(wor) + ["B-" + type] + ["I-" + type] * (len(choice_ent) - 1)
        return te, ta

    def entities_extend(self, text, tag, ents):
        cho = [self.no_work, self.entity_union, self.entity_mask, self.entity_replace, self.no_work]
        new_text = text.copy()
        new_tag = tag.copy()
        sign = 0
        for ent in ents:
            p = random.choice(cho)
            te, ta = p(text[ent[0]:ent[1] + 1], tag[ent[0]:ent[1] + 1], ent[2])
            new_text[ent[0] + sign:ent[1] + 1 + sign], new_tag[ent[0] + sign:ent[1] + 1 + sign] = te, ta
            sign += len(te) - (ent[1] - ent[0] + 1)

        return new_text, new_tag


class Nerdataset(Dataset):
    def __init__(self, all_text, all_label, tokenizer, max_len, tag2idx, is_dev=False, enhance_data=False):
        self.all_text = all_text
        self.all_label = all_label
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tag2idx = tag2idx
        self.is_dev = is_dev
        self.entity_extend = Entity_Extend()
        self.enhance_data = enhance_data

    def __getitem__(self, x):
        text, label = self.all_text[x], self.all_label[x]
        if self.is_dev:
            max_len = min(len(self.all_text[x]) + 2, 500)
        else:
            if self.enhance_data and e >= 7 and e % 2 == 1:
                ents = find_entities(label)
                text, label = self.entity_extend.entities_extend(text, label, ents)
            max_len = self.max_len
        text, label = text[:max_len - 2], label[:max_len - 2]

        x_len = len(text)
        assert len(text) == len(label)
        text_idx = self.tokenizer.encode(text, add_special_token=True)
        label_idx = [self.tag2idx['<PAD>']] + [self.tag2idx[i] for i in label] + [self.tag2idx['<PAD>']]

        text_idx += [0] * (max_len - len(text_idx))
        label_idx += [self.tag2idx['<PAD>']] * (max_len - len(label_idx))
        return torch.tensor(text_idx), torch.tensor(label_idx), x_len

    def __len__(self):
        return len(self.all_text)


def build_tag2idx(all_tag):
    tag2idx = {'<PAD>': 0}
    for sen in all_tag:
        for tag in sen:
            tag2idx[tag] = tag2idx.get(tag, len(tag2idx))
    return tag2idx


class Bert_Model(nn.Module):
    def __init__(self, model_name, hidden_size, tag_num, bi):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.gru = nn.RNN(input_size=768, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=bi)
        if bi:
            self.classifier = nn.Linear(hidden_size * 2, tag_num)
        else:
            self.classifier = nn.Linear(hidden_size, tag_num)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, x, label=None):
        bert_0, _ = self.bert(x, attention_mask=(x > 0), return_dict=False)
        gru_0, _ = self.gru(bert_0)
        pre = self.classifier(gru_0)
        if label is not None:
            loss = self.loss_fn(pre.reshape(-1, pre.shape[-1]), label.reshape(-1))
            return loss
        else:
            return torch.argmax(pre, dim=-1).squeeze(0)


def merge(model_result_word, rule_result):
    """
    【修复】采用最长不重叠策略合并模型与规则实体，避免短实体被长实体完全删除。
    - result 按实体长度降序排列
    - 遍历时若当前位置已被占用，则尝试用更长实体替换已占用的短实体（如果更长实体覆盖了短实体，则替换）
    - 最终保留一组不相交且尽可能长的实体。
    """
    result = model_result_word + rule_result
    # 按实体长度降序，长度相同则按起始位置升序
    result = sorted(result, key=lambda x: (len(x[-1]), x[0]), reverse=True)

    check_result = []  # [(be, ed, type, word), ...]
    occupied = {}       # 记录每个位置被哪个实体占用，key=索引，value=实体信息

    for res in result:
        be, ed, typ, word = res
        # 检查是否有冲突
        conflict = any(i in occupied for i in range(be, ed + 1))
        if not conflict:
            # 无冲突直接加入
            check_result.append(res)
            for i in range(be, ed + 1):
                occupied[i] = res
        else:
            # 有冲突时，判断当前实体是否比重叠的实体更长，若更长，则替换掉那些短实体
            overlapping = set()
            for i in range(be, ed + 1):
                if i in occupied:
                    overlapping.add(occupied[i])
            # 如果所有重叠的实体都比当前短，则把它们从结果中移除，换上这个更长的
            if all(len(item[-1]) < len(word) for item in overlapping):
                # 移除短实体
                for rem_item in overlapping:
                    if rem_item in check_result:
                        check_result.remove(rem_item)
                    # 清除其占用标记
                    for idx in range(rem_item[0], rem_item[1] + 1):
                        if occupied.get(idx) == rem_item:
                            del occupied[idx]
                # 加入当前长实体
                check_result.append(res)
                for i in range(be, ed + 1):
                    occupied[i] = res
    return check_result


def get_ner_result(model, tokenizer, sen, rule, tfidf_r, device, idx2tag):
    sen_to = tokenizer.encode(sen, add_special_tokens=True, return_tensors='pt').to(device)

    pre = model(sen_to).tolist()

    pre_tag = [idx2tag[i] for i in pre[1:-1]]
    model_result = find_entities(pre_tag)
    model_result_word = []
    for res in model_result:
        word = sen[res[0]:res[1] + 1]
        model_result_word.append((res[0], res[1], res[2], word))
    rule_result = rule.find(sen)

    merge_result = merge(model_result_word, rule_result)
    tfidf_result = tfidf_r.align(merge_result, bge_model=_lazy_bge())
    return tfidf_result


# BGE 语义兜底模型懒加载
_bge = None


def _lazy_bge():
    """懒加载 BGE 嵌入模型，用于 TF-IDF 对齐失败时的语义兜底。
    首次调用约 1-2s 加载（模型约 100MB），后续复用。
    """
    global _bge
    if _bge is not None:
        return _bge
    try:
        from sentence_transformers import SentenceTransformer
        _bge = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        print("[NER] BGE 语义兜底模型加载完成")
        return _bge
    except Exception as e:
        print(f"[NER] BGE 模型加载失败（非致命，仅 TF-IDF 工作）: {e}")
        return None



# 修改：适配 ONNX Session 的 NER 推理函数
def get_ner_result_onnx(ort_session, tokenizer, sen, rule, tfidf_r, idx2tag):
    """
    使用 ONNX Runtime 进行推理，完全摆脱 PyTorch 前向传播
    """
    # Tokenize 并转换为 numpy 数组
    sen_to = tokenizer.encode(sen, add_special_tokens=True)
    input_ids = np.array([sen_to], dtype=np.int64)
    
    # ONNX 推理
    ort_inputs = {ort_session.get_inputs()[0].name: input_ids}
    ort_outs = ort_session.run(None, ort_inputs)
    
    # 获取输出。由于原 PyTorch 模型的 forward 已经执行了 argmax 和 squeeze
    # 这里的 ort_outs[0] 已经是具体的标签索引了
    pre = ort_outs[0]
    
    # 兼容处理：判断输出是 1D (seq_len,) 还是 2D (1, seq_len)
    if pre.ndim == 2:
        pre_labels = pre[0].tolist()
    else:
        pre_labels = pre.tolist()
    
    # 掐头去尾 [CLS] [SEP]
    pre_tag = [idx2tag[i] for i in pre_labels[1:len(sen_to)-1]]
    
    model_result = find_entities(pre_tag)
    model_result_word = []
    for res in model_result:
        # 防越界保护
        if res[1] < len(sen):
            word = sen[res[0]:res[1] + 1]
            model_result_word.append((res[0], res[1], res[2], word))
            
    rule_result = rule.find(sen)
    merge_result = merge(model_result_word, rule_result)
    tfidf_result = tfidf_r.align(merge_result, bge_model=_lazy_bge())

    return tfidf_result


if __name__ == "__main__":
    all_text,all_label = get_data(os.path.join('data','ner_data_aug.txt'))
    train_text, dev_text, train_label, dev_label = train_test_split(all_text, all_label, test_size = 0.02, random_state = 42)

    #加载太慢了，预处理一下
    if os.path.exists('tmp_data/tag2idx.npy'):
        with open('tmp_data/tag2idx.npy','rb') as f:
            tag2idx = pickle.load(f)
    else:
        tag2idx = build_tag2idx(all_label)
        with open('tmp_data/tag2idx.npy','wb') as f:
            pickle.dump(tag2idx,f)


    idx2tag = list(tag2idx)

    max_len = 50
    epoch = 30
    batch_size = 60
    hidden_size = 128
    bi = True
    model_name='model/chinese-roberta-wwm-ext'#bert_base_chinese
    tokenizer = BertTokenizer.from_pretrained(model_name)
    lr =1e-5
    is_train=True

    device = torch.device('cuda:0') if torch.cuda.is_available()   else torch.device('cpu')

    train_dataset = Nerdataset(train_text,train_label,tokenizer,max_len,tag2idx,enhance_data=True)
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

    dev_dataset = Nerdataset(dev_text, dev_label, tokenizer, max_len, tag2idx,is_dev=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False)

    model = Bert_Model(model_name,hidden_size,len(tag2idx),bi)
    # if os.path.exists(f'model/best_roberta_gru_model_ent_aug.pt'):
    #     model.load_state_dict(torch.load('model/best_roberta_gru_model_ent_aug.pt'))
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(),lr = lr)
    bestf1 = -1
    if is_train:
        for e in range(epoch):
            loss_sum = 0
            ba = 0
            for x,y,batch_len in tqdm(train_dataloader):
                x = x.to(device)
                y = y.to(device)
                opt.zero_grad()
                loss = model(x,y)
                loss.backward()

                opt.step()
                loss_sum+=loss
                ba += 1
            all_pre = []
            all_label = []
            for x,y,batch_len in tqdm(dev_dataloader):
                assert len(x)==len(y)
                x = x.to(device)
                pre = model(x)
                pre = [idx2tag[i] for i in pre[1:batch_len+1]]
                all_pre.append(pre)

                label = [idx2tag[i] for i in y[0][1:batch_len+1]]
                all_label.append(label)
            f1 = f1_score(all_pre, all_label)
            if f1>bestf1:
                bestf1 = f1
                print(f'e={e},loss={loss_sum / ba:.5f} f1={f1:.5f} ---------------------->best')
                torch.save(model.state_dict(),f'model/{cache_model}.pt')
            else:print(f'e={e},loss={loss_sum/ba:.5f} f1={f1:.5f}')

    rule = rule_find()
    tfidf_r = tfidf_alignment()

    while(True):
        sen = input('请输入:')
        print(get_ner_result(model, tokenizer, sen, rule, tfidf_r,device,idx2tag))
