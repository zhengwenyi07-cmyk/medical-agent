import torch
import pickle
import time
import onnxruntime as ort
from transformers import BertTokenizer
import os

# 导入你自己的 NER 模块
import ner_model as zwk

def main():
    print("=== 1. 准备基础组件 ===")
    # 加载 tag2idx
    with open('tmp_data/tag2idx.npy', 'rb') as f:
        tag2idx = pickle.load(f)
    idx2tag = list(tag2idx)

    # 加载规则和 TF-IDF 对齐组件
    rule = zwk.rule_find()
    tfidf_r = zwk.tfidf_alignment()

    # 加载 Tokenizer
    model_name = 'model/chinese-roberta-wwm-ext'
    tokenizer = BertTokenizer.from_pretrained(model_name)

    print("=== 2. 加载 PyTorch 模型 ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pt_model = zwk.Bert_Model(model_name, hidden_size=128, tag_num=len(tag2idx), bi=True)
    
    # 请确保这里的模型权重名称与你实际训练出来的文件一致
    model_weight_path = 'model/best_roberta_rnn_model_ent_aug.pt'
    if not os.path.exists(model_weight_path):
        print(f"❌ 找不到 PyTorch 权重文件: {model_weight_path}")
        return
        
    pt_model.load_state_dict(torch.load(model_weight_path, map_location=device))
    pt_model = pt_model.to(device)
    pt_model.eval()

    print("=== 3. 导出 ONNX 模型 ===")
    onnx_path = 'model/ner_model.onnx'
    # 调用你在 ner_model.py 中补充的导出函数
    zwk.export_model_to_onnx(pt_model, tokenizer, max_len=50, save_path=onnx_path)

    print("=== 4. 加载 ONNX Session ===")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
    ort_session = ort.InferenceSession(onnx_path, providers=providers)
    print(f"使用的后端: {ort_session.get_providers()[0]}")

    # ==========================
    # 开始进行 A/B 测试
    # ==========================
    query = "我最近经常头痛，还会伴随恶心，请问要吃分布在广州的白云山制药厂的什么药？"
    print(f"\n[测试输入] {query}\n")

    # [PyTorch 推理]
    print("=== PyTorch 推理测试 ===")
    pt_start = time.time()
    pt_result = zwk.get_ner_result(pt_model, tokenizer, query, rule, tfidf_r, device, idx2tag)
    pt_time = time.time() - pt_start
    print(f"PyTorch 耗时: {pt_time:.4f} 秒")
    print(f"PyTorch 结果: {pt_result}\n")

    # [ONNX 推理]
    print("=== ONNX 推理测试 ===")
    onnx_start = time.time()
    onnx_result = zwk.get_ner_result_onnx(ort_session, tokenizer, query, rule, tfidf_r, idx2tag)
    onnx_time = time.time() - onnx_start
    print(f"ONNX 耗时: {onnx_time:.4f} 秒")
    print(f"ONNX 结果: {onnx_result}\n")

    # [对齐校验]
    print("=== 🎯 精度对齐校验结论 ===")
    if pt_result == onnx_result:
        print("✅ 测试通过！ONNX 模型与 PyTorch 模型输出完全一致，精度无损对齐！")
        if onnx_time < pt_time:
            print(f"🚀 ONNX 提速效果: 约 {pt_time/onnx_time:.2f} 倍")
        else:
            print("💡 提示：在首次运行或极短文本下，提速可能不明显，但在高并发场景下 ONNX 占用资源极低。")
    else:
        print("❌ 测试失败！两边输出不一致，请检查模型输入输出的维度截断逻辑。")

if __name__ == "__main__":
    main()