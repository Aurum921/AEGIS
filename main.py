import gradio as gr
import pandas as pd
import csv
from predict_label import predict_single_sample, DynamicFeatureWeighter
from entropy import compute_max_token_and_entropy, interact
import pickle
import ast


def poison_retrieval(query, intensity):  # 输入查询和注毒量，输出检索到的文本

    csv_file_path = "queries_and_documents.csv"
    poison_list = []
    clean_list = []

    with open(csv_file_path, encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        # 跳过前两行
        next(reader)

        for row in reader:
            if not row or len(row) < 3:
                continue

            # 首列检查
            if row[0] == query:
                category = row[2].strip().lower()  # 第三列类别标识，统一小写
                second_col = row[1] if len(row) > 1 else ''

                if category == 'poison':
                    poison_list.append(second_col)
                elif category == 'clean':
                    clean_list.append(second_col)
    clean_data = []
    poisoned_data = []
    clean_txt = []
    poisoned_txt = []
    i = 0

    for x in clean_list:
        clean_data.append((x+"\n", 0.0))
        clean_txt.append(x)
    for x in poison_list:
        if i < intensity:
            poisoned_data.append((x+"\n", 0.0))
            poisoned_txt.append(x)
            i += 1
        else:
            break

    return poisoned_data + clean_data, poisoned_txt, clean_txt


def detox(poisoned_doc):  # 输入查询与所有备选文本，输出查询与干净文本
    clean_list = []
    for x in poisoned_doc:
        if '[CLEAN]' in x[0]:
            clean_list.append(x[0])
    return clean_list


def tokens(poisoned_doc):
    dirty_list = []
    for x in poisoned_doc:
        dirty_list.append(x[0])
    return dirty_list


def compute(query_text, poison_doc_texts, clean_doc_texts, poisoned_doc):  # 计算每条文本的注意力熵，返回格式为字典列表，字典的两个键为”文本“、”可信度“
    model_file = 'dynamic_weighted_classifier.pkl'
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    poison_list = ast.literal_eval(poison_doc_texts)
    clean_list = ast.literal_eval(clean_doc_texts)
    total_list = compute_max_token_and_entropy(query_text, poison_list, clean_list)
    final_list = []
    for i, x in enumerate(total_list):
        max_token_attention, entropy = x[0], x[1]
        label, probability = predict_single_sample(model, max_token_attention, entropy)
        final_list.append({"文本": poisoned_doc[i]['token'], "可疑度": probability})
        if label == "poison":
            poisoned_doc[i] = (f"[POISON]{poisoned_doc[i]['token']}", probability)
        else:
            poisoned_doc[i] = (f"[CLEAN]{poisoned_doc[i]['token']}", 0.0)

    return final_list, poisoned_doc


def detoxify_retrieval(query_text, poison_doc_texts, clean_doc_texts, poisoned_doc):  # 前面几个函数接口没问题的话后面的理论上就都不用改了
    """ 解毒处理：过滤带错误标记的文本 """

    data, poisoned_doc = compute(query_text, poison_doc_texts, clean_doc_texts, poisoned_doc)
    clean_data = detox(poisoned_doc)
    poisoned_data = tokens(poisoned_doc)
    dirty_answer = interact(query_text, poisoned_data)
    clean_answer = interact(query_text, clean_data)
    df = pd.DataFrame(data)
    df = gr.Dataframe(value=df)
    return dirty_answer, clean_answer, df, poisoned_doc


with gr.Blocks(theme=gr.themes.Soft(), title="大模型检索攻防演示") as demo:
    gr.Markdown("## 🔍 大模型检索污染与解毒系统")

    with gr.Row():
        # 左侧污染模块
        with gr.Column(variant="panel"):
            gr.Markdown("### ⚠️ 检索污染攻击")
            query_input = gr.Textbox(label="输入查询问题", placeholder="example:who recorded i can't help falling in love with you")
            intensity = gr.Slider(1, 5, step=1, label="污染强度（错误条目数）")
            attack_btn = gr.Button("执行污染注入", variant="stop")
            poisoned_doc = gr.HighlightedText(label="污染后检索结果",
                                              color_map={"POISON": "red", "CLEAN": "green"})

        # 右侧解毒模块
        with gr.Column(variant="panel"):
            gr.Markdown("### 🛡️ 检索解毒过滤")
            detox_btn = gr.Button("执行毒性过滤", variant="primary")
            dirty_answer = gr.Textbox(label="净化前回答", lines=4, interactive=False)
            clean_answer = gr.Textbox(label="净化后回答", lines=4, interactive=False)
            debug_output = gr.DataFrame(label="解毒过程数据", headers=["文本", "可疑度"], datatype=["str", "number"])

    poisoned_txt = gr.Textbox(visible=False)
    clean_txt = gr.Textbox(visible=False)
    # 事件绑定
    attack_btn.click(
        poison_retrieval,
        inputs=[query_input, intensity],
        outputs=[poisoned_doc, poisoned_txt, clean_txt]
    )
    detox_btn.click(
        detoxify_retrieval,
        inputs=[query_input, poisoned_txt, clean_txt, poisoned_doc],
        outputs=[dirty_answer, clean_answer, debug_output, poisoned_doc]
    )

    # 示例数据
    gr.Examples(
        examples=[
            ["who recorded i can't help falling in love with you", 3],
        ],
        inputs=[query_input, intensity]
    )

demo.launch()
