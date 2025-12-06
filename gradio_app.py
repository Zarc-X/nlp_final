# 使用 Qwen2.5-Coder 模型的 Gradio 界面
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gradio as gr

# 全局变量存储模型和分词器
model = None
tokenizer = None
device = None

def load_model():
    """加载模型和分词器"""
    global model, tokenizer, device
    
    # 加载模型和分词器
    model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    
    print("正在加载模型和分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 确定设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )
    model = model.to(device)
    model.eval()  # 设置为评估模式
    print("模型加载完成！")
    return "模型加载完成！"

def generate_code(prompt, system_prompt, max_tokens, temperature, top_p):
    """生成代码的函数"""
    if model is None or tokenizer is None:
        return "错误：模型尚未加载，请先点击'加载模型'按钮。"
    
    if not prompt or prompt.strip() == "":
        return "错误：请输入代码生成提示。"
    
    try:
        # 准备对话消息
        messages = [
            {"role": "system", "content": system_prompt if system_prompt else "你是一个专业的编程助手，擅长编写和解释代码。"},
            {"role": "user", "content": prompt},
        ]
        
        # 应用聊天模板
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 将文本转换为模型输入
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        
        # 生成代码
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                do_sample=True
            )
        
        # 提取生成的文本（去掉输入部分）
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
        
    except Exception as e:
        return f"生成代码时出错：{str(e)}"

# 创建 Gradio 界面
with gr.Blocks(title="Qwen2.5-Coder 代码生成器", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Qwen2.5-Coder 代码生成器")
    gr.Markdown("使用 Qwen2.5-Coder 模型生成代码。请先加载模型，然后输入提示词生成代码。")
    
    with gr.Row():
        with gr.Column():
            load_btn = gr.Button("🔄 加载模型", variant="primary", size="lg")
            load_status = gr.Textbox(label="模型状态", interactive=False)
            
            with gr.Accordion("⚙️ 高级设置", open=False):
                system_prompt_input = gr.Textbox(
                    label="系统提示词",
                    value="你是一个专业的编程助手，擅长编写和解释代码。",
                    lines=2,
                    placeholder="输入系统提示词..."
                )
                max_tokens_input = gr.Slider(
                    label="最大生成token数",
                    minimum=50,
                    maximum=2048,
                    value=512,
                    step=50
                )
                temperature_input = gr.Slider(
                    label="Temperature (创造性)",
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1
                )
                top_p_input = gr.Slider(
                    label="Top-p (核采样)",
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.05
                )
        
        with gr.Column():
            prompt_input = gr.Textbox(
                label="代码生成提示",
                placeholder="例如：请用Python编写一个快速排序算法。",
                lines=5
            )
            generate_btn = gr.Button("✨ 生成代码", variant="primary", size="lg")
            output = gr.Code(
                label="生成的代码",
                language="python",
                lines=20
            )
    
    # 绑定事件
    load_btn.click(
        fn=load_model,
        outputs=load_status
    )
    
    generate_btn.click(
        fn=generate_code,
        inputs=[prompt_input, system_prompt_input, max_tokens_input, temperature_input, top_p_input],
        outputs=output
    )
    
    # 示例提示词
    gr.Examples(
        examples=[
            ["请用Python编写一个快速排序算法。"],
            ["用Python实现一个简单的HTTP服务器。"],
            ["写一个函数来计算斐波那契数列的第n项。"],
            ["用Python实现一个简单的计算器类。"],
        ],
        inputs=prompt_input
    )

if __name__ == "__main__":
    # 启动 Gradio 界面
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)

