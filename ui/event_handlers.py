"""
事件处理函数
"""
from core.model_manager import load_model
from utils.text_utils import detect_evolution_mode, extract_problems_from_text
from config import API_CONFIG, EVOLUTION_CONFIG, GENERATION_CONFIG


def update_api_config(api_key, api_32b, api_14b):
    """更新API配置"""
    API_CONFIG["api_key"] = api_key
    API_CONFIG["qwen_32b_api_url"] = api_32b
    API_CONFIG["qwen_14b_api_url"] = api_14b
    return "API配置已更新"


def update_evolution_config(enable, keywords, batch, lr):
    """更新自我演化配置"""
    EVOLUTION_CONFIG["enable_self_evolution"] = enable
    EVOLUTION_CONFIG["evolution_keywords"] = [k.strip() for k in keywords.split(",") if k.strip()]
    EVOLUTION_CONFIG["evolution_batch_size"] = batch
    EVOLUTION_CONFIG["learning_rate"] = lr
    return "自我演化配置已更新"


def detect_mode(prompt):
    """检测当前模式"""
    if not prompt:
        return "**当前模式：** 等待输入..."
    
    should_evolve, problems = detect_evolution_mode(prompt)
    
    if should_evolve:
        if problems:
            return f"**当前模式：** 批量自我演化模式（检测到{len(problems)}个问题）"
        else:
            return "**当前模式：** 单问题自我演化模式"
    
    return "**当前模式：** 普通代码生成模式"


def test_problem_extraction(prompt):
    """测试问题提取"""
    should_evolve, problems = detect_evolution_mode(prompt)
    
    if not should_evolve:
        return "未检测到自我演化关键词。"
    
    if not problems:
        return "检测到自我演化关键词，但没有提取到问题。"
    
    result = f"检测到自我演化模式\n"
    result += f"提取到 {len(problems)} 个问题：\n\n"
    
    for i, problem in enumerate(problems, 1):
        result += f"{i}. {problem}\n"
    
    result += f"\n提示：点击'执行自我演化'按钮开始批量训练。"
    return result