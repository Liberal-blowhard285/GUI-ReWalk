# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import string

def get_gpt_generated_text(agent, element) -> str:
    """
    使用GPT生成适合当前上下文的输入文本

    参数:
        agent: AI代理对象，用于访问GPT
        element: 要输入文本的UI元素

    返回:
        str: 生成的文本
    """
    try:
        hint_text = getattr(element, "hint_text", "") or ""
        prompt = f"""
        Generate random searches based on screenshots, but the content must be meaningful and relevant to the current screenshot
        - Hint text: "{hint_text}"
        Only return the text to input, without any explanation, formatting or punctuation.
        Separate different words with spaces.
        """
        screenshot = agent.get_post_transition_state().pixels.copy()

        if hasattr(agent, "llm") and agent.llm:
            response_text, _, _ = agent.llm.predict_mm(prompt, [screenshot])
            response_text = response_text.replace("\"","")
            print(f"Generated text: '{response_text}'")
            return response_text
    except Exception as e:
        print(f"Error generating text with GPT: {e}")

    random_text = "".join(
        random.choices(string.ascii_letters + string.digits, k=random.randint(5, 10))
    )
    print(f"Falling back to random text: '{random_text}'")
    return random_text


def get_meaningful_click_element(agent, ui_elements, logical_screen_size):
    """获取有意义的点击元素列表，过滤掉明显无语义的系统UI"""
    
    # 定义明显无语义的元素模式
    meaningless_patterns = [
        "battery", "percent", "电池",
        "wifi signal", "phone", "bars", "信号",
        "notification:", "通知",
        "clock", "时间", "01:", "02:", "03:", "04:", "05:", "06:", "07:", "08:", "09:", "10:", "11:", "12:",
        "13:", "14:", "15:", "16:", "17:", "18:", "19:", "20:", "21:", "22:", "23:", "00:",
        "status bar", "状态栏", "close", "关闭"
    ]
    
    def is_system_ui(elem):
        """判断是否为系统UI元素"""
        text = getattr(elem, "text", "") or ""
        content_desc = getattr(elem, "content_description", "") or ""
        combined_text = (text + " " + content_desc).lower()
        
        # 检查是否匹配无语义模式
        for pattern in meaningless_patterns:
            if pattern.lower() in combined_text:
                return True
        return False
    
    def is_valid_clickable(elem):
        """判断是否为有效的可点击元素"""
        return (
            m3a_utils.validate_ui_element(elem, logical_screen_size)
            and not is_system_ui(elem)
        )
    
    # 筛选有效的可点击元素
    valid_clickable_elements = [
        (i, elem)
        for i, elem in enumerate(ui_elements)
        if is_valid_clickable(elem)
    ]
    
    if not valid_clickable_elements:
        print("No meaningful clickable elements found")
        return []
    
    # 如果元素较少，直接返回
    if len(valid_clickable_elements) <= 3:
        print(f"Found {len(valid_clickable_elements)} meaningful clickable elements")
        return valid_clickable_elements
    
    # 使用模型排除无意义的元素
    try:
        # 构建元素描述列表
        element_descriptions = []
        for i, elem in valid_clickable_elements:
            text = getattr(elem, "text", "") or ""
            content_desc = getattr(elem, "content_description", "") or ""
            class_name = getattr(elem, "class_name", "") or ""
            resource_name = getattr(elem, "resource_name", "") or ""
            
            description = f"Index {i}: text='{text}', desc='{content_desc}', class='{class_name}', resource='{resource_name}'"
            element_descriptions.append(description)
        
        # 构建排除逻辑的提示
        prompt = f"""
        Analyze the following UI elements and identify which elements should be excluded (meaningless elements):
        {chr(10).join(element_descriptions)}
        
        Please list the indices of elements that should be excluded (comma-separated). These elements are typically:
        - System notifications
        - Status bar elements
        - Decorative elements without actual functionality
        
        Only return the index numbers to exclude (comma-separated). If no elements need to be excluded, return "none".
        """
        
        screenshot = agent.get_post_transition_state().pixels.copy()
        
        if hasattr(agent, "llm") and agent.llm:
            response_text, _, _ = agent.llm.predict_mm(prompt, [screenshot])
            
            # 解析模型返回的要排除的index
            exclude_indices = set()
            try:
                response_text = response_text.strip().lower()
                if response_text != "none" and response_text:
                    # 解析逗号分隔的index
                    indices_str = response_text.replace(" ", "").split(",")
                    for idx_str in indices_str:
                        if idx_str.isdigit():
                            exclude_indices.add(int(idx_str))
                    
                    print(f"Model suggested excluding indices: {exclude_indices}")
            except Exception as e:
                print(f"Error parsing exclude indices: {e}")
            
            # 过滤掉被排除的元素
            filtered_elements = [
                (i, elem) for i, elem in valid_clickable_elements
                if i not in exclude_indices
            ]
            
            # 如果过滤后还有元素，返回过滤后的列表
            if filtered_elements:
                print(f"After model filtering: {len(filtered_elements)} elements remaining")
                return filtered_elements
            else:
                print("Model excluded all elements, falling back to original list")
                return valid_clickable_elements
        else:
            print("No model available, using rule-based filtering only")
            return valid_clickable_elements
            
    except Exception as e:
        print(f"Error using model to exclude elements: {e}")
        return valid_clickable_elements


def get_random_element(ui_elements, logits):
    # 若 ui_elements 元素数量少于 10 个，全量选取
    if len(ui_elements) < 10:
        ui_bboxes = ui_elements
    else:
        ui_bboxes = random.sample(ui_elements, min(10, len(ui_elements)))
    return ui_bboxes