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

import logging
import os
import random
import time
from typing import Optional, Dict, Any, List
from ...config.config import CLICK_ACTION_TYPES, EXC_INIT_APP, HOTKEY_ACTION_TYPES, SCROLL_ACTION_TYPES, TOTAL_ACTION_TYPES, TYPE_ACTION_TYPES, OS_TYPE, UBUNTU_APP_NAMES
from ...config.prompt import SYS_TASK_FOLLOWUP_PERSONA_ON_UBUNTU, SYS_CROSS_APP_NEXT_TASK_PREDICTOR_ON_UBUNTU, SYS_SUBTASK_SUMMARY, SYS_TASK_SUMMARY, RANDOM_ACTION_PROMPT
from ...utils.utils import save_image
from ..reverse.image_ssim_calculator import get_image_ssim
from ..reverse_inference import process_trajectory
from ..reverse.task_correction import task_error_correction
from .action_generator import get_random_element
import json
import re
from PIL import Image
from io import BytesIO
import numpy as np

logger = logging.getLogger("andom_walker")

def parse_action_from_string(action: str):
    match = re.search(r"(\w+)\(", action)
    action_type = match.group(1)
    action_type = action_type.upper()
    result_map = {}
    if action_type == 'WAIT':
        result_map = {} 
    elif action_type in CLICK_ACTION_TYPES:
        if action_type == 'DRAG':
            # 定义正则表达式模式
            pattern = re.compile(r"start_point='<point>(\S+)\s+(\S+)</point>', end_point='<point>(\S+)\s+(\S+)</point>'")
            # 查找匹配项
            match = pattern.search(action)

            if match:
                x1, y1, x2, y2 = match.groups()
                result_map = {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                }
                print(f'action_type: {action_type}, {result_map}')
        else:
            pattern = re.compile(r"\w+\(point='<point>(\S+)\s+(\S+)</point>'\)")
            match = pattern.search(action)
            if action_type == 'RIGHT_SINGLE':
                button = 'right'
            else:
                button = 'left'
            
            if match:
                # 将匹配结果转换为列表
                x, y = match.groups()

                result_map = {
                    "x": int(x),
                    "y": int(y),
                    "button": button,
                }
                print(f'action_type: {action_type}, {result_map}')
    elif action_type in TYPE_ACTION_TYPES:
        pattern = re.compile(r"(\w+)\(content='(.*?)'\)")
        match = pattern.search(action)
        if match:
            content = match.group(2)
            result_map = {"text": content}
            print(f'action_type: {action_type}, {result_map}')
    elif action_type in HOTKEY_ACTION_TYPES:
        pattern = re.compile(r"\w+\(key='(.*?)'\)")
        match = pattern.search(action)
        if match:
            key = match.group(1)
            key_list = key.split(' ')
            result_map = {"keys": key_list}
            print(f'action_type: {action_type}, {result_map}')
    elif action_type in SCROLL_ACTION_TYPES:
        pattern = re.compile(r"\w+\(point='<point>(\S+)\s+(\S+)</point>', direction='(.*?)', amount='(\S+)'")
        match = pattern.search(action)
        if match:
            x = match.group(1)
            y = match.group(2)
            direction = match.group(3)
            amount = match.group(4)
            result_map = {"point": {"x": int(x), "y": int(y)}, "direction": direction, "amount": int(amount)}
            print(f'action_type: scroll, {result_map}')
    else:
        raise ValueError(f"Unknown action type: {action_type}")
            
    
    action_json_dict = {
        "action_type": action_type,
        "parameters": result_map,
    }
    return action_json_dict

def parse_action_from_ocr_detect(content, action, ui_elements):
    if content.isdigit():
        if len(ui_elements) > int(content):
            points = ui_elements[int(content)]
            result_map = {"x": int((points[0] + points[2]) // 2), "y": int((points[1] + points[3]) // 2)}
    elif isinstance(content, str):
        result_map = {"text": content}
    elif isinstance(content, list):
        if len(content) == 2:
            result_map = {
                    "x": int(content[0]),
                    "y": int(content[1]),
                    "button": "left",
                }
    
    action_type = action.upper()
        
    action_json_dict = {
        "action_type": action_type,
        "parameters": result_map,
    }
    return action_json_dict

def generate_random_action_from_ui(agent, env, ocr, screen, init_app_name, idx, screenshot_dir):
    """
    从UI元素中生成随机动作

    参数:
        agent: AI代理对象
        ui_elements: 检测到的UI元素列表
        logical_screen_size: 逻辑屏幕大小

    返回:
        随机动作
    """
    ui_elements, logits = ocr.detect_gui(screen)
    
    ui_bboxes = get_random_element(ui_elements, logits)
    
    image_detect_array = ocr.draw_bbox(screen, ui_bboxes, logits)
    
    # save ocr img
    # result_image = Image.fromarray(image_detect_array)
    # result_image.save(os.path.join(screenshot_dir, f"{idx}_before_ocr.png"))

    random_action_prompt = RANDOM_ACTION_PROMPT.format(
        os_type=OS_TYPE,
        init_app_name=init_app_name,
        ocr_bbox_nums=len(ui_bboxes),
        action_types=TOTAL_ACTION_TYPES,
        app_names=UBUNTU_APP_NAMES,
    )
    
    ocr_items, prompt_token, completion_token, retry_counter = agent.predict_random_rewalk(random_action_prompt, [image_detect_array])
    random.shuffle(ocr_items)

    bbox_id = int()
    for idx, (key, json_content) in enumerate(ocr_items):
        if not key.isdigit():
            continue
        if "action" not in json_content or json_content["action"] == 'dropped':
            continue
        key = int(key)
        # print(f'随机OCR框 key: {key}, json_content: {json_content}')
        if key >= len(ui_bboxes):
            continue
        bbox_id = idx
        break
    if bbox_id is None:
        print('随机OCR框没有找到匹配的动作')
        print()
        print(ocr_items)
        bbox_id = 0
    action_type = ocr_items[bbox_id][1]["action"].upper()
    ui_bbox = ui_bboxes[bbox_id]
    point = [int((ui_bbox[0] + ui_bbox[2]) // 2), int((ui_bbox[1] + ui_bbox[3]) // 2)]
    software_tag = ocr_items[bbox_id][1]['tag']
    
    if action_type in CLICK_ACTION_TYPES and action_type != 'DRAG':
        if action_type == 'RIGHT_SINGLE':
            button = 'right'
        else:
            button = 'left'
        
        result_map = {
            "x": point[0],
            "y": point[1],
            "button": button,
        }  
        logger.info(f'RANDOM WALK: action_type: {action_type}, {result_map}')
        action = {"action_type": action_type, "parameters": result_map}
        env.controller.execute_gui_action(action)
        
        action_json_output = json.dumps({"action_type": action_type, "parameters": {
            "button": button,
            "x": point[0],
            "y": point[1],
            "x_min": ui_bbox[0],
            "y_min": ui_bbox[1],
            "x_max": ui_bbox[2],
            "y_max": ui_bbox[3],
        }})
    elif action_type in TYPE_ACTION_TYPES:
        pre_result_map = {
            "x": point[0],
            "y": point[1],
            "button": "left",
        }  
        pre_action = {"action_type": "CLICK", "parameters": pre_result_map}
        env.controller.execute_gui_action(pre_action)
        text = ocr_items[bbox_id][1]['text']
        action = {"action_type": action_type, "parameters": {"text": text}}
        env.controller.execute_gui_action(action)
        action_json_output = json.dumps({"action_type": action_type, "parameters": {
            "text": text,
            "x_min": ui_bbox[0],
            "y_min": ui_bbox[1],
            "x_max": ui_bbox[2],
            "y_max": ui_bbox[3],
        }})
    else:
        raise TypeError(f"Unknown action type: {action_type}")
    
    
    return action_json_output, action_type, software_tag, prompt_token, completion_token, retry_counter


def perform_random_action(
    agent, env, ocr, obs, screenshot_dir: Optional[str] = None, step_data_before: Optional[Dict[str, Any]] = None, i: int = 0, init_app_name: str = "Ubuntu"
) -> Optional[Dict[str, Any]]:
    """
    根据当前界面执行随机动作并记录结果

    参数:
        agent: AI代理对象
        screenshot_dir: 截图保存目录（可选）

    返回:
        Optional[Dict[str, Any]]: 包含动作相关数据的步骤数据字典
    """

    screen_byte = obs["screenshot"]
    screen = Image.open(BytesIO(screen_byte))
    logical_screen_size = screen.size  # 获取图片的宽和高
    
    screen_before = save_image(screen, screenshot_dir, i, "before")
    
    action_json, action_type, software_tag, prompt_token, completion_token, retry_counter = generate_random_action_from_ui(
        agent, env, ocr, screen, init_app_name, i, screenshot_dir
    )
    
    # print(f'最后执行action内容: {action_json}')
    
    if action_json is None:
        print("Could not generate a random action")
        return None
    # agent.env.execute_action(action)
    time.sleep(6.0)
    
    observation = env._get_obs()
    screen_after = save_image(observation['screenshot'], screenshot_dir, i, "after")

    step_data = {
        "step": i,
        "screen_before": screen_before,
        "screen_after": screen_after,
        "action_type": action_type,
        "action": action_type,
        "software_tag": software_tag,
        "forward_prompt_token": prompt_token,
        "forward_completion_token": completion_token,
        "forward_retry_counter": retry_counter,
        "action_json": action_json,
        "timestamp": time.time(),
        "obs": observation,
    }

    print(f"Random action completed and saved")
    return step_data

# TODO 随机游走器参数调整
# 步数调整 min-max
# 判断 status
def execute_random_actions(
    agent,
    env,
    ocr,
    obs,
    max_num_actions: int = 20,
    max_num_guided_actions: int = 20,
    max_num_guided_actions_after_openapp: int = 10,
    screenshot_dir: Optional[str] = None,
    app_name: str = "Unknown App",
    exce_task_completion: bool = True,
    reverse_inference: bool = True,
    summary_inference: bool = True,
    evaluate_trajectory: bool = True,
    random_results: Optional[List[Dict[str, Any]]] = [],
    task_num: int = 0,
    last_stride: bool = False,
    exc_init_action: Optional[str] = None,
) -> Dict[str, Any]:
    """
    执行一系列随机动作并保存结果

    参数:
        agent: AI代理对象
        min_num_actions: 最小随机动作数量
        max_num_actions: 最大随机动作数量
        screenshot_dir: 截图保存目录（可选）
        app_name: 应用名称
        evaluate_trajectory: 是否使用GPT4评估轨迹有效性
        save_dataframe: 是否保存为DataFrame格式（默认False）
        dataframe_output_path: DataFrame输出文件路径

    返回:
        Dict[str, Any]: 包含随机动作结果和评估信息的字典
    """
    step_id = 1
    init_app_name = "Ubuntu"
    
    if random_results is not None:
        # 跨app预先执行
        num_actions = len(random_results["trajectory"])
        step_id += num_actions
        random_actions = random_results["trajectory"]
        task_info = task_propose(random_results["summary"],random_actions[-1]['screen_after'],agent, task_num)
        # app_name = task_info['app']
        # obs = open_app(app_name, env, agent, ocr, random_actions[-1]['screen_after'], screenshot_dir, step_id)
        # step_id += 1
        action_data, step_id_update, obs, is_done = execute_single_task_without_reset(agent, env, obs, ocr, task_info, step_id, max_num_guided_actions_after_openapp, screenshot_dir)
        step_id = step_id_update
        random_actions.extend(action_data)
        agent.reset()
        
    else:
        random_actions = []
        num_actions = 0
        
    print(f"Starting to execute {max_num_actions} random actions...")
     
    # 随机游走
    for i in range(max_num_actions):       
        print(f"\nExecuting random action {len(random_actions)+1}/{max_num_actions + num_actions}")
        before_step_data = random_actions[-1] if len(random_actions) > 0 else None
        step_data = perform_random_action(agent, env, ocr, obs, screenshot_dir, before_step_data, step_id, init_app_name)
        if step_data:
            step_data["step"] = len(random_actions) + 1
            obs = step_data['obs']
            # print(f"reverse inference {len(random_actions)+1}/{max_num_actions + num_actions}")
            random_actions.extend(process_trajectory([step_data], app_name, agent, screenshot_dir))
        step_id += 1
        time.sleep(1.0)

    # 任务补全
    if exce_task_completion:
        # logger.info("task completion...")
        task_info = task_propose(random_actions[-1]['purpose'],random_actions[-1]['screen_after'], agent, task_num)
        # print(f"task_info: {task_info}")
        action_data, step_id_update, obs, is_done = execute_single_task_without_reset(agent, env, obs, ocr, task_info, step_id, max_num_guided_actions, screenshot_dir)
        step_id = step_id_update
        reverse_actions_loc_id = len(random_actions)
        random_actions.extend(action_data)
    else:
        task_info = ""

    # 反向推理
    if reverse_inference and random_actions:
        random_actions[num_actions:] = process_trajectory(random_actions[num_actions:], app_name, agent, screenshot_dir)
    
    # 总结子任务
    if summary_inference:
        sub_instruction = " ".join([action['high_level_instruction'] for action in random_actions[num_actions:]])
        sub_screen_after = [action['screen_after'] for action in random_actions[num_actions:]]
        sub_processed_imgs = [load_image_as_ndarray(img) for img in sub_screen_after]
        # summary of sub-tasks
        sub_summary = generate_summary(sub_instruction, sub_processed_imgs, agent)
        random_actions[-1]['summary'] = sub_summary
        
        # 总结生成instruction
        if last_stride:
            instruction = " ".join([action['high_level_instruction'] for action in random_actions])
            screen_after = [action['screen_after'] for action in random_actions]
            processed_imgs = [load_image_as_ndarray(img) for img in screen_after]
            summary_list = [action['summary'] for action in random_actions if 'summary' in action]
            summary = generate_summary(instruction, processed_imgs, agent, False, summary_list)
    else:
        summary = ""

    # 构建返回结果
    result = {
        "trajectory": random_actions,
        "app_name": app_name,
        "num_actions_executed": len(random_actions),
        "task_info": task_info,    
        "summary": summary if last_stride else "",
    }

    return result, obs

def task_propose(task_history, img, agent, task_num=0):
    if task_num == 0:
        sys_prompt = SYS_TASK_FOLLOWUP_PERSONA_ON_UBUNTU.format(os_type=OS_TYPE)
    else:
        # sys_prompt = SYS_NEXT_TASK_PREDICTOR
        sys_prompt = SYS_CROSS_APP_NEXT_TASK_PREDICTOR_ON_UBUNTU.format(os_type=OS_TYPE, app_names=EXC_INIT_APP)
    user_prompt = f"Given the action history {task_history}, what would be a followup task?"
    prompt = sys_prompt + user_prompt
    # if failed_task:
    #     user_prompt += f" Note that these tasks {failed_task} are too hard for the agent, propose a simplier one."
    img = load_image_as_ndarray(img)
    
    while True:
        try:
            # llm_output = call_gpt(sys_prompt, user_prompt, img)
            response = agent.predict_mm(prompt, [img])
            task_info = parse_json(response[0],fields=["task","app"])
            logger.info(f"task_info: {task_info}")  
            task_info["prompt_token"] = response[1]
            task_info["completion_token"] = response[2]
            task_info["retry_counter"] = response[3]
            break
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(10)
            continue
    return task_info

def open_app(app_name, env, agent, ocr, img, screenshot_dir, i):

    ui_elements, logits = ocr.detect_gui(img)
    # 将 ui_elements 转换为指定格式
    image_detect_array = ocr.draw_bbox(img, ui_elements, logits)

    user_prompt = f"bbox id num: {len(ui_elements)}, return the id from 0 to {len(ui_elements) - 1}"
    sys_prompt = SYS_TASK_OPENAPP
    prompt = sys_prompt + user_prompt
    # if failed_task:
    #     user_prompt += f" Note that these tasks {failed_task} are too hard for the agent, propose a simplier one."
    
    screen_before = save_image(img, screenshot_dir, f"openapp_{i}", "before")
    
    try:
        # llm_output = call_gpt(sys_prompt, user_prompt, img)
        response = agent.predict_mm(prompt, [image_detect_array])
        input_string = response[0]
        prompt_token = response[1]
        completion_token = response[2]
        retry_counter = response[3]
        response_json = parse_json(input_string)
        
        thoughts = response_json.get("thoughts", None)
        action = response_json.get("action", None)
        content = response_json.get("content", None)
        
        action_json_dict = parse_action_from_ocr_detect(content, action, ui_elements)
        obs = env.step(action_json_dict, pause=2)
        
        screen_after = save_image(obs, screenshot_dir, f"openapp_{i}", "after")
        
        return obs
        
    except Exception as e:
        print(f"Error: {e}. Retrying...")
        time.sleep(10)

    time.sleep(3.0)


def parse_json(llm_output, fields=None):
    """
    从 LLM 输出中提取 JSON，并可选择只保留特定字段（如 task 和 app）。

    Args:
        llm_output (str or tuple): LLM 返回的字符串或三元组。
        fields (list[str], optional): 只提取的字段列表，如 ['task', 'app']。

    Returns:
        dict or None: 提取结果，可能是完整 JSON，也可能是字段子集。
    """
    # 兼容 tuple 类型输入
    if isinstance(llm_output, tuple):
        llm_output = llm_output[0]
    # 匹配 markdown 格式 ```json { ... } ```
    # match = re.search(r"```json\s*(\{.*?\})\s*```", llm_output, re.DOTALL)
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", llm_output, re.DOTALL)
    if not match:
        # 直接尝试匹配任何JSON对象
        match = re.search(r"\{.*?\}", llm_output, re.DOTALL)

    if match:
        json_str = match.group(1)
        try:
            data = json.loads(json_str)
            # print("Extracted JSON:", data)
            if fields:
                return {key: data.get(key, "") for key in fields}
            return data
        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", e, llm_output)
            return None
    else:
        print("No JSON block found.", llm_output)
        return None


def load_image_as_ndarray(img) -> np.ndarray:
    """
    将图像路径 / PIL.Image / ndarray 输入统一转换为 RGB 的 numpy 数组。
    """
    if isinstance(img, np.ndarray):
        return img
    elif isinstance(img, Image.Image):
        return np.array(img.convert("RGB"))
    elif isinstance(img, str):
        if not os.path.exists(img):
            raise FileNotFoundError(f"图片路径不存在: {img}")
        img_pil = Image.open(img).convert("RGB")
        return np.array(img_pil)
    else:
        raise TypeError(f"不支持的图像类型: {type(img)}")
    
def execute_single_task_without_reset(agent, env, obs, ocr, task_item, step_id, max_num_guided_actions, screenshot_dir):
    """
    执行单个任务

    参数:
        env: 环境对象
        task_item: 当前任务项
        aw_instructions: 所有任务指令(用于更新)
        agent: AI代理对象
    """
    # 重置环境
    # agent.env.reset(go_home=True)

    # 获取任务参数
    if isinstance(task_item, dict):
        app_name = task_item["app"]
        pre_action = task_item.get("pre_action", [])  # 使用get方法，默认为空列表
        instruction = task_item.get("task", "")  # 使用get方法，默认为空字符串
    else:
        app_name = "Unknown"
        pre_action = []
        instruction = task_item

    # 打开对应的应用
    gpt_traj, step_id_update, obs, is_done = execute_task(
        agent, env, obs, ocr, app_name, instruction, pre_action, step_id, max_num_guided_actions, screenshot_dir
    )

    return gpt_traj, step_id_update, obs, is_done

def execute_task(agent, env, obs, ocr, app_name, instruction, pre_action, step_id, max_num_guided_actions, screenshot_dir):
    """
    打开应用并执行任务

    参数:
        agent: AI代理对象
        app_name: 应用名称
        instruction: 任务指令（可能为空）
        pre_action: 预执行动作列表（可能为空）

    返回:
        tuple: (是否完成, 轨迹数据列表)
    """
    # 处理指令（如果为空，提供默认指令）
    if not instruction or instruction.strip() == "":
        return True, []
    else:
        print("Goal: " + str(instruction))

    # 执行任务
    is_done = False
    gpt_traj = []
    id = step_id
    correct_count = 0

    for i, _ in enumerate(range(max_num_guided_actions)):  # 最多执行15步
        # 调用execute_task_step并获取更新后的轨迹和完成状态
        obs, gpt_traj, step_done = execute_task_step(
            agent, env, obs, ocr, instruction, gpt_traj, screenshot_dir, id
        )
        id += 1
        if gpt_traj[-1]["action"] not in ("FINISHED", "WAIT", "HOTKEY") and get_image_ssim(gpt_traj[-1]["screen_before"], gpt_traj[-1]["screen_after"]) > 0.99:
            if correct_count <= 0:
                instruction =  task_error_correction(gpt_traj, instruction, agent)
                print("Due to action no changes, new goal: ", instruction)
                correct_count += 1
                continue
            else:
                is_done = True
                break
        
        # 检查任务是否完成
        if step_done:
            is_done = True
            break

    return gpt_traj, id, obs, is_done

def execute_task_step(agent, env, obs, ocr, instruction, gpt_traj, screenshot_dir, step_id):
    """
    执行任务的单个步骤

    参数:
        agent: AI代理对象
        instruction: 任务指令
        gpt_traj: 当前轨迹数据
        step_index: 步骤索引
        app_name: 应用名称

    返回:
        tuple: (更新后的轨迹数据, 任务是否完成)
    """
    # 获取执行前的环境状态

    screen_byte = obs["screenshot"]
    screen = Image.open(BytesIO(screen_byte))
    # logical_screen_size = screen.size  # 获取图片的宽和高
    # ui_elements, logits = ocr.detect_gui(screen)
    
    screen_before = save_image(screen, screenshot_dir, step_id, "before")

    # 执行一步
    print(f'补全...调用ui_tars api')
    thoughts, action, summary_problem, prompt_token, completion_token, retry_counter = agent.predict(
            instruction,
            obs,
            ocr,
            step_id,
            screenshot_dir,
        )
    print(f'补全...返回api结果')
    
    action_json_dict = parse_action_from_string(action)
    action_type = action_json_dict['action_type']
    
    obs = env.step(action_json_dict, pause=6)
    
    screen_after = save_image(obs['screenshot'], screenshot_dir, step_id, "after")
    
    # 保存步骤数据
    step_data = {
        "step": step_id, ####修改
        "action":action_type,
        "action_json": json.dumps(action_json_dict),
        "screen_before": screen_before,
        "screen_after": screen_after,
        "sub_instruction":thoughts,
        "goal":instruction,
        "high_level_instruction":"",
        "forward_prompt_token": prompt_token,
        "forward_completion_token": completion_token,
        "retry_counter": retry_counter,
        # "screen_before_som": screen_before_som,
        # "action_reason": action_reason,
        "summary_problem": summary_problem,
    }
    gpt_traj.append(step_data)

    return obs, gpt_traj, True if action_type == 'FINISHED' else False  # 返回更新后的轨迹和完成状态


def generate_summary(high_level_instruction, img, agent, sub_summary=True, summary_list=[]):
    if sub_summary:
        sys_prompt = SYS_SUBTASK_SUMMARY.format(os_type=OS_TYPE)
        # user_prompt = f"Given the subtasks history {task_history} and the final screenshot, what would be a single task description that will be accomplished by performing these subtasks in the given sequence?"
        user_prompt = f"Given the set of screenshots of actions and high-level instructions {high_level_instruction}, what would be a single task description that will be accomplished by performing these actions in the given sequence?"
    else:
        sys_prompt = SYS_TASK_SUMMARY.format(os_type=OS_TYPE)
        user_prompt = f"Given the set of screenshots of actions, high-level instructions {high_level_instruction}, and sub_task summary {summary_list}  what would be a single task description that will be accomplished by performing these actions in the given sequence?"
    prompt = sys_prompt + user_prompt
    
    prompt_token_total = 0
    completion_token_total = 0
    retry_counter = 0
    while True:
        try:
            summary = agent.predict_mm(prompt, img)
            parsed_json = parse_json(summary[0])
            prompt_token_total += summary[1]
            completion_token_total += summary[2]
            if parsed_json is None:
                print("Failed to parse JSON, retrying...")
                time.sleep(10)
                continue
            task_info = parsed_json.get('task', "")
            result = {
                "task": task_info,
                "prompt_token": prompt_token_total,
                "completion_token": completion_token_total,
                "retry_counter": retry_counter,
            }
            break
        except Exception as e:
            retry_counter += 1
            print(f"Error: {e}. Retrying...")
            time.sleep(10)
            continue

    return result