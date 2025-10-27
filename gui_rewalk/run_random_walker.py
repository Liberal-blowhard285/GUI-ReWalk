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

from gui_rewalk.env.desktop_gui_gen_env import DesktopGUIGenEnv
from gui_rewalk.env.gui_gen_agent import GUIGenAgent
from gui_rewalk.env.gui_detector import GUIDecorator
from gui_rewalk.src.core.random_walker.walker import execute_random_actions
from gui_rewalk.src.config.config import setup_environment
from gui_rewalk.src.core.reward_evaluator import evaluate_random_walker_trajectory
from gui_rewalk.src.utils.static_token import sum_tokens
from gui_rewalk.src.utils.argparser import get_config
import argparse
import datetime
import json
import traceback
import logging
import os
import sys
import time

os.makedirs("logs", exist_ok=True)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

file_handler = logging.FileHandler(
    os.path.join("logs", "normal-{:}.log".format(datetime_str)), encoding="utf-8"
)
debug_handler = logging.FileHandler(
    os.path.join("logs", "debug-{:}.log".format(datetime_str)), encoding="utf-8"
)
stdout_handler = logging.StreamHandler(sys.stdout)
sdebug_handler = logging.FileHandler(
    os.path.join("logs", "sdebug-{:}.log".format(datetime_str)), encoding="utf-8"
)

file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(logging.INFO)
sdebug_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s"
)
file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)
sdebug_handler.setFormatter(formatter)

stdout_handler.addFilter(logging.Filter("desktopenv"))
sdebug_handler.addFilter(logging.Filter("desktopenv"))

logger.addHandler(file_handler)
logger.addHandler(debug_handler)
logger.addHandler(stdout_handler)
logger.addHandler(sdebug_handler)
#  }}} Logger Configs #

logger = logging.getLogger("desktopenv.experiment")


def get_desktop_env(cfg_args, agent):
    if cfg_args.vm_provider == "vmware":
        return DesktopGUIGenEnv(
        provider_name=cfg_args.vm_provider,
        path_to_vm=cfg_args.path_to_vm,
        action_space=agent.action_space,
        screen_size=(cfg_args.screen_width, cfg_args.screen_height),
        headless=cfg_args.headless,
        require_a11y_tree=True,
        os_type = "Ubuntu",
    )
    else:
        raise ValueError(f"Unsupported vm_provider: {cfg_args.vm_provider}")

def test(args: argparse.Namespace,) -> None:
    logger.info("Args: %s", args)

    agent = GUIGenAgent(
            model=args.model,
            model_version=args.model_version,  
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            temperature=args.temperature,
            action_space=args.action_space,
            observation_type=args.observation_type,
            enable_ocr=True,
            max_trajectory_length=args.max_trajectory_length,
            max_retry=3,
            enable_thinking=args.enable_thinking,
            use_ark=args.use_ark
        )

    env = get_desktop_env(args, agent)
    
    ocr = GUIDecorator(args.ocr_model_path, BOX_TRESHOLD = 0.05, device = 'cpu', yolo_print=False)
    
    result_dir = os.path.join(
        args.result_dir,
        args.action_space,
        args.model
    )
    os.makedirs(result_dir, exist_ok=True)

    example_json_path = os.path.join(result_dir, "example.json")
    if not os.path.exists(example_json_path):
        example_content = [{"task_id": f"{i}", "app_name": "Unknown App"} for i in range(args.task_num)]
        with open(example_json_path, 'w', encoding='utf-8') as f:
            json.dump(example_content, f, ensure_ascii=False, indent=4)
    aw_instructions = json.load(open(example_json_path, "r"))

    # 执行任务
    trajectorys = []
    time_list = []
    for task_item in aw_instructions:
        screenshot_dir = os.path.join(result_dir, task_item["task_id"], "screenshots")
        os.makedirs(screenshot_dir, exist_ok=True)
        
        show_task_stats(aw_instructions)
        time_start = time.time()

        if "random_actions" in task_item:
            trajectorys = trajectorys + task_item["random_actions"]["trajectory"]
            print(f"Task {task_item['task_id']} already has random actions, skipping...")
            continue
        
        try:
            random_results = None

            runtime_logger = setup_logger(result_dir)
            agent.reset(runtime_logger)
            env.reset()
            time.sleep(60) # Wait for the environment to be ready
            env.controller.start_recording()
            
            # logger.info("初始化..点击取消系统软件更新")
            # init_action = {"action_type": "CLICK", "parameters": {"button": "left", "x": 1299, "y": 479}}
            # env.controller.execute_action(init_action)
            time.sleep(2)
            obs = env._get_obs()
                
            
            if args.random_walker:
                app_name = task_item.get("app_name", "Unknown App")
                for i in range(0, args.random_walk_cross_app):
                    logger.info(f'Check structure: No.{i} walking step, waiting for filling...')
                    # 执行随机动作并生成指令
                    random_results, obs = execute_random_actions(
                        agent,
                        env,
                        ocr,
                        obs,
                        app_name=app_name,
                        max_num_actions=args.max_random_actions,
                        max_num_guided_actions=args.max_guided_actions,
                        max_num_guided_actions_after_openapp=args.max_guided_actions_after_openapp,
                        exce_task_completion=args.exce_task_completion, 
                        reverse_inference=args.reverse_inference,
                        summary_inference=args.summary_inference,
                        evaluate_trajectory=EVALUATE_TRAJECTORY,
                        random_results=random_results,
                        screenshot_dir=screenshot_dir,
                        task_num=i
                    )
            
                
            evaluation_result = None
            if random_results:
                try:
                    print("\nEvaluating trajectory effectiveness ...")
                    instruction = random_results["summary"]
                    score, reason, prompt_tokens, completion_tokens, retry_counter = evaluate_random_walker_trajectory(agent, random_results["trajectory"], instruction)
                    
                    evaluation_result = {
                        "score": score,
                        "reason": reason,
                        "instruction": instruction,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "retry_counter": retry_counter
                    }
                    
                    print(f"Trajectory Evaluation - Score: {score}/5")
                    print(f"Reason: {reason}")
                    task_item["evaluation_result"] = evaluation_result
                    
                except Exception as e:
                    print(f"Error during trajectory evaluation: {e}")
                    evaluation_result = {
                        "score": None,
                        "reason": f"Evaluation failed: {str(e)}",
                        "instruction": instruction
                    }

                
                # 如果随机动作执行成功，更新任务项
                if random_results:
                    static_token = sum_tokens(random_results["trajectory"])
                    print(f"Static token sum: {static_token}")
                    trajectorys = trajectorys + random_results["trajectory"]
                    time_end = time.time()
                    total_seconds = time_end - time_start
                    # 使用 divmod 函数计算分钟和剩余秒数
                    minutes, seconds = divmod(total_seconds, 60)
                    formatted_time = f"{int(minutes)}min {int(seconds)}s"
                    task_item["random_actions"] = random_results
                    task_item["prompt_token_cost"] = static_token["prompt_token_sum"]
                    task_item["completion_token_cost"] = static_token["completion_token_sum"]
                    task_item["time"] = formatted_time
                    # 修改 json.dump 写入方式，解决中文乱码问题
                    with open(example_json_path, 'w', encoding='utf-8') as f:
                        json.dump(aw_instructions, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()
            time_end = time.time()
            total_seconds = time_end - time_start
            # 使用 divmod 函数计算分钟和剩余秒数
            minutes, seconds = divmod(total_seconds, 60)
            formatted_time = f"{int(minutes)}min {int(seconds)}s"
            task_item["task_fail"] = "fail"
            task_item["time"] = formatted_time
            # 修改 json.dump 写入方式，解决中文乱码问题
            with open(example_json_path, 'w', encoding='utf-8') as f:
                json.dump(aw_instructions, f, ensure_ascii=False, indent=4)
            time.sleep(10)
            
       
        time_list.append(time_end - time_start)
        logger.info(f"Task {task_item['task_id']} of {len(aw_instructions)} time: {time_end - time_start}s")
        
        
    env.close()
        


def setup_logger(result_dir):
    runtime_logger = logging.getLogger(f"desktopenv.gen_data")
    runtime_logger.setLevel(logging.DEBUG)
    runtime_logger.addHandler(logging.FileHandler(os.path.join(result_dir, "runtime.log")))
    return runtime_logger

def show_task_stats(aw_instructions):
    """显示任务状态统计信息"""
    total_tasks = len(aw_instructions)
    annotated_tasks = len([item for item in aw_instructions if "random_actions" in item])
    logger.info(f"Total task: {total_tasks} --- Annotated task: {annotated_tasks}")

    failed_tasks = len([item for item in aw_instructions if "task_fail" in item])
    logger.info(f"Total task: {total_tasks} --- Failed task: {failed_tasks}")

if __name__ == "__main__":
    ####### The complete version of the list of examples #######
    args = get_config()
    setup_environment()
    test(args)
