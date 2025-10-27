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

import time
from ...config.prompt import SYS_RECOVERED_TASK_PREDICTOR
from ...utils.utils import load_image_as_ndarray, parse_json

def task_error_correction(traj_info, goal, agent):
    print("任务执行失败...")
    print("正在尝试修正任务...")
    sys_prompt = SYS_RECOVERED_TASK_PREDICTOR
    user_prompt = f"Given the action summary {traj_info[-1]['summary_problem']} and original goal {goal}, what would be a followup task?" 
    prompt = sys_prompt + user_prompt
    
    img = load_image_as_ndarray(traj_info[-1]['screen_after'])
    while True:
        try:
            # llm_output = call_gpt(sys_prompt, user_prompt, img)
            response = agent.predict_mm(prompt, [img])
            task_info = parse_json(response[0],fields=["task"])
            break
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            time.sleep(10)
            continue
    return task_info['task']