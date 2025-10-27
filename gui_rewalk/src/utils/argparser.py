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

import argparse

def get_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GUI ReWalk to product GUI instructions"
    )

    # environment config
    parser.add_argument("--task_num", type=int, default=1)
    parser.add_argument("--path_to_vm", type=str, default=None)
    parser.add_argument(
        "--headless", action="store_true", help="Run in headless machine"
    )
    parser.add_argument(
        "--action_space", type=str, default="pyautogui", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
        default="screenshot",
        help="Observation type",
    )
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=15)
    
    # emulator config
    parser.add_argument(
        "--vm_provider", choices=["custom", "vmware"], default="vmware", help="VM Provider"
    )
    parser.add_argument("--custom_vm_ip", type=str, default="127.0.0.1")
    parser.add_argument("--custom_vm_port", type=int, default=5000)
    parser.add_argument("--use_ark", type=bool, default=False)
    parser.add_argument("--parallel", type=int, default=1)

    # agent config
    parser.add_argument("--max_trajectory_length", type=int, default=0)
    parser.add_argument(
        "--test_config_base_dir", type=str, default="evaluation_examples"
    )

    # lm config
    parser.add_argument("--exc_init_action", type=str, default=None, choices=[None, "default", "chrome", "writer"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model_version", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=1500)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument("--ocr_model_path", type=str, default=None)
    parser.add_argument("--random_walker", type=bool, default=False)
    parser.add_argument("--enable_thinking", type=bool, default=False)

    # example config
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument(
        "--test_all_meta_path", type=str, default="evaluation_examples/test_all.json"
    )

    # random walker config
    parser.add_argument("--exce_task_completion", type=bool, default=False)
    parser.add_argument("--reverse_inference", type=bool, default=False)
    parser.add_argument("--summary_inference", type=bool, default=False)
    parser.add_argument("--max_random_actions", type=int, default=3)
    parser.add_argument("--max_guided_actions", type=int, default=10)
    parser.add_argument("--random_walk_cross_app", type=int, default=1)
    parser.add_argument("--max_guided_actions_after_openapp", type=int, default=5)

    # logging related
    parser.add_argument("--result_dir", type=str, default="./results")
    parser.add_argument("--pq_format", type=str, default="uitars")
    parser.add_argument("--score_threshold", type=int, default=3)
    parser.add_argument("--save_max_steps", type=int, default=25)

    args = parser.parse_args()

    return args
