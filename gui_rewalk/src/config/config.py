# Copyright 2024 The android_world Authors.
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

# References:
# - android_world/android_world/env/json_action.py action
# - android_world/android_world/env/adb_utils.py
# - android_world/android_world/env/setup_device/setup.py name
# - action type: android_world/android_world/env/actuation.py
# {
#     "task_id": "0",
#     "app_name": "com.taobao.taobao"
# },
# {
#     "task_id": "0",
#     "app_name": "com.ss.android.ugc.aweme"
# },

"""
配置文件，包含环境设置、命令行参数定义等。
"""

import os
from absl import flags
from typing import Dict

# 全局变量，存储flags的引用
_OPENAI_API_KEY = None
_ADB_PATH = None
_EMULATOR_SETUP = None
_DEVICE_CONSOLE_PORT = None
_TASK = None
_RANDOM_WALK_STEPS = None

# 屏幕截图保存目录
SCREEN_GPT_DIR = "./screenshots_gpt_v2"
# 指令文件路径
INSTRUCTION_PATH = "./os_genesis/example.json"

OS_TYPE = "ubuntu"
ACTION_TYPES = ["CLICK", "TYPE", "SCROLL"]  # 可用的动作类型
CLICK_ACTION_TYPES = ["CLICK", "LEFT_DOUBLE", "RIGHT_SINGLE", "DRAG"]
TYPE_ACTION_TYPES = ["TYPE", "FINISHED"]
HOTKEY_ACTION_TYPES = ["HOTKEY"]
SCROLL_ACTION_TYPES = ["SCROLL"]
ACTION_WEIGHTS = [1.0, 0.0, 0.0]       # 对应的权重
RANDOM_ACTION_DELAY = 2.0                       # 随机动作执行后的延迟时间（秒）
RANDOM_SCREENSHOTS_DIR = './screenshots_random' # 随机动作截图保存目录
DEFAULT_MIN_RANDOM_ACTIONS = 1                 # 默认随机动作最小值
DEFAULT_MAX_RANDOM_ACTIONS = 20          # 默认随机动作最大值
DEFAULT_MAX_GUIDED_ACTIONS = 0          # 默认引导动作最大值
DEFAULT_RANDOM_WALK_CROSS_APP = 1              # 默认随机游走跨app数
DEFAULT_MAX_GUIDED_ACTIONS_AFTER_OPENAPP=5          # 默认引导动作最大值（打开app后）
EVALUATE_TRAJECTORY = False

EXC_INIT_APP = [
    "Chrome", 
    "LibreOffice writer", 
    "LibreOffice calc", 
    "GNU image", 
    "setting",
    "calendar",
    "calculator",
    "terminal",
    "mines",
]

LOGIN_WEBSITE = [
    "www.baidu.com",
    "www.bing.com",
    "www.microsoft.com",
    "www.csdn.net",
    "www.anyknew.com",
    "github.com",
    "www.google.com",
    ""
]

TOTAL_ACTION_TYPES = [
    "CLICK", 
    "LEFT_DOUBLE", 
    "RIGHT_SINGLE", 
    "TYPE"
]

UBUNTU_APP_NAMES = [
    "Chrome", 
    "Thuderbird mail", 
    "Vscode", 
    "VLC media player", 
    "LibreOffice writer", 
    "LibreOffice calc", 
    "LibreOffice impress", 
    "GNU image", 
    "File", 
    "Ubuntu Software", 
    "Help", 
    "Trash",
    "Other"
]


def setup_environment():
    """设置环境变量"""
    ####### The complete version of the list of examples #######
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["GRPC_VERBOSITY"] = "ERROR"  # 只显示错误
    os.environ["GRPC_TRACE"] = "none"  # 禁用追踪




def _find_adb_directory() -> str:
    """
    查找ADB可执行文件的路径

    返回:
        str: ADB可执行文件的路径

    异常:
        EnvironmentError: 如果找不到ADB
    """
    potential_paths = [
        os.path.expanduser("~/Library/Android/sdk/platform-tools/adb"),
        os.path.expanduser("~/Android/Sdk/platform-tools/adb"),
    ]
    for path in potential_paths:
        if os.path.isfile(path):
            return path
    raise EnvironmentError(
        "adb not found in the common Android SDK paths. Please install Android"
        " SDK and ensure adb is in one of the expected directories. If it's"
        " already installed, point to the installed location."
    )


def define_flags():
    """定义命令行参数"""
    global _ADB_PATH, _EMULATOR_SETUP, _DEVICE_CONSOLE_PORT, _TASK,_RANDOM_WALK_STEPS

    _ADB_PATH = flags.DEFINE_string(
        "adb_path",
        _find_adb_directory(),
        "Path to adb. Set if not installed through SDK.",
    )

    _EMULATOR_SETUP = flags.DEFINE_boolean(
        "perform_emulator_setup",
        False,
        "Whether to perform emulator setup. This must be done once and only once"
        " before running Android World. After an emulator is setup, this flag"
        " should always be False.",
    )

    _DEVICE_CONSOLE_PORT = flags.DEFINE_integer(
        "console_port",
        5554,
        "The console port of the running Android device. This can usually be"
        " retrieved by looking at the output of `adb devices`. In general, the"
        " first connected device is port 5554, the second is 5556, and"
        " so on.",
    )

    _TASK = flags.DEFINE_string(
        "task",
        None,
        "A specific task to run.",
    )


    # 新增flag
    _RANDOM_WALK_STEPS = flags.DEFINE_integer(
        "random_walk_steps", 100, "Number of steps for random walk exploration."
    )




def get_flags():
    """
    获取所有命令行参数的值

    返回:
        dict: 包含所有命令行参数值的字典
    """
    return {
        "adb_path": _ADB_PATH.value,
        "emulator_setup": _EMULATOR_SETUP.value,
        "console_port": _DEVICE_CONSOLE_PORT.value,
        "task": _TASK.value,
        "RANDOM_WALK_STEPS":_RANDOM_WALK_STEPS.value,
    }
