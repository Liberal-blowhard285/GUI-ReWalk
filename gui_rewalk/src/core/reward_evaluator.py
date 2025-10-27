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

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import numpy as np


class TrajectoryRewardEvaluator:
    """使用GPT4评估轨迹有效性的奖励评估器"""
    
    def __init__(self, agent):
        """
        初始化评估器
        
        参数:
            agent: M3A代理对象，用于访问GPT4
        """
        self.agent = agent
        self.evaluation_prompt = self._get_evaluation_prompt()
    
    def _get_evaluation_prompt(self) -> str:
        """获取轨迹评估的prompt"""
        return """Trajectory Reward Model Prompt

You are an expert in evaluating GUI agent task trajectories. Your task is to assess the quality and effectiveness of task trajectories for GUI manipulation tasks.

A trajectory consists of the following components:
1. High-level Instruction: Describes the user's intended task (e.g., "Create a new blank project name 'OS-Genesis'").
2. Action History: Includes two key parts:
   - Reasoning and Action for Each Step: A sequence of actions performed by the agent, including the reasoning thought and final executed action.
   - GUI Screenshots: Screenshots of the last state: (if there are at least three states; otherwise, include all states).

When evaluating a trajectory, consider these key aspects:

Evaluation Criteria:
1. Trajectory Coherence:
   - Do the low-level steps and corresponding actions follow a logical sequence toward the goal?
   - Are the actions clearly described and specific?
   - Are there redundant or unnecessary actions?

2. Task Completion:
   - Does the trajectory successfully achieve the instructed task?
   - Are all necessary interactions completed?
   - Are error cases handled appropriately?

Scoring Guidelines:
Rate the trajectory on a scale of 1 to 5 based on the evaluation criteria:
- 5: The task is perfectly completed, successfully executing multiple actions to achieve the goal. The sequence is logically clear with no noticeable redundancies.
- 4: The task is mostly completed, successfully executing multiple actions. However, due to challenges or ambiguities in the instructions, the completion is not perfect, or there are inefficiencies in the process.
- 3: The task is partially completed, with some successful actions executed. However, due to task or environmental constraints, the goal is not fully achieved, or the sequence ends in a loop or error.
- 2: Only a few actions are executed. Although there is an attempt to complete the task, the trajectory deviates from the goal early on or demonstrates significant inefficiencies in execution and logic.
- 1: The task fails completely, with no meaningful actions executed at the start. The sequence either falls into an immediate deadlock, a repetitive loop, or demonstrates no value in completing the task. Or the tasks are completely inaccessible.

Note: If the task is relatively complex, but the trajectory demonstrates valuable attempts, even if the task is not fully completed, consider adjusting the score upward. However, if the task is complex but the trajectory fails to perform actions that contribute meaningfully to task completion, no extra points should be awarded.

You need to judge the score based on the agent's actions and screenshots combined.

Response Format:
Format your response into two lines as shown below:
Reason: <your thoughts and reasoning process for the score>
Score: <your score from 1-5>"""

    def _format_trajectory_for_evaluation(
        self, 
        trajectory: List[Dict[str, Any]], 
        instruction: str = "Random exploration task"
    ) -> str:
        """格式化轨迹数据用于评估"""
        
        formatted_text = f"High-level Instruction: {instruction}\n\n"
        formatted_text += "Action History:\n"
        
        for i, step in enumerate(trajectory):
            step_num = step.get('step', i + 1)
            
            # 兼容不同的轨迹数据结构
            if 'action_reason' in step and 'summary' in step:
                # 来自task_runner的轨迹结构
                action_reason = step.get('action_reason', 'No reasoning provided')
                action_data = step.get('action_json', {})
                action_json = json.loads(action_data)
                summary = step.get('summary', 'No summary provided')
                
                formatted_text += f"Step {step_num}:\n"
                formatted_text += f"  Reasoning: {action_reason}\n"
                formatted_text += f"  Action: {action_data}\n"
                formatted_text += f"  Summary: {summary}\n\n"
                
            else:
                # 来自random_walker的轨迹结构
                action_data = step.get('action_json', {})
                action_json = json.loads(action_data)
                action_type = action_json.get('action_type', 'UNKNOWN')
                
                formatted_text += f"Step {step_num}:\n"
                formatted_text += f"  Reasoning: Performing {action_type} action on the current interface\n"
                formatted_text += f"  Action: {action_data}\n\n"
        
        return formatted_text

    def _get_trajectory_screenshots(self, trajectory: List[Dict[str, Any]]) -> List[str]:
        """获取轨迹的关键截图"""
        screenshots = []
        
        # 如果轨迹少于3步，包含所有截图
        if len(trajectory) <= 3:
            for step in trajectory:
                # 兼容不同的截图字段名
                if 'screen_before' in step:
                    screenshots.append(step['screen_before'])
                if 'screen_before_som' in step:
                    screenshots.append(step['screen_before_som'])
                if 'screen_after' in step:
                    screenshots.append(step['screen_after'])
        else:
            # 如果轨迹超过3步，只包含最后几个状态
            for step in trajectory[-3:]:
                # 兼容不同的截图字段名
                if 'screen_before' in step:
                    screenshots.append(step['screen_before'])
                if 'screen_before_som' in step:
                    screenshots.append(step['screen_before_som'])
                if 'screen_after' in step:
                    screenshots.append(step['screen_after'])
        
        # 去重并保持顺序
        unique_screenshots = []
        seen = set()
        for screenshot in screenshots:
            if screenshot and screenshot not in seen:
                unique_screenshots.append(screenshot)
                seen.add(screenshot)
        
        return unique_screenshots

    def evaluate_trajectory(
        self, 
        trajectory: List[Dict[str, Any]], 
        instruction: str = "Random exploration task"
    ) -> Tuple[int, str]:
        """
        评估轨迹的有效性
        
        参数:
            trajectory: 轨迹数据列表
            instruction: 高级指令描述
            
        返回:
            tuple: (评分(1-5), 评估理由)
        """
        if not trajectory:
            return 1, "Empty trajectory provided"
        
        try:
            # 格式化轨迹文本
            trajectory_text = self._format_trajectory_for_evaluation(trajectory, instruction)
            
            # 获取关键截图
            screenshot_paths = self._get_trajectory_screenshots(trajectory)
            
            # 构建完整的评估prompt
            full_prompt = f"{self.evaluation_prompt}\n\n"
            full_prompt += f"Please evaluate the following trajectory:\n\n{trajectory_text}"
            
            # 编码截图
            screenshot_images = []
            for screenshot_path in screenshot_paths:
                if os.path.exists(screenshot_path):
                    img = np.array(Image.open(screenshot_path))
                    screenshot_images.append(img)
            
            # 使用GPT4进行评估
            if hasattr(self.agent, "predict_mm") and self.agent:
                if screenshot_images:
                    # 使用多模态预测
                    response_text, prompt_tokens, completion_tokens, counter  = self.agent.predict_mm(full_prompt, screenshot_images)
                else:
                    # 仅使用文本预测
                    response_text, prompt_tokens, completion_tokens, counter  = self.agent.predict_mm(full_prompt, [])
                
                # 解析响应
                score, reason = self._parse_evaluation_response(response_text)
                return score, reason, prompt_tokens, completion_tokens, counter
            else:
                return 3, "GPT4 model not available, returning default score"
                
        except Exception as e:
            print(f"Error evaluating trajectory: {e}")
            return 2, f"Evaluation failed due to error: {str(e)}"

    def _parse_evaluation_response(self, response: str) -> Tuple[int, str]:
        """解析GPT4的评估响应"""
        try:
            lines = response.strip().split('\n')
            reason = ""
            score = 3  # 默认分数
            
            for line in lines:
                line = line.strip()
                if line.startswith("Reason:"):
                    reason = line[7:].strip()
                elif line.startswith("Score:"):
                    score_text = line[6:].strip()
                    # 提取数字
                    import re
                    score_match = re.search(r'(\d+)', score_text)
                    if score_match:
                        score = int(score_match.group(1))
                        score = max(1, min(5, score))  # 确保分数在1-5范围内
            
            if not reason:
                reason = response.strip()
            
            return score, reason
            
        except Exception as e:
            print(f"Error parsing evaluation response: {e}")
            return 3, f"Failed to parse response: {response[:200]}..."

    def evaluate_trajectory_batch(
        self, 
        trajectories: List[List[Dict[str, Any]]], 
        instructions: Optional[List[str]] = None
    ) -> List[Tuple[int, str]]:
        """
        批量评估多个轨迹
        
        参数:
            trajectories: 轨迹数据列表的列表
            instructions: 对应的指令列表（可选）
            
        返回:
            List[Tuple[int, str]]: 评分和理由的列表
        """
        results = []
        
        if instructions is None:
            instructions = ["Random exploration task"] * len(trajectories)
        
        for i, trajectory in enumerate(trajectories):
            instruction = instructions[i] if i < len(instructions) else "Random exploration task"
            score, reason = self.evaluate_trajectory(trajectory, instruction)
            results.append((score, reason))
            print(f"Trajectory {i+1}: Score={score}, Reason={reason[:100]}...")
        
        return results


def evaluate_random_walker_trajectory(
    agent,
    trajectory: List[Dict[str, Any]],
    instruction: str = "Random exploration and interaction with the mobile application interface"
) -> Tuple[int, str]:
    """
    便捷函数：评估随机游走轨迹
    
    参数:
        agent: M3A代理对象
        trajectory: 随机游走生成的轨迹数据
        instruction: 任务描述
        
    返回:
        tuple: (评分(1-5), 评估理由)
    """
    evaluator = TrajectoryRewardEvaluator(agent)
    return evaluator.evaluate_trajectory(trajectory, instruction) 