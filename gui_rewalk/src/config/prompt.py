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


############## exc init prompt ##############

EXC_INIT_DEFAULT_PROMPT = "Open the {exc_init_prompt}, and maximize the window."


############## random prompt ##############
RANDOM_ACTION_PROMPT = """
You will act as an agent that follows my instructions and performs desktop computer tasks as required. You have extensive knowledge of the {os_type} system. You can only execute the actions in the {init_app_name}.

First, you need to sequentially identify the content within the {ocr_bbox_nums} OCR detection boxes I have selected and output that content in the final <description> field.

Second, you need to determine whether each OCR detection box content corresponds to an interactive GUI element based on the output content of each OCR detection box.

Third, if it is an interactive element, select the appropriate Action from the following list: {action_types}. If it is a non-interactive element, set the Action to "dropped".  If you choose action "type", you must provide the text to be typed in the <text> field.

Finally, select the application tag that matches the current OCR detection box from the following applications. Tags: {app_names}.

Please explain your thinking and reasoning process in the <thoughts> section.

## Task proposal rules:
- The returned OCR ID must be a number and must not exceed the number of detection boxes in the image.
- The returned OCR ID is adjacent to the selected detection box.
- DO NOT choose the detection box that is for suspending or powering off the computer.
- You must carefully select the actions; it is not allowed for all of them to be marked as "dropped".


## NOTE:
- Please strictly follow the following JSON output format and provide a complete output.

## Respond in the following JSON format:
```json
{{
"<id1>": {{\"description\": \"xxx\", \"action\": \"xxx\", \"tag\": \"xxx\", \"text\": \"xxx\", \"thoughts\": \"xxx\"}},
"<id2>": {{\"description\": \"xxx\", \"action\": \"xxx\", \"tag\": \"xxx\", \"text\": \"xxx\", \"thoughts\": \"xxx\"}},
...
}}
```\n
"""



############## reverse prompt ##############
REVERSE_INFERENCE_TEMPLATE = """
You are an expert at envisioning specific tasks corresponding to changes in {os_type} screenshots. Current Action: {current_action}
I will provide you with the following:

1. The currently executed action. The type of action currently being executed, which can be one of the following types: {action_types}.

2. The original screenshot before execution and the original screenshot after execution. You need to explain the overall changes in the interface between the two screenshots.

3. [OPTIONAL]The enlarged partial screenshot with red OCR detection boxes before execution. You need to focus on the area of the red OCR detection box, as this is the area where the actual interaction occurred. You need to explain what the content in this red OCR detection box is, think about what interaction will happen when executing the current action, and assist in constructing the Sub-Instruction later.

## Your task is to envision a specific task based on the current action and the corresponding changes in the screenshots. The output should include five parts:

1. Sub-Instruction: Focus on the enlarged partial screenshot with the OCR detection box mentioned in point 3 above, combine it with the screenshots before and after execution, and generate a corresponding natural language instruction for the current action. The instruction should be concise, clear, and executable. It must include specific details critical to the operation, such as file names, times, or other content as they appear in the screenshots. For example: "Click the chat interface, allowing the user to view and participate in conversation", "Type the username 'Agent', preparing for the next step in logging into the account".

2. Analysis: Analyze the changes in the screenshots before and after execution, ignoring system changes such as time and date in the title bar、the red box frame and focus on the inner area of the enlarged OCR detection box.

3. Purpose: Based on the historical and existing screenshots and operations, infer what I am doing and what kind of task I hope to complete.

4. High-Level-Instruction: Based on the analysis results, envision a reasonable and effective high-level task from the historical interface to the current interface. There are two types of high-level instructions: Task-Oriented: Complete a series of operations to achieve a specific goal. Question-Oriented: Perform a series of operations and derive an answer to a specific question.

5. Status: Pay attention to the overall changes in the screenshots before and after execution. If you find that there are no changes between the screenshot before and the screenshot after, ignoring system changes such as time and date in the title bar、the red box frame, output "Dropped"; if you focus on the inner area of the enlarged OCR detection box and find that the changes between the screenshots before and after have no logical connection with the enlarged OCR detection box, output "Dropped"; if you find that the enlarged OCR detection box frames a non-interactive GUI element, output "Dropped"; if none of the above three situations are met, output "Kept".

## Respond in the following JSON format:
```json
{{
  "Sub-Instruction": "xxx",
  "Analysis": "xxx",
  "Purpose": "xxx",
  "High-Level-Instruction": "xxx",
  "Status": "xxx"
}}
```\n
"""


REVERSE_INFERENCE_TEMPLATE_NO_USE = """
You are an expert at envisioning specific tasks corresponding to changes in {os_type} screenshots. Current Action: {current_action}
I will provide you with the following:

1. The currently executed action. The type of action currently being executed, which can be one of the following types: {action_types}.

2. The original screenshot before execution and the original screenshot after execution. You need to explain the overall changes in the interface between the two screenshots.

3. The screenshot with red OCR detection boxes before execution and the enlarged partial screenshot with the OCR detection box. You need to focus on the area of the red OCR detection box, as this is the area where the actual interaction occurred. You need to explain what the content in this red OCR detection box is, think about what interaction will happen when executing the current action, and assist in constructing the Sub-Instruction later.

## Your task is to envision a specific task based on the current action and the corresponding changes in the screenshots. The output should include five parts:

1. Sub-Instruction: Focus on the enlarged partial screenshot with the OCR detection box mentioned in point 3 above, combine it with the screenshots before and after execution, and generate a corresponding natural language instruction for the current action. The instruction should be concise, clear, and executable. It must include specific details critical to the operation, such as file names, times, or other content as they appear in the screenshots. For example: "Click the chat interface, allowing the user to view and participate in conversation", "Type the username 'Agent', preparing for the next step in logging into the account".

2. Analysis: Analyze the changes in the screenshots before and after execution, ignoring system changes such as time and date in the title bar and focus on the area of the enlarged OCR detection box.

3. Purpose: Based on the historical and existing screenshots and operations, infer what I am doing and what kind of task I hope to complete.

4. High-Level-Instruction: Based on the analysis results, envision a reasonable and effective high-level task from the historical interface to the current interface. There are two types of high-level instructions: Task-Oriented: Complete a series of operations to achieve a specific goal. Question-Oriented: Perform a series of operations and derive an answer to a specific question.

5. Status: Pay attention to the overall changes in the screenshots before and after execution. If you find that there are no changes between the screenshot before and the screenshot after, ignoring system changes such as time and date in the title bar, output "Dropped"; if you focus on the area of the enlarged OCR detection box and find that the changes between the screenshots before and after have no logical connection with the enlarged OCR detection box, output "Dropped"; if you find that the enlarged OCR detection box frames a non-interactive GUI element, output "Dropped"; if none of the above three situations are met, output "Kept".

## Respond in the following JSON format:
```json
{{
  "Sub-Instruction": "xxx",
  "Analysis": "xxx",
  "Purpose": "xxx",
  "High-Level-Instruction": "xxx",
  "Status": "xxx"
}}
```\n
"""

############## task prompt ##############

SYS_TASK_FOLLOWUP_PERSONA_ON_UBUNTU = """
You are an intelligent assistant observing a user who has just completed a task on {os_type} device. Based on this previous task and its context, infer the most likely next task the user would perform. Your goal is to propose a plausible, purposeful, and clearly defined follow-up task that logically continues from the completed one.

### Task Generation Requirements:

1. **Logical Continuation**  
   - The next task must logically build upon the previous one. It should extend or deepen the prior behavior based on user interest, app state, or content.
   - Do not repeat, paraphrase, or contradict the previous task.

2. **Goal-Oriented and Specific**  
   - The task must have a clear purpose and a well-defined end state.  
   - Avoid vague descriptions such as “browse more”, “explore related content”, or “look around”.  
   - Use concrete references (e.g., video titles, place names, keywords, objects, timestamps).

3. **Result-Completeness and Closure**  
   - The task must include the **final user interaction needed to achieve the goal**, not just the initiation of a process.  
   - Do **not** stop at intermediate steps like opening an app or search results.  
   - Always include the next logical interaction — such as watching a specific video, opening a particular article, or confirming a key detail — that completes the task.

4. **Completable Within 3 Atomic Actions**  
   - The task should be feasible with no more than 3 user interactions (e.g., tap, type, select).
   - Tasks that require login, account switching, or permission setting are **not allowed**.

5. **Realistic and Executable**  
   - The task must reflect real usage patterns and be executable in a typical mobile environment.
   - Avoid speculative, unsupported, or abstract behaviors.
    - Avoid tasks that require restarting, shutting down, or updating the system.

6. **Content-Aware**  
   - Leverage the context of the prior task: topic, keywords, apps used, content viewed, and user intent.

7. **No Communication Tasks**  
   - Do not include actions involving messaging, emailing, posting to social media, or sharing content.

### Output Instructions:

Respond in the following JSON format:
```json
{{
  "thoughts": "<Detailed reasoning: Why this next task logically follows? How it continues user intent? Why it reaches a meaningful goal within constraints?>",
  "task": "<Concrete, result-driven, executable next task with a clear end state>",
  "action": "<The first UI action the user would take to begin this task>",
  "app": "<The Android app used to perform this task>"
}}
"""

SYS_CROSS_APP_NEXT_TASK_PREDICTOR_ON_UBUNTU = """
You are an intelligent assistant observing a user who just completed a task on {os_type} device. The user is now about to switch apps to perform the next most likely task. Your goal is to propose a plausible, goal-oriented, and clearly defined next task that logically follows from the previous one — but must be completed in a different app, chosen from the list below:

{app_names}

### Task Generation Requirements:

1. **Cross-App Transition**
   - The task must take place in a different app from the one just used.
   - The new app must be selected from the provided list.
   - Do not continue in or return to the current app.

2. **Logical Continuation**
   - The task must logically extend the user's prior goal, intent, or content.
   - Use topic, keywords, content type, or interest signals from the prior task to justify the transition.

3. **Result-Completeness and Closure**
   - The task must reach a clearly **observable outcome** (e.g., opening and watching a specific video, reading an article, confirming a location).
   - Do **not** stop at intermediate actions like opening the app, reaching a search page, or listing results.
   - Always include the follow-up interaction that completes the intended action.

4. **Clarity and Specificity**
   - Avoid vague terms like “explore”, “browse”, “check out more”.
   - Use real or plausible entities: keywords, names, places, or identifiers.

5. **Minimal Interaction Constraint**
   - The entire task must be achievable within 2 atomic actions (e.g., tap + type, tap + select).

6. **Feasibility**
   - Do not propose tasks requiring login, sharing, permission granting, or complex navigation.
   - The task must be executable in a standard Android environment.
   - Avoid tasks that require restarting, shutting down, or updating the system.

### Output Instructions:

Respond in the following JSON format:
```json
{{
  "thoughts": "<Explain why this app is chosen and why the task is a logical continuation of the previous one. Justify that it is feasible, relevant, and result-complete.>",
  "task": "<Specific, result-oriented next task completed in a different app>",
  "action": "<First action the user would take to begin this task>",
  "app": "<The app name chosen from the list where the task will be completed>"
}}
"""



SYS_RECOVERED_TASK_PREDICTOR = """
You are an intelligent assistant helping to recover from a failed or stuck mobile automation task.

You will be given:
- The user's **original goal**
- A **summary** of attempted actions and why they failed
- The **current screen description** (visible app and UI state)

Your job:
Reformulate the task so it is **actually achievable**, while preserving the user's **core intent** and maintaining logical continuity between tasks.

---

### Recovery Decision Process

1. **Feasibility Assessment**
   - Based on the `summary` and `current screen`, determine if the original goal is realistically achievable in the current environment.
   - Criteria for "Not Achievable": 
     - The target object/content does not exist or cannot be found
     - The app lacks the required function or permission
     - The path has been fully tried with no results

2. **If Achievable → Path Adjustment Mode**
   - Keep the same overall intent but **change the execution path** (use different UI elements, menus, search terms, or filters).
   - Explicitly avoid any UI element, keyword, or path already used in failed attempts.

3. **If NOT Achievable → Intent Reconstruction Mode**
   - Keep the **main topic keywords** (e.g., subject name, file name, product title).
   - Change the environment, app, or method to achieve a **related but feasible outcome**.
   - Examples:
     - If searching for a file failed → switch to opening a website or app to download it
     - If opening a folder failed → use an alternative source for similar content
   - The new goal can differ significantly from the original in method, but must stay relevant to the original intent.

4. **Goal Requirements**
   - Must have a concrete end state achievable within 3 atomic actions.
   - Avoid vague “explore more” or “browse around” type tasks.
   - No login, messaging, posting, or speculative actions without visible context.

5. **Reasoning Requirements**
   - In `thoughts`, explicitly state:
     - Feasibility judgment (Achievable / Not Achievable)
     - Failure reason from summary
     - Which mode was chosen (Path Adjustment / Intent Reconstruction)
     - How the new goal differs in execution but keeps logical continuity

---

- If the attempted actions repeatedly fail due to the target object being non-existent or non-interactive, 
  do NOT rephrase or retry the same goal.  
- Instead, switch to a new but logically related goal by:
  1. Retaining the core topic keywords .
  2. Redirecting the user to an alternative but feasible outcome .  
- This ensures the task moves forward instead of being trapped in repeated reformulations.

### Output Format
Respond in the following JSON format:
```json
{
  "thoughts": "<Feasibility check, mode chosen, banned paths, reasoning for changes, and why success is more likely>",
  "task": "<Revised, achievable, goal-driven task with a clear end state>",
  "action": "<First UI action to begin this task>",
  "app": "<The Android app to perform this task>"
}
"""

############## summary prompt ##############



SYS_SUBTASK_SUMMARY = """You are given a short sequence of screenshots representing a coherent segment of user actions on {os_type} device. For each step, you are also given a high-level instruction that reflects the likely intent at that moment.

Your goal is to summarize what specific, complete subtask was accomplished by this group of actions. Do not simply repeat or stitch together the individual high-level instructions. Instead, infer a single, well-defined objective the user was trying to accomplish within this local context.

### Requirements:

1. **Completeness**  
   - The subtask description must reflect the full span of the input sequence, including its final step.
   - Do not truncate the summary prematurely.

2. **Goal-Directedness**  
   - The subtask must express a clear local objective that the user successfully completed.
   - Avoid describing open-ended or ambiguous behaviors such as passive browsing or exploration.

3. **Clarity of Description**  
   - The description must use precise and specific language.
   - Avoid vague terms like “browse”, “explore”, or “check out something”.
   - If the user performs a search or opens specific content, clearly state what was searched or accessed.

4. **Logical Coherence**  
   - The subtask must have a logical internal flow; ensure that actions in the sequence lead naturally toward the described goal.

### Output Style:

- The subtask must be described in a **formal, instructional tone**, suitable for documentation or agent planning.
- Avoid hypothetical or speculative phrases (e.g., “might have”, “possibly”, “if needed”).
- Only describe actually completed and meaningful actions in the subtask.

### Output Format:
```json
{{
  "thoughts": "<Detailed reasoning and interpretation of this subtask segment>",
  "task": "<Subtask description in formal instructional tone>"
}}
"""

SYS_TASK_SUMMARY = """
You are given a complete sequence of user actions performed on {os_type} device. For each step, you have access to:
- The corresponding screenshot,
- An inferred high-level instruction (describing the likely intent of the user at that step),
- A summarized subtask description derived from groups of related actions.

Your goal is to summarize the entire user session as a **single, complete, and clearly defined task** that was accomplished by performing these actions in sequence. This task should reflect the actual goal the user achieved — not just transient interactions, UI distractions, or speculative behavior.

### Summary Requirements:

1. **Task-Oriented Abstraction**
   - Focus on summarizing the **goal-directed behavior** completed across the session.
   - Do **not** include irrelevant, passive, or system-generated steps (e.g., default text suggestions, placeholder content, momentary misclicks).
   - Only describe actions that clearly contributed to the user's intent.

2. **Completeness**
   - Cover the full behavioral trace, including the final meaningful step.
   - Avoid premature truncation or skipping the ending goal.

3. **Relevance Filtering**
   - Exclude intermediate or background steps that do not meaningfully advance the user's task (e.g., UI defaults, empty search suggestions).
   - Ignore content not clearly chosen or interacted with by the user.

4. **Clarity and Specificity**
   - Use precise language to describe what was done and why.
   - For search, clearly state the keyword or target topic.
   - Avoid vague or generic phrases such as “browse content”, “explore topics”, or “view related info”.

5. **Logical Coherence**
   - Ensure the steps form a **cohesive and purposeful progression**, not a fragmented list.
   - If multiple apps are used, explain how they connect toward the same goal.

### Output Style:

- Write the task in a **formal, instructional tone**, as if specifying a goal in a product spec or user intent model.
- Avoid uncertain or hypothetical phrasing (e.g., "might have", "possibly", "if needed").
- The final output should be **specific, executable, and self-contained**.

### Output Format:
```json
{{
  "thoughts": "<Detailed reasoning and interpretation of the user's session, focusing on core goal, meaningful steps, and logical structure. Discard irrelevant or passive actions.>",
  "task": "<Final task description, formal and precise, covering only essential, purposeful actions>"
}}
"""

############## task check ##############

PROMPT_PREFIX = """
You are an agent who can operate {os_type} on behalf of a user.
You will be given current screenshot、goal and action history.
"""
