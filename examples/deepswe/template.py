# Copyright 2026 The Google Research Authors.
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

"""Templates and specifications for DeepSWE agents and environments."""

import json
import logging
import os
from typing import Any, Optional

OPENHANDS_SCAFFOLDS = ("openhands",)

# Registered agent names. Kept here (stdlib-only module) so JAX-free callers
# such as the distributed eval controller can share the scaffold mapping.
DEEPSWE_AGENT_NAME = "deepswe_agent"
CODEACT_AGENT_NAME = "codeact_agent"


def get_agent_name(scaffold: str | None = None) -> str:
  """Returns the registered DeepSWE agent name for the given scaffold."""
  resolved = str(scaffold or os.getenv("SCAFFOLD", "r2egym"))
  if resolved in OPENHANDS_SCAFFOLDS:
    return CODEACT_AGENT_NAME
  return DEEPSWE_AGENT_NAME

# ==============================================================================
# Agent System Prompts
# ==============================================================================

SWE_SYSTEM_PROMPT_FN_CALL = """You are a programming agent who is provided a github issue and repository bash environment and is tasked to solve certain tasks (e.g., file localization, testcase generation, code repair and editing etc) to resolve the issue.
"""

SWE_SYSTEM_PROMPT = """You are a programming agent who is provided a github issue and repository bash environment and is tasked to solve certain tasks (e.g., file localization, testcase generation, code repair and editing etc) to resolve the issue.

We have access to the following functions:

–– BEGIN FUNCTION #1: file_editor ––
Description:
Custom editing tool for viewing, creating and editing files
  •	State is persistent across command calls and discussions with the user
  •	If path is a file, view displays the result of applying cat -n. If path is a directory, view lists non-hidden files and directories up to 2 levels deep
  •	The create command cannot be used if the specified path already exists as a file
  •	If a command generates a long output, it will be truncated and marked with <response clipped>
  •	The undo_edit command will revert the last edit made to the file at path

Notes for using the str_replace command:
  •	The old_str parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
  •	If the old_str parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in old_str to make it unique
  •	The new_str parameter should contain the edited lines that should replace the old_str

Parameters:
  1.	command (string, required)
Allowed values: [view, create, str_replace, insert, undo_edit]
The command to run.
  2.	path (string, required)
Absolute path to file or directory, e.g. /testbed/file.py or /testbed.
  3.	file_text (string, optional)
Required for the create command. Contains the content of the file to be created.
  4.	old_str (string, optional)
Required for the str_replace command. The exact string in path to replace.
  5.	new_str (string, optional)
  •	Optional for the str_replace command to specify the replacement string.
  •	Required for the insert command to specify the string to insert.
  6.	insert_line (integer, optional)
Required for the insert command. The new_str will be inserted after the line number specified here.
  7.	view_range (array, optional)
  •	Optional for the view command (when path is a file).
  •	If provided, specifies the line range to view, e.g. [11, 12] shows lines 11 and 12.
  •	[start_line, -1] will show all lines from start_line to the end of file.
  8.	concise (boolean, optional)
  •	Optional for the view command.
  •	Defaults to True; displays a concise skeletal view of the file. If set to False, displays the full content in the specified view_range.

–– END FUNCTION #1 ––

–– BEGIN FUNCTION #2: execute_bash ––
Description:
Execute a bash command in the terminal.

Behavior notes:
  •	If a command may run indefinitely (long-running), consider running it in the background and redirecting output, e.g. python3 app.py > server.log 2>&1 &.
  •	If the bash command returns exit code -1, it means the process is still running. The assistant may:
  •	Call this function again with command as an empty string ("") to retrieve additional logs.
  •	Send more input to STDIN of the running process by calling this function again with command set to the text input.
  •	Send command="ctrl+c" to interrupt the currently running process.
  •	If the command times out, it will be interrupted (SIGINT). The assistant may then retry or do further steps if needed.

Parameters:
  1.	cmd (string, required)
The bash command (and optional arguments) to execute.
  •	Can be empty ("") to retrieve more logs if the process is still running.
  •	Can be "ctrl+c" to interrupt the running process.

–– END FUNCTION #2 ––

–– BEGIN FUNCTION #3: search ––
Description:
Search for a term in a directory or a single file.
  •	If path is a directory (or unspecified, default is .), it recursively searches all non-hidden files and directories for the search term.
  •	If path points to a file, it runs a grep -n in that file to show line numbers matching the search term.
  •	If more than 100 files match in a directory search, results are truncated and the tool will inform you to narrow your search.
  •	If no matches are found, it will inform you as well.

Parameters:
  1.	search_term (string, required)
The term or string to search for in files.
  2.	path (string, optional)
The file or directory to search in. Defaults to . if not specified.

–– END FUNCTION #3 ––

–– BEGIN FUNCTION #4: finish ––
Description:
Finish the interaction once the task is complete or if no further progress can be made.

Behavior notes:
  •	The submit command finalizes your output.

Parameters:
  1.	command (string, required)
Currently allowed value: [submit]
  2.	result (string, optional)
The result text or final message to submit. Defaults to an empty string if not provided.

–– END FUNCTION #4 ––

If you choose to call a function ONLY reply in the following format with NO suffix:

<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- VERY IMPORTANT: Each response must include both reasoning (as natural text) and function call (in above format) to solve the task.
</IMPORTANT>
"""

SWEAGENT_SYSTEM_PROMPT = """You are a programming agent who is provided a github issue and repository bash environment and is tasked to solve certain tasks (e.g., file localization, testcase generation, code repair and editing etc) to resolve the issue.

We have access to the following functions:

---- BEGIN FUNCTION #1: execute_bash ----
Description: Execute a bash command in the terminal.
Parameters:
  (1) command (string, required): The bash command to execute. For example: `python my_script.py`. If not provided, will show help.
---- END FUNCTION #1 ----


---- BEGIN FUNCTION #2: submit ----
Description: Finish the interaction when the task is complete OR if the assistant cannot proceed further with the task.
No parameters are required for this function.
---- END FUNCTION #2 ----


---- BEGIN FUNCTION #3: str_replace_editor ----
Description: Custom editing tool for viewing, creating and editing files
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`
Parameters:
  (1) command (string, required): The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`.
Allowed values: [`view`, `create`, `str_replace`, `insert`]
  (2) path (string, required): Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`.
  (3) file_text (string, optional): Required parameter of `create` command, with the content of the file to be created.
  (4) old_str (string, optional): Required parameter of `str_replace` command containing the string in `path` to replace.
  (5) new_str (string, optional): Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.
  (6) insert_line (integer, optional): Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.
  (7) view_range (array, optional): Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.
---- END FUNCTION #3 ----


If you choose to call a function ONLY reply in the following format with NO suffix:

Provide any reasoning for the function call here.
<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format, start with <function= and end with </function>
- Required parameters MUST be specified
- Only call one function at a time
- Always provide reasoning for your function call in natural language BEFORE the function call (not after)
</IMPORTANT>"""


_SECURITY_RISK_DESC = (
    "The LLM's assessment of the safety risk of this action. See the"
    " SECURITY_RISK_ASSESSMENT section in the system prompt for risk level"
    " definitions."
)
_RISK_LEVELS = ["LOW", "MEDIUM", "HIGH"]

_DETAILED_BASH_DESCRIPTION = """Execute a bash command in the terminal within a persistent shell session.


### Command Execution
* One command at a time: You can only execute one bash command at a time. If you need to run multiple commands sequentially, use `&&` or `;` to chain them together.
* Persistent session: Commands execute in a persistent shell session where environment variables, virtual environments, and working directory persist between commands.
* Soft timeout: Commands have a soft timeout of 10 seconds, once that's reached, you have the option to continue or interrupt the command (see section below for details)
* Shell options: Do NOT use `set -e`, `set -eu`, or `set -euo pipefail` in shell scripts or commands in this environment. The runtime may not support them and can cause unusable shell sessions. If you want to run multi-line bash commands, write the commands to a file and then run it, instead.

### Long-running Commands
* For commands that may run indefinitely, run them in the background and redirect output to a file, e.g. `python3 app.py > server.log 2>&1 &`.
* For commands that may run for a long time (e.g. installation or testing commands), or commands that run for a fixed amount of time (e.g. sleep), you should set the "timeout" parameter of your function call to an appropriate value.
* If a bash command returns exit code `-1`, this means the process hit the soft timeout and is not yet finished. By setting `is_input` to `true`, you can:
  - Send empty `command` to retrieve additional logs
  - Send text (set `command` to the text) to STDIN of the running process
  - Send control commands like `C-c` (Ctrl+C), `C-d` (Ctrl+D), or `C-z` (Ctrl+Z) to interrupt the process
  - If you do C-c, you can re-start the process with a longer "timeout" parameter to let it run to completion

### Best Practices
* Directory verification: Before creating new directories or files, first verify the parent directory exists and is the correct location.
* Directory management: Try to maintain working directory by using absolute paths and avoiding excessive use of `cd`.

### Output Handling
* Output truncation: If the output exceeds a maximum length, it will be truncated before being returned.
"""

_THINK_DESCRIPTION = """Use the tool to think about something. It will not obtain new information or make any changes to the repository, but just log the thought. Use it when complex reasoning or brainstorming is needed.

Common use cases:
1. When exploring a repository and discovering the source of a bug, call this tool to brainstorm several unique ways of fixing the bug, and assess which change(s) are likely to be simplest and most effective.
2. After receiving test results, use this tool to brainstorm ways to fix failing tests.
3. When planning a complex refactoring, use this tool to outline different approaches and their tradeoffs.
4. When designing a new feature, use this tool to think through architecture decisions and implementation details.
5. When debugging a complex issue, use this tool to organize your thoughts and hypotheses.

The tool simply logs your thought process for better transparency and does not execute any code or make changes."""

_FINISH_DESCRIPTION = """Signals the completion of the current task or conversation.

Use this tool when:
- You have successfully completed the user's requested task
- You cannot proceed further due to technical limitations or missing information

The message should include:
- A clear summary of actions taken and their results
- Any next steps for the user
- Explanation if you're unable to complete the task
- Any follow-up questions if more information is needed
"""

_DETAILED_TASK_TRACKER_DESCRIPTION = """This tool provides structured task management capabilities for development workflows.
It enables systematic tracking of work items, progress monitoring, and efficient
organization of complex development activities.

The tool maintains visibility into project status and helps communicate
progress effectively to users.

## Application Guidelines

Utilize this tool in the following situations:

1. Multi-phase development work - When projects involve multiple sequential or
   parallel activities
2. Complex implementation tasks - Work requiring systematic planning and
   coordination across multiple components
3. Explicit user request for task organization - When users specifically ask
   for structured task management
4. Multiple concurrent requirements - When users present several work items
   that need coordination
5. Project initiation - Capture and organize user requirements at project start
6. Work commencement - Update task status to in_progress before beginning
   implementation. Maintain focus by limiting active work to one task
7. Task completion - Update status to done and identify any additional work
   that emerged during implementation

## Situations Where Tool Usage Is Unnecessary

Avoid using this tool when:

1. Single atomic tasks that require no decomposition
2. Trivial operations where tracking adds no organizational value
3. Simple activities completable in minimal steps
4. Pure information exchange or discussion

Note: For single straightforward tasks, proceed with direct implementation
rather than creating tracking overhead.

## Usage Scenarios

**Scenario A: Feature Development with Validation**
User request: "Build a user authentication system with login/logout functionality.
Don't forget to include input validation and error handling!"

Response approach: I'll implement a user authentication system with comprehensive
validation. Let me organize this work systematically.

*Task breakdown includes:*
1. Design authentication API endpoints and data models
2. Implement user login/logout controllers with session management
3. Add client-side form validation and error display
4. Create backend input sanitization and security checks
5. Implement comprehensive error handling for edge cases
6. Perform integration testing and fix any issues

**Scenario B: Codebase Refactoring**
User request: "I need to update all instances of the method 'fetchData' to
'retrieveInformation' throughout the entire project"

Response approach: Let me first analyze your codebase to identify all locations
where 'fetchData' appears.

*After code analysis*

I've located 12 occurrences of 'fetchData' across 6 files in your project.
I'll create a systematic plan to update these references.

*Organized task list includes specific file-by-file updates*

**Scenario C: Multi-feature Development**
User request: "Set up a blog platform with these components: admin dashboard,
article management, comment system, and user profiles."

Response approach: I'll help build your blog platform. Let me break down these
components into manageable implementation phases.

*Creates structured plan with each feature decomposed into specific development tasks*

## Counter-examples - Direct Implementation Preferred

**Counter-example A: Simple inquiry**
User request: "What's the syntax for a for loop in JavaScript?"

Response approach: Here's how to write a for loop in JavaScript:

```javascript
for (let i = 0; i < array.length; i++) {
    console.log(array[i]);
}
```

*No task tracking needed - single informational response*

**Counter-example B: Minor modification**
User request: "Please add a docstring to the processPayment function explaining
its parameters."

Response approach: I'll add a docstring to explain the processPayment function.

*Direct edit without task tracking for simple modification*

## Status Management and Workflow

1. **Status Values**: Track work using these states:
   - todo: Not yet initiated
   - in_progress: Currently active (maintain single focus)
   - done: Successfully completed

2. **Workflow Practices**:
   - Update status dynamically as work progresses
   - Mark completion immediately upon task finish
   - Limit active work to ONE task at any given time
   - Complete current activities before initiating new ones
   - Remove obsolete tasks from tracking entirely

3. **Completion Criteria**:
   - Mark tasks as done only when fully achieved
   - Keep status as in_progress if errors, blocks, or partial completion exist
   - Create new tasks for discovered issues or dependencies
   - Never mark done when:
       - Test suites are failing
       - Implementation remains incomplete
       - Unresolved errors persist
       - Required resources are unavailable

4. **Task Organization**:
   - Write precise, actionable descriptions
   - Decompose complex work into manageable units
   - Use descriptive, clear naming conventions

When uncertain, favor using this tool. Proactive task management demonstrates
systematic approach and ensures comprehensive requirement fulfillment.
"""

_DETAILED_STR_REPLACE_EDITOR_DESCRIPTION = """Custom editing tool for viewing, creating and editing files in plain-text format
* State is persistent across command calls and discussions with the user
* If `path` is a text file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The following binary file extensions can be viewed in Markdown format: [".xlsx", ".pptx", ".wav", ".mp3", ".m4a", ".flac", ".pdf", ".docx"]. IT DOES NOT HANDLE IMAGES.
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
* The `undo_edit` command will revert the last edit made to the file at `path`
* This tool can be used for creating and editing files in plain-text format.


Before using this tool:
1. Use the view tool to understand the file's contents and context
2. Verify the directory path is correct (only applicable when creating new files):
   - Use the view tool to verify the parent directory exists and is the correct location

When making edits:
   - Ensure the edit results in idiomatic, correct code
   - Do not leave the code in a broken state
   - Always use absolute file paths (starting with /)

CRITICAL REQUIREMENTS FOR USING THIS TOOL:

1. EXACT MATCHING: The `old_str` parameter must match EXACTLY one or more consecutive lines from the file, including all whitespace and indentation. The tool will fail if `old_str` matches multiple locations or doesn't match exactly with the file content.

2. UNIQUENESS: The `old_str` must uniquely identify a single instance in the file:
   - Include sufficient context before and after the change point (3-5 lines recommended)
   - If not unique, the replacement will not be performed

3. REPLACEMENT: The `new_str` parameter should contain the edited lines that replace the `old_str`. Both strings must be different.

Remember: when making multiple file edits in a row to the same file, you should prefer to send all edits in a single message with multiple calls to this tool, rather than multiple messages with a single call each.
"""


def get_openhands_tools(
    max_timeout: int | None = None,
    workspace_mount_path_in_sandbox: str = "/workspace",
    enable_think: bool = False,
    enable_task_tracker: bool = True,
) -> list[dict[str, Any]]:
  """Returns the OpenHands CodeActAgent tool schemas matching nv-OpenHands@0d766ad0."""
  if max_timeout is None:
    max_timeout = int(os.getenv("COMMAND_EXEC_TIMEOUT", "60"))

  tools: list[dict[str, Any]] = [
      {
          "type": "function",
          "function": {
              "name": "execute_bash",
              "description": _DETAILED_BASH_DESCRIPTION,
              "parameters": {
                  "type": "object",
                  "properties": {
                      "command": {
                          "type": "string",
                          "description": (
                              "The bash command to execute. Can be empty string"
                              " to view additional logs when previous exit code"
                              " is `-1`. Can be `C-c` (Ctrl+C) to interrupt the"
                              " currently running process. Note: You can only"
                              " execute one bash command at a time. If you need"
                              " to run multiple commands sequentially, you can"
                              " use `&&` or `;` to chain them together."
                          ),
                      },
                      "is_input": {
                          "type": "string",
                          "description": (
                              "If True, the command is an input to the running"
                              " process. If False, the command is a bash"
                              " command to be executed in the terminal. Default"
                              " is False."
                          ),
                          "enum": ["true", "false"],
                      },
                      "timeout": {
                          "type": "number",
                          "description": (
                              "Optional. Sets a hard timeout in seconds for the"
                              " command execution. If not provided, the command"
                              " will use the default soft timeout behavior. Max"
                              f" timeout allowed is {max_timeout} seconds."
                          ),
                      },
                      "security_risk": {
                          "type": "string",
                          "description": _SECURITY_RISK_DESC,
                          "enum": list(_RISK_LEVELS),
                      },
                  },
                  "required": ["command", "security_risk"],
              },
          },
      },
  ]
  if enable_think:
    tools.append({
        "type": "function",
        "function": {
            "name": "think",
            "description": _THINK_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "The thought to log.",
                    },
                },
                "required": ["thought"],
            },
        },
    })
  tools.append({
      "type": "function",
      "function": {
          "name": "finish",
          "description": _FINISH_DESCRIPTION,
          "parameters": {
              "type": "object",
              "required": ["message"],
              "properties": {
                  "message": {
                      "type": "string",
                      "description": "Final message to send to the user",
                  },
              },
          },
      },
  })
  if enable_task_tracker:
    tools.append({
        "type": "function",
        "function": {
            "name": "task_tracker",
            "description": _DETAILED_TASK_TRACKER_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["view", "plan"],
                        "description": (
                            "The command to execute. `view` shows the current"
                            " task list. `plan` creates or updates the task"
                            " list based on provided requirements and progress."
                            " Always `view` the current list before making"
                            " changes."
                        ),
                    },
                    "task_list": {
                        "type": "array",
                        "description": (
                            "The full task list. Required parameter of `plan`"
                            " command."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": "Unique task identifier",
                                },
                                "title": {
                                    "type": "string",
                                    "description": "Brief task description",
                                },
                                "status": {
                                    "type": "string",
                                    "description": "Current task status",
                                    "enum": ["todo", "in_progress", "done"],
                                },
                                "notes": {
                                    "type": "string",
                                    "description": (
                                        "Optional additional context or details"
                                    ),
                                },
                            },
                            "required": ["title", "status", "id"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    })
  tools.append({
      "type": "function",
      "function": {
          "name": "str_replace_editor",
          "description": _DETAILED_STR_REPLACE_EDITOR_DESCRIPTION,
          "parameters": {
              "type": "object",
              "properties": {
                  "command": {
                      "description": (
                          "The commands to run. Allowed options are: `view`,"
                          " `create`, `str_replace`, `insert`, `undo_edit`."
                      ),
                      "enum": [
                          "view",
                          "create",
                          "str_replace",
                          "insert",
                          "undo_edit",
                      ],
                      "type": "string",
                  },
                  "path": {
                      "description": (
                          "Absolute path to file or directory, e.g."
                          f" `{workspace_mount_path_in_sandbox}/file.py` or"
                          f" `{workspace_mount_path_in_sandbox}`."
                      ),
                      "type": "string",
                  },
                  "file_text": {
                      "description": (
                          "Required parameter of `create` command, with the"
                          " content of the file to be created."
                      ),
                      "type": "string",
                  },
                  "old_str": {
                      "description": (
                          "Required parameter of `str_replace` command"
                          " containing the string in `path` to replace."
                      ),
                      "type": "string",
                  },
                  "new_str": {
                      "description": (
                          "Optional parameter of `str_replace` command"
                          " containing the new string (if not given, no string"
                          " will be added). Required parameter of `insert`"
                          " command containing the string to insert."
                      ),
                      "type": "string",
                  },
                  "insert_line": {
                      "description": (
                          "Required parameter of `insert` command. The"
                          " `new_str` will be inserted AFTER the line"
                          " `insert_line` of `path`."
                      ),
                      "type": "integer",
                  },
                  "view_range": {
                      "description": (
                          "Optional parameter of `view` command when `path`"
                          " points to a file. If none is given, the full file"
                          " is shown. If provided, the file will be shown in"
                          " the indicated line number range, e.g. [11, 12] will"
                          " show lines 11 and 12. Indexing at 1 to start."
                          " Setting `[start_line, -1]` shows all lines from"
                          " `start_line` to the end of the file."
                      ),
                      "items": {"type": "integer"},
                      "type": "array",
                  },
                  "security_risk": {
                      "type": "string",
                      "description": _SECURITY_RISK_DESC,
                      "enum": list(_RISK_LEVELS),
                  },
              },
              "required": ["command", "path", "security_risk"],
          },
      },
  })
  return tools


OPENHANDS_TOOLS = get_openhands_tools()

OPENHANDS_CODEACT_SYSTEM_PROMPT = """You are OpenHands agent, a helpful AI assistant that can interact with a computer to solve tasks.

<ROLE>
Your primary role is to assist users by executing commands, modifying code, and solving technical problems effectively. You should be thorough, methodical, and prioritize quality over speed.
* If the user asks a question, like "why is X happening", don't try to fix the problem. Just give an answer to the question.
</ROLE>

<EFFICIENCY>
* Each action you take is somewhat expensive. Wherever possible, combine multiple actions into a single action, e.g. combine multiple bash commands into one, using sed and grep to edit/view multiple files at once.
* When exploring the codebase, use efficient tools like find, grep, and git commands with appropriate filters to minimize unnecessary operations.
</EFFICIENCY>

<FILE_SYSTEM_GUIDELINES>
* When a user provides a file path, do NOT assume it's relative to the current working directory. First explore the file system to locate the file before working on it.
* If asked to edit a file, edit the file directly, rather than creating a new file with a different filename.
* For global search-and-replace operations, consider using `sed` instead of opening file editors multiple times.
* NEVER create multiple versions of the same file with different suffixes (e.g., file_test.py, file_fix.py, file_simple.py). Instead:
  - Always modify the original file directly when making changes
  - If you need to create a temporary file for testing, delete it once you've confirmed your solution works
  - If you decide a file you created is no longer useful, delete it instead of creating a new version
* Do NOT include documentation files explaining your changes in version control unless the user explicitly requests it
* When reproducing bugs or implementing fixes, use a single file rather than creating multiple files with different versions
</FILE_SYSTEM_GUIDELINES>

<CODE_QUALITY>
* Write clean, efficient code with minimal comments. Avoid redundancy in comments: Do not repeat information that can be easily inferred from the code itself.
* When implementing solutions, focus on making the minimal changes needed to solve the problem.
* Before implementing any changes, first thoroughly understand the codebase through exploration.
* If you are adding a lot of code to a function or file, consider splitting the function or file into smaller pieces when appropriate.
* Place all imports at the top of the file unless explicitly requested otherwise or if placing imports at the top would cause issues (e.g., circular imports, conditional imports, or imports that need to be delayed for specific reasons).
* If working in a git repo, before you commit code create a .gitignore file if one doesn't exist. And if there are existing files that should not be included then update the .gitignore file as appropriate.
</CODE_QUALITY>

<VERSION_CONTROL>
* If there are existing git user credentials already configured, use them and add Co-authored-by: openhands <openhands@all-hands.dev> to any commits messages you make. if a git config doesn't exist use "openhands" as the user.name and "openhands@all-hands.dev" as the user.email by default, unless explicitly instructed otherwise.
* Exercise caution with git operations. Do NOT make potentially dangerous changes (e.g., pushing to main, deleting repositories) unless explicitly asked to do so.
* When committing changes, use `git status` to see all modified files, and stage all files necessary for the commit. Use `git commit -a` whenever possible.
* Do NOT commit files that typically shouldn't go into version control (e.g., node_modules/, .env files, build directories, cache files, large binaries) unless explicitly instructed by the user.
* If unsure about committing certain files, check for the presence of .gitignore files or ask the user for clarification.
</VERSION_CONTROL>

<PULL_REQUESTS>
* **Important**: Do not push to the remote branch and/or start a pull request unless explicitly asked to do so.
* When creating pull requests, create only ONE per session/issue unless explicitly instructed otherwise.
* When working with an existing PR, update it with new commits rather than creating additional PRs for the same issue.
* When updating a PR, preserve the original PR title and purpose, updating description only when necessary.
</PULL_REQUESTS>

<PROBLEM_SOLVING_WORKFLOW>
1. EXPLORATION: Thoroughly explore relevant files and understand the context before proposing solutions
2. ANALYSIS: Consider multiple approaches and select the most promising one
3. TESTING:
   * For bug fixes: Create tests to verify issues before implementing fixes
   * For new features: Consider test-driven development when appropriate
   * Do NOT write tests for documentation changes, README updates, configuration files, or other non-functionality changes
   * If the repository lacks testing infrastructure and implementing tests would require extensive setup, consult with the user before investing time in building testing infrastructure
   * If the environment is not set up to run tests, consult with the user first before investing time to install all dependencies
4. IMPLEMENTATION:
   * Make focused, minimal changes to address the problem
   * Always modify existing files directly rather than creating new versions with different suffixes
   * If you create temporary files for testing, delete them after confirming your solution works
5. VERIFICATION: If the environment is set up to run tests, test your implementation thoroughly, including edge cases. If the environment is not set up to run tests, consult with the user first before investing time to run tests.
</PROBLEM_SOLVING_WORKFLOW>

<SECURITY>
* Only use GITHUB_TOKEN and other credentials in ways the user has explicitly requested and would expect.
* Use APIs to work with GitHub or other platforms, unless the user asks otherwise or your task requires browsing.
</SECURITY>

<SECURITY_RISK_ASSESSMENT>
# 🔐 Security Risk Policy
When using tools that support the security_risk parameter, assess the safety risk of your actions:


- **LOW**: Read-only actions inside sandbox.
  - Inspecting container files, calculations, viewing docs.
- **MEDIUM**: Container-scoped edits and installs.
  - Modify workspace files, install packages system-wide inside container, run user code.
- **HIGH**: Data exfiltration or privilege breaks.
  - Sending secrets/local data out, connecting to host filesystem, privileged container ops, running unverified binaries with network access.



**Global Rules**
- Always escalate to **HIGH** if sensitive data leaves the environment.
</SECURITY_RISK_ASSESSMENT>

<EXTERNAL_SERVICES>
* When interacting with external services like GitHub, GitLab, or Bitbucket, use their respective APIs instead of browser-based interactions whenever possible.
* Only resort to browser-based interactions with these services if specifically requested by the user or if the required operation cannot be performed via API.
</EXTERNAL_SERVICES>

<ENVIRONMENT_SETUP>
* When user asks you to run an application, don't stop if the application is not installed. Instead, please install the application and run the command again.
* If you encounter missing dependencies:
  1. First, look around in the repository for existing dependency files (requirements.txt, pyproject.toml, package.json, Gemfile, etc.)
  2. If dependency files exist, use them to install all dependencies at once (e.g., `pip install -r requirements.txt`, `npm install`, etc.)
  3. Only install individual packages directly if no dependency files are found or if only specific packages are needed
* Similarly, if you encounter missing dependencies for essential tools requested by the user, install them when possible.
</ENVIRONMENT_SETUP>

<TROUBLESHOOTING>
* If you've made repeated attempts to solve a problem but tests still fail or the user reports it's still broken:
  1. Step back and reflect on 5-7 different possible sources of the problem
  2. Assess the likelihood of each possible cause
  3. Methodically address the most likely causes, starting with the highest probability
  4. Document your reasoning process
* When you run into any major issue while executing a plan from the user, please don't try to directly work around it. Instead, propose a new plan and confirm with the user before proceeding.
</TROUBLESHOOTING>

<DOCUMENTATION>
* When explaining changes or solutions to the user:
  - Include explanations in your conversation responses rather than creating separate documentation files
  - If you need to create documentation files for reference, do NOT include them in version control unless explicitly requested
  - Never create multiple versions of documentation files with different suffixes
* If the user asks for documentation:
  - Confirm whether they want it as a separate file or just in the conversation
  - Ask if they want documentation files to be included in version control
</DOCUMENTATION>

<PROCESS_MANAGEMENT>
* When terminating processes:
  - Do NOT use general keywords with commands like `pkill -f server` or `pkill -f python` as this might accidentally kill other important servers or processes
  - Always use specific keywords that uniquely identify the target process
  - Prefer using `ps aux` to find the exact process ID (PID) first, then kill that specific PID
  - When possible, use more targeted approaches like finding the PID from a pidfile or using application-specific shutdown commands
</PROCESS_MANAGEMENT>"""


def format_qwen_tools_system_prompt(
    system_prompt: str = OPENHANDS_CODEACT_SYSTEM_PROMPT,
    tools: list[dict[str, Any]] | None = None,
) -> str:
  """Renders the Qwen3/Qwen3.5 native tool-calling system prompt."""
  if tools is None:
    tools = get_openhands_tools()
  tools_json = "\n".join(json.dumps(tool, ensure_ascii=False) for tool in tools)
  return (
      "# Tools\n\n"
      "You have access to the following functions:\n\n"
      "<tools>\n"
      f"{tools_json}\n"
      "</tools>\n\n"
      "If you choose to call a function ONLY reply in the following format"
      " with NO suffix:\n\n"
      "<tool_call>\n"
      "<function=example_function_name>\n"
      "<parameter=example_parameter_1>\n"
      "value_1\n"
      "</parameter>\n"
      "<parameter=example_parameter_2>\n"
      "This is the value for the second parameter\n"
      "that can span\n"
      "multiple lines\n"
      "</parameter>\n"
      "</function>\n"
      "</tool_call>\n\n"
      "<IMPORTANT>\n"
      "Reminder:\n"
      "- Function calls MUST follow the specified format: an inner"
      " <function=...></function> block must be nested within"
      " <tool_call></tool_call> XML tags\n"
      "- Required parameters MUST be specified\n"
      "- You may provide optional reasoning for your function call in"
      " natural language BEFORE the function call, but NOT after\n"
      "- If there is no function call available, answer the question like"
      " normal with your current knowledge and do not tell the user about"
      " function calls\n"
      "</IMPORTANT>\n\n"
      f"{system_prompt.strip()}"
  )


OPENHANDS_SYSTEM_PROMPT = format_qwen_tools_system_prompt()

OPENHANDS_FAKE_USER_RESPONSE = (
    "Please continue working on the task on whatever approach you think is"
    " suitable.\n"
    "When you think you have solved the question, please use the finish tool"
    " and include your final answer in the message parameter of the finish"
    " tool.\n"
    "IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP."
)

# ==============================================================================
# Agent User Prompts
# ==============================================================================

SWE_USER_PROMPT_FN_CALL = """Consider the following github issue:
<github_issue>
{problem_statement}
</github_issue>

Can you help me implement the necessary changes to the repository to fix the <github_issue>?
I've already taken care of all changes to any of the test files described in the <github_issue>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Your task is to make the minimal changes to non-tests files in the /testbed directory to ensure the <github_issue> is satisfied.

IMPORTANT TIP:
Follow these steps to resolve the issue:
1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.
2. Create a script ('reproduce_issue.py') to reproduce the error and execute it to confirm the error
  2.1 reproduce_issue.py script finishes quickly after checking the error, fix etc. There no long running background servers for django for instance etc. It should be a quick script which checks the error and fix to provide a visible response.
  2.2 SUPER IMPORTANT: to ensure this reproduce_script.py must have a timeout logic of 20 seconds. If the script runs for more than 30 seconds, it should output a timeout message and you can interpret accordingly.
3. Edit the sourcecode of the repo to resolve the issue
4. Rerun your reproduce script and confirm that the error is fixed!
5. Think about edgecases and make sure your fix handles them as well

VERY IMPORTANT: each response must include both reasoning and function call to solve the task.
You are being told a million times, each response must include a function call. Must inlcude a function call at all costs.

You can take multiple turns to solve the task. So please only finish / submit when you are confident in your response. Dont rush. Be comprehensive.
You are being told a million times, please dont just submit without proper reasoning. Try to fully analyse the problem statement, explore the repository, reproduce the issue, fix it, check edge cases and then submit.

Your thinking should be thorough and so it's fine if it's very long.
VERY IMPORTANT: file_editor old_str and new_str must be w/o the line numbers. line numbers are only shown in the view for clarity.

Also if a file_editor edit fails, its a good idea to view the file near the edit location before trying to edit again. Dont keep trying the same edit over and over again. It will keep leading to the same failure.
Again do not get stuck trying to do the same thing over and over again. Please be efficient.
"""

SWE_USER_PROMPT = """Consider the following github issue:
<github_issue>
{problem_statement}
</github_issue>

Can you help me implement the necessary changes to the repository to fix the <github_issue>?
I've already taken care of all changes to any of the test files described in the <github_issue>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Your task is to make the minimal changes to non-tests files in the /testbed directory to ensure the <github_issue> is satisfied.

IMPORTANT TIP:
Follow these steps to resolve the issue:
1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.
2. Create a script ('reproduce_issue.py') to reproduce the error and execute it to confirm the error
3. Edit the sourcecode of the repo to resolve the issue
4. Rerun your reproduce script and confirm that the error is fixed!
5. Think about edgecases and make sure your fix handles them as well
6. When viewing large files, use specific line-ranges, usually within 50 to 100 lines) as required
7. NOTE: The repository is at '/testbed' and the current working directory is already '/testbed', so DO NOT include 'testbed/' or 'testbed.' in relative paths in bash commands or reproduction python files.
"""

SWEAGENT_USER_PROMPT = """I have uploaded a python code repository in the /testbed directory.

Now consider the following Github issue:

<github_issue>
{problem_statement}
</github_issue>

Can you help me implement the necessary changes to the repository to fix the <github_issue>?
I have already taken care of all changes to any of the test files described in the <github_issue>. This means you DON'T have to modify the testing logic or any of the tests in any way! Your task is to make changes to non-test files in the /testbed directory to ensure the <github_issue> is resolved.

Follow these steps to resolve the issue:
1. First, explore the codebase to locate and understand the code relevant to the <github_issue>.
  - Use efficient search commands to identify key files and functions.
  - You should err on the side of caution and look at various relevant files and build your understanding of
    - how the code works
    - what are the expected behaviors and edge cases
    - what are the potential root causes for the given issue

2. Assess whether you can reproduce the issue:
    - Create a script at '/testbed/reproduce_issue.py' that demonstrates the error.
    - Execute this script to confirm the error behavior.
    - You should reproduce the issue before fixing it.
    - Your reproduction script should also assert the expected behavior for the fixed code.

3. Analyze the root cause:
    - Identify the underlying problem based on your code exploration and reproduction results.
    - Critically analyze different potential approaches to fix the issue.
    - You NEED to explicitly reason about multiple approaches to fix the issue. Next, find the most elegant and effective solution among them considering the tradeoffs (correctness, generality, side effects, etc.).
    - You would need to reason about execution paths, edge cases, and other potential issues. You should look at the unit tests to understand the expected behavior of the relevant code.

4. Implement your solution:
    - Make targeted changes to the necessary files following idiomatic code patterns once you determine the root cause.
    - You should be thorough and methodical.

5. Verify your solution:
    - Rerun your reproduction script to confirm the error is fixed.
    - If verification fails, iterate on your solution until successful. If you identify the reproduction script is buggy, adjust it as needed.

6. Run unit tests:
    - Find and run the relevant unit tests relevant to the performed fix.
    - You should run the unit tests to ensure your solution is correct and does not cause any regressions.
    - In cases where the unit tests are do not pass, you should consider whether the unit tests does not reflect the *new* expected behavior of the code. If so, you can test it by writing additional edge test cases.
    - Use the existing test runner to run the unit tests you identify as relevant to the changes you made. For example:
        - `python -m pytest -xvs sympy/physics/units/tests/test_dimensions_transcendental.py`
        - `python -m pytest tests/test_domain_py.py::test_pymethod_options`
        - `./tests/runtests.py constraints.tests.CheckConstraintTests -v 2`
    - RUN ALL relevant unit tests to ensure your solution is correct and does not cause any regressions.

7. Test edge cases:
    - Identify potential edge cases that might challenge your solution.
    - Create additional test cases in a separate file '/testbed/edge_case_tests.py'.
    - Execute these tests to verify your solution's robustness.
    - You should run multiple rounds of edge cases. When creating edge cases:
      - Consider complex scenarios beyond the original issue description
      - Test for regressions to ensure existing functionality remains intact

8. Refine if necessary:
    - If edge case testing reveals issues, refine your solution accordingly.
    - Ensure your final implementation handles all identified scenarios correctly.
    - Document any assumptions or limitations of your solution.

9. Submit your solution:
    - Once you have verified your solution, submit your solution using the `submit` tool.

A successful resolution means:
- The specific error/issue described no longer occurs
- Your changes maintain compatibility with existing functionality
- Edge cases are properly handled


Additional recommendations:
- You should be thorough, methodical, and prioritize quality over speed. Be comprehensive.
- You should think carefully before making the tool call about what should be done. However, each step should only use one tool call. YOU SHOULD NOT USE TOOLS INSIDE YOUR THOUGHT PROCESS. YOU SHOULD PRIMARILY USE THINKING FOR IDENTIFYING THE ROOT CAUSE OF THE ISSUE, MAKING THE CHANGES, AND CREATING TEST CASES (REPRODUCTION OR EDGE CASES).
- Each action you take is somewhat expensive. Wherever possible, combine multiple actions into a single action (e.g., combine multiple bash commands, use sed/grep for bulk operations).
    - Your grep commands should identify both relevant files and line numbers so you can use the file_editor tool.
    - Use grep with `-A -B -C` flags to quickly identify the relevant code blocks during your exploration.
- When exploring the codebase, use targeted search patterns to minimize unnecessary operations.
- When creating edge cases, you should look at the relevant existing tests to understand existing "regression" test cases. Ensure the fix doesn't break existing functionality.
"""


OPENHANDS_USER_PROMPT = """<uploaded_files>
/testbed
</uploaded_files>


I've uploaded a python code repository in the directory /testbed. Consider the following issue description:

<issue_description>
{problem_statement}
</issue_description>

Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?
I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Also the development environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.
Your task is to make the minimal changes to non-test files in the /testbed directory to ensure the <issue_description> is satisfied.

Follow these phases to resolve the issue:

Phase 1. READING: read the problem and reword it in clearer terms
   1.1 If there are code or config snippets. Express in words any best practices or conventions in them.
   1.2 Hightlight message errors, method names, variables, file names, stack traces, and technical details.
   1.3 Explain the problem in clear terms.
   1.4 Enumerate the steps to reproduce the problem.
   1.5 Hightlight any best practices to take into account when testing and fixing the issue

Phase 2. RUNNING: install and run the tests on the repository
   2.1 Follow the readme
   2.2 Install the environment and anything needed
   2.2 Iterate and figure out how to run the tests

Phase 3. EXPLORATION: find the files that are related to the problem and possible solutions
   3.1 Use `grep` to search for relevant methods, classes, keywords and error messages.
   3.2 Identify all files related to the problem statement.
   3.3 Propose the methods and files to fix the issue and explain why.
   3.4 From the possible file locations, select the most likely location to fix the issue.

Phase 4. TEST CREATION: before implementing any fix, create a script to reproduce and verify the issue.
   4.1 Look at existing test files in the repository to understand the test format/structure.
   4.2 Create a minimal reproduction script that reproduces the located issue.
   4.3 Run the reproduction script to confirm you are reproducing the issue.
   4.4 Adjust the reproduction script as necessary.

Phase 5. FIX ANALYSIS: state clearly the problem and how to fix it
   5.1 State clearly what the problem is.
   5.2 State clearly where the problem is located.
   5.3 State clearly how the test reproduces the issue.
   5.4 State clearly the best practices to take into account in the fix.
   5.5 State clearly how to fix the problem.

Phase 6. FIX IMPLEMENTATION: Edit the source code to implement your chosen solution.
   6.1 Make minimal, focused changes to fix the issue.

Phase 7. VERIFICATION: Test your implementation thoroughly.
   7.1 Run your reproduction script to verify the fix works.
   7.2 Add edge cases to your test script to ensure comprehensive coverage.
   7.3 Run existing tests related to the modified code to ensure you haven't broken anything.

8. FINAL REVIEW: Carefully re-read the problem description and compare your changes with the base commit .
   8.1 Ensure you've fully addressed all requirements.
   8.2 Run any tests in the repository related to:
     8.2.1 The issue you are fixing
     8.2.2 The files you modified
     8.2.3 The functions you changed
   8.3 If any tests fail, revise your implementation until all tests pass

Be thorough in your exploration, testing, and reasoning. It's fine if your thinking process is lengthy - quality and completeness are more important than brevity."""


def format_openhands_user_prompt(
    problem_statement: str,
    workspace_path: str = "/testbed",
    repo_language: str = "python",
    base_commit: str = "",
) -> str:
  """Renders the OpenHands swe_default.j2 user prompt."""
  prompt = OPENHANDS_USER_PROMPT
  if workspace_path != "/testbed":
    prompt = prompt.replace(
        "<uploaded_files>\n/testbed\n</uploaded_files>",
        f"<uploaded_files>\n{workspace_path}\n</uploaded_files>",
    )
    prompt = prompt.replace(
        "in the directory /testbed.",
        f"in the directory {workspace_path}.",
    )
    prompt = prompt.replace(
        "in the /testbed directory",
        f"in the {workspace_path} directory",
    )
  if repo_language != "python":
    prompt = prompt.replace(
        "I've uploaded a python code repository",
        f"I've uploaded a {repo_language} code repository",
    )
  if base_commit:
    prompt = prompt.replace(
        "compare your changes with the base commit .",
        f"compare your changes with the base commit {base_commit}.",
    )
  return prompt.replace("{problem_statement}", str(problem_statement))


def get_system_prompt(
    scaffold: str = "r2egym",
    use_fn_calling: bool = False,
) -> str:
  """Get system prompt for the given scaffold and function calling mode."""
  if scaffold == "sweagent":
    return SWEAGENT_SYSTEM_PROMPT
  elif scaffold in OPENHANDS_SCAFFOLDS:
    return OPENHANDS_SYSTEM_PROMPT
  return SWE_SYSTEM_PROMPT_FN_CALL if use_fn_calling else SWE_SYSTEM_PROMPT


def get_user_prompt_template(
    scaffold: str = "r2egym",
    use_fn_calling: bool = False,
) -> str:
  """Get user prompt template for the given scaffold and function calling mode."""
  if scaffold == "sweagent":
    return SWEAGENT_USER_PROMPT
  elif scaffold in OPENHANDS_SCAFFOLDS:
    return OPENHANDS_USER_PROMPT
  return SWE_USER_PROMPT_FN_CALL if use_fn_calling else SWE_USER_PROMPT


# ==============================================================================
# Environment Sandbox Fleet Pod Templates
# ==============================================================================

DEFAULT_OPENHANDS_KEEPALIVE_CMD = [
    "sh",
    "-c",
    (
        "chmod +x /oh/openhands-agent-server 2>/dev/null || true; ([ -d /oh/glibc236 ] && export LD_LIBRARY_PATH=\"/oh/glibc236:${LD_LIBRARY_PATH:-}\"); ([ -d"
        " /testbed ] && [ ! -e /workspace ] && ln -s /testbed /workspace"
        " 2>/dev/null || true); ([ -d /workspace ] && [ ! -e /testbed ] && ln"
        " -s /workspace /testbed 2>/dev/null || true); git config --global"
        " --add safe.directory '*' 2>/dev/null || true; [ -d /testbed ] && cd"
        " /testbed; if [ -x /oh/openhands-agent-server ]; then exec"
        " /oh/openhands-agent-server --host 0.0.0.0 --port 8000; elif [ -x"
        " /usr/local/bin/openhands-agent-server ]; then exec tini --"
        " /usr/local/bin/openhands-agent-server --host 0.0.0.0 --port 8000;"
        " elif [ -x /openhands/poetry/openhands-ai-5O4_aCHf-py3.12/bin/agent-server ]; then"
        " exec /openhands/poetry/openhands-ai-5O4_aCHf-py3.12/bin/agent-server --host 0.0.0.0 --port 8000;"
        " fi"
    ),
]


def get_openhands_pod_template(
    node_selector: Optional[dict[str, str]] = None,
) -> Any:
  """Builds and returns the TemplateSpec for OpenHands agent sandbox."""
  try:
    from agent_sandbox_rl import (  # pytype: disable=import-error
        ResourceSpec,
        TemplateSpec,
    )
  except ImportError as e:
    raise ImportError(
        "use_agent_sandbox=True strictly requires the 'agent_sandbox_rl'"
        " package. Install via: pip install"
        " git+https://github.com/kubernetes-sigs/agent-sandbox.git#subdirectory=examples/agent-sandbox-rl"
    ) from e

  session_key = os.getenv("SANDBOX_SESSION_KEY", "")
  if os.getenv("AGENT_SERVER_COMMAND"):
    try:
      keepalive_cmd = json.loads(os.environ["AGENT_SERVER_COMMAND"])
    except Exception:
      keepalive_cmd = os.environ["AGENT_SERVER_COMMAND"].split()
  else:
    keepalive_cmd = list(DEFAULT_OPENHANDS_KEEPALIVE_CMD)

  server_image = (
      os.getenv("OPENHANDS_SERVER_IMAGE")
      or os.getenv("SANDBOX_RUNTIME_CONTAINER_IMAGE")
      or os.getenv("AGENT_SERVER_IMAGE")
      or "gcr.io/cloud-tpu-multipod-dev/tunix/openhands-agent-server:0.62"
  )

  extra_pod_spec = {
      "initContainers": [{
          "name": "oh-server",
          "image": server_image,
          "command": [
              "sh",
              "-c",
              (
                  "cp -a /opt/oh/. /oh/ 2>/dev/null || cp -a"
                  " /usr/local/bin/openhands-agent-server /oh/ 2>/dev/null ||"
                  " cp -a /openhands/poetry/*/bin/agent-server /oh/openhands-agent-server 2>/dev/null || true"
              ),
          ],
          "volumeMounts": [{"name": "oh", "mountPath": "/oh"}],
      }],
      "volumes": [{"name": "oh", "emptyDir": {}}],
      "containers": [{
          "ports": [{"containerPort": 8000}],
          "volumeMounts": [{"name": "oh", "mountPath": "/oh"}],
          "readinessProbe": {
              "httpGet": {"path": "/health", "port": 8000},
              "periodSeconds": 2,
              "failureThreshold": 150,
          },
          "resources": {
              "limits": {
                  "cpu": os.getenv("SANDBOX_CPU_LIMIT", "2"),
                  "memory": os.getenv("SANDBOX_MEM_LIMIT", "4Gi"),
              }
          },
          "env": (
              [{"name": "OH_SESSION_API_KEYS_0", "value": session_key}]
              if session_key
              else []
          ),
      }],
  }

  tolerations_raw = os.getenv("SANDBOX_TOLERATIONS")
  if tolerations_raw:
    try:
      extra_pod_spec["tolerations"] = json.loads(tolerations_raw)
    except Exception:
      try:
        import ast  # pylint: disable=g-import-not-at-top
        extra_pod_spec["tolerations"] = ast.literal_eval(tolerations_raw)
      except Exception as e:
        logging.warning("Failed to parse SANDBOX_TOLERATIONS: %s", e)

  if "tolerations" not in extra_pod_spec and node_selector and node_selector.get("cloud.google.com/gke-nodepool") == "sandbox-np":
    extra_pod_spec["tolerations"] = [{
        "key": "workload",
        "operator": "Equal",
        "value": "sandbox",
        "effect": "NoSchedule",
    }]

  return TemplateSpec(
      keepalive_command=keepalive_cmd,
      resources=ResourceSpec(
          cpu=os.getenv("SANDBOX_CPU", "500m"),
          memory=os.getenv("SANDBOX_MEM", "1Gi"),
      ),
      extra_pod_spec=extra_pod_spec,
      node_selector=node_selector,
  )


def get_template(
    scaffold: str,
    node_selector: Optional[dict[str, str]] = None,
) -> Any:
  """Returns the fleet TemplateSpec for the given scaffold, or None."""
  if scaffold in OPENHANDS_SCAFFOLDS:
    return get_openhands_pod_template(node_selector=node_selector)
  return None

