"""Task decomposition and safe tool registry for the Expert 802 agent runtime."""
from dataclasses import dataclass, asdict
from .programming import build_work_plan
from .code_intelligence import inspect_project, InspectionError

@dataclass
class ToolResult:
    ok: bool
    name: str
    output: dict
    error: str = ""

class ToolRegistry:
    def __init__(self):
        self._tools = {"inspect_project": inspect_project}

    def describe(self):
        return [{"name":name,"description":"Statically inspect a bounded source snapshot; never execute uploaded code."} for name in sorted(self._tools)]

    def call(self, name, arguments):
        tool = self._tools.get(name)
        if tool is None:
            return asdict(ToolResult(False, name, {}, "Tool is not registered."))
        try:
            return asdict(ToolResult(True, name, tool(**arguments), ""))
        except (InspectionError, TypeError, ValueError) as error:
            return asdict(ToolResult(False, name, {}, str(error)))

class AgentPlanner:
    def __init__(self):
        self.tools = ToolRegistry()

    def plan(self, task, files=None):
        result = build_work_plan(task, files)
        result["available_tools"] = self.tools.describe()
        result["execution_mode"] = "plan_and_inspect"
        result["code_execution_enabled"] = False
        return result

    def inspect(self, files):
        return self.tools.call("inspect_project", {"files":files})
