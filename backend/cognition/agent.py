"""Task decomposition and safe tool registry for the Expert 802 agent runtime."""
from dataclasses import dataclass, asdict
from .programming import build_work_plan
from .code_intelligence import inspect_project, InspectionError
from .sandbox_runtime import SandboxRuntime

@dataclass
class ToolResult:
    ok: bool
    name: str
    output: dict
    error: str = ""

class ToolRegistry:
    def __init__(self):
        self.sandbox = SandboxRuntime()
        self._tools = {"inspect_project": inspect_project, "run_python_tests": self.run_python_tests, "check_javascript": self.check_javascript}

    def describe(self):
        descriptions = {
            "inspect_project": "Statically inspect a bounded source snapshot without executing it.",
            "run_python_tests": "Explicitly run Python unittest discovery in a private network-disabled Docker container.",
            "check_javascript": "Explicitly run Node syntax checking for one supplied JavaScript file in a private container.",
        }
        return [{"name":name,"description":descriptions[name],"ready":True if name=="inspect_project" else self.sandbox.health()["ready"]} for name in sorted(self._tools)]

    def run_python_tests(self, files):
        if not self.sandbox.health()["ready"]:
            raise InspectionError("Private Docker test runner is not ready.")
        result = self.sandbox.run(files, "python_unittest")
        if not result.get("ok"):
            raise InspectionError(result.get("error") or result.get("stderr") or "Python tests failed.")
        return result

    def check_javascript(self, files, target):
        if not self.sandbox.health()["ready"]:
            raise InspectionError("Private Docker test runner is not ready.")
        result = self.sandbox.run(files, "node_check", target)
        if not result.get("ok"):
            raise InspectionError(result.get("error") or result.get("stderr") or "JavaScript syntax check failed.")
        return result

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
