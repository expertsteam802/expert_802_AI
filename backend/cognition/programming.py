"""Programming task classification and engineering work planning."""
import re
CODE = re.compile(r"\b(code|coding|program|programming|software|website|app|api|python|javascript|typescript|html|css|sql|react|node|debug|bug|error|compile|build|deploy|repository|repo|function|class|database|script|terminal|git|github|refactor|test)\b", re.I)
ACTION = re.compile(r"\b(build|make|create|write|implement|fix|debug|repair|explain|review|analy[sz]e|refactor|test|deploy|optimi[sz]e|add|remove|change|update)\b", re.I)
PATTERNS = [
 ("debug", r"\b(fix|debug|repair|broken|error|exception|crash|not working|fails?)\b"),
 ("implement", r"\b(build|make|create|write|implement|add|develop)\b"),
 ("explain", r"\b(explain|understand|what does|how does|walk me through)\b"),
 ("review", r"\b(review|audit|inspect|analy[sz]e|check)\b"),
 ("refactor", r"\b(refactor|clean up|restructure|optimi[sz]e)\b"),
 ("test", r"\b(test|tests|unit test|coverage)\b"),
 ("deploy", r"\b(deploy|publish|release|production|vercel|hosting)\b"),
]
def is_programming_request(text):
 value=text or ""
 return bool(CODE.search(value) and ACTION.search(value))
def classify_programming_task(text):
 for kind,pattern in PATTERNS:
  if re.search(pattern,text or "",re.I): return kind
 return "engineering_question"
def build_work_plan(task,files=None):
 kind=classify_programming_task(task)
 plans={
 "debug":["Reproduce the failure and capture the exact error and environment.","Trace the failing path through source files and dependencies.","Identify the root cause before changing code.","Apply the smallest coherent fix and add a regression test.","Run relevant tests and verify deployment."],
 "implement":["Translate the requested behavior into acceptance criteria.","Map architecture, data flow, interfaces and constraints.","Design a modular change that fits the existing system.","Implement validation, error handling and tests.","Run regression checks and verify runtime behavior."],
 "explain":["Identify the language, framework and entry point.","Trace inputs through functions and data structures.","Explain execution flow, dependencies and side effects.","Call out assumptions, edge cases and security implications."],
 "review":["Map modules, entry points, dependencies and trust boundaries.","Inspect correctness, error handling, security and maintainability.","Prioritize findings by severity and impact.","Propose testable fixes with file and symbol references."],
 "refactor":["Record current behavior and establish regression tests.","Identify coupling, duplication and unclear module boundaries.","Define stable interfaces and migration steps.","Refactor incrementally while preserving behavior.","Run regression and performance checks."],
 "test":["Identify public interfaces and expected behavior.","Cover normal, boundary, failure and adversarial inputs.","Add unit and integration tests.","Run tests and report exact failures."],
 "deploy":["Identify platform, build command and runtime.","Check environment variables, secrets, routes and build configuration.","Reproduce the production build and inspect logs.","Fix the root cause, deploy and verify the live endpoint."],
 "engineering_question":["Identify the relevant language and system.","Establish desired behavior and technical constraints.","Use source code, tests and runtime evidence to form the answer."]}
 result={"kind":kind,"steps":plans[kind],"requires_project_files":kind in {"debug","implement","review","refactor","test"},"project_files_supplied":bool(files)}
 if not files and result["requires_project_files"]:
  result["next_input"]="Provide relevant source files, repository context, or exact error output so work is grounded in the actual project."
 return result
