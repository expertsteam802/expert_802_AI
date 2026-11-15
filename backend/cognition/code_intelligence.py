"""Safe, non-executing project inspection for common programming languages."""
import ast
import json
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import PurePosixPath

MAX_FILES = 40
MAX_FILE_BYTES = 120000
MAX_TOTAL_BYTES = 1000000
ALLOWED_SUFFIXES = {".py",".js",".jsx",".mjs",".cjs",".ts",".tsx",".html",".css",".json",".sql",".md",".txt",".yml",".yaml"}

class InspectionError(ValueError):
    pass

def safe_path(value):
    if not isinstance(value, str) or not value.strip():
        raise InspectionError("Every file needs a relative path.")
    path = PurePosixPath(value.replace(chr(92), "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise InspectionError("Absolute paths and parent-directory traversal are not allowed.")
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise InspectionError("Unsupported file type: " + (path.suffix or "(no extension)"))
    return str(path)

class HTMLIds(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
    def handle_starttag(self, tag, attrs):
        self.ids.extend(value for key, value in attrs if key == "id" and value)

def bracket_diagnostics(source, path):
    pairs = {")":"(", "]":"[", "}":"{"}
    stack, diagnostics = [], []
    quote, escaped, line, index = None, False, 1, 0
    while index < len(source):
        ch = source[index]
        if ch == "\n":
            line += 1
            if quote in ("'", '"') and not escaped:
                diagnostics.append({"severity":"warning","code":"possible_unterminated_string","path":path,"line":line-1,"message":"A quoted string may cross a line boundary."})
                quote = None
        if quote:
            if escaped:
                escaped = False
            elif ch == chr(92):
                escaped = True
            elif ch == quote:
                quote = None
            index += 1
            continue
        if ch in ("'", '"', chr(96)):
            quote = ch
            index += 1
            continue
        if source.startswith("//", index):
            end = source.find("\n", index)
            index = len(source) if end < 0 else end
            continue
        if source.startswith("/*", index):
            end = source.find("*/", index + 2)
            if end < 0:
                diagnostics.append({"severity":"error","code":"unterminated_comment","path":path,"line":line,"message":"Block comment is not closed."})
                break
            line += source[index:end+2].count("\n")
            index = end + 2
            continue
        if ch in "([{":
            stack.append((ch, line))
        elif ch in ")]}":
            if not stack or stack[-1][0] != pairs[ch]:
                diagnostics.append({"severity":"error","code":"unmatched_delimiter","path":path,"line":line,"message":"Unexpected closing delimiter " + ch + "."})
            else:
                stack.pop()
        index += 1
    for ch, opened_line in stack:
        diagnostics.append({"severity":"error","code":"unclosed_delimiter","path":path,"line":opened_line,"message":"Delimiter " + ch + " is never closed."})
    return diagnostics

def inspect_python(path, source):
    diagnostics, symbols, imports = [], [], []
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as error:
        return ([{"severity":"error","code":"python_syntax","path":path,"line":error.lineno or 1,"column":error.offset or 0,"message":error.msg}], symbols, imports)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append({"kind":"class" if isinstance(node, ast.ClassDef) else "function","name":node.name,"line":node.lineno})
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"eval","exec"}:
            diagnostics.append({"severity":"warning","code":"dynamic_"+node.func.id,"path":path,"line":node.lineno,"message":node.func.id+"() executes dynamic code; review its input trust boundary."})
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = node.func.value.id if isinstance(node.func.value, ast.Name) else ""
            if owner == "subprocess" and node.func.attr in {"run","Popen","call","check_call","check_output"}:
                shell_true = any(keyword.arg == "shell" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True for keyword in node.keywords)
                if shell_true:
                    diagnostics.append({"severity":"warning","code":"subprocess_shell_true","path":path,"line":node.lineno,"message":"Subprocess uses shell=True; validate interpolated arguments to avoid command injection."})
            if owner == "requests" and node.func.attr in {"get","post","put","delete","patch"} and not any(keyword.arg == "timeout" for keyword in node.keywords):
                diagnostics.append({"severity":"warning","code":"request_without_timeout","path":path,"line":node.lineno,"message":"HTTP request has no explicit timeout and may hang indefinitely."})
            if owner == "pickle" and node.func.attr in {"load","loads"}:
                diagnostics.append({"severity":"warning","code":"unsafe_pickle","path":path,"line":node.lineno,"message":"Unpickling untrusted data can execute code; use a safe serialization format."})
    return diagnostics, symbols, imports

def inspect_project(files):
    if not isinstance(files, list) or not files:
        raise InspectionError("Send a non-empty files array containing path and content.")
    if len(files) > MAX_FILES:
        raise InspectionError("Inspect at most " + str(MAX_FILES) + " files per request.")
    seen, normalized, total = set(), [], 0
    for item in files:
        if not isinstance(item, dict):
            raise InspectionError("Each file must be an object with path and content.")
        path = safe_path(item.get("path"))
        if path in seen:
            raise InspectionError("Duplicate file path: " + path)
        seen.add(path)
        source = item.get("content")
        if not isinstance(source, str):
            raise InspectionError("File content must be text: " + path)
        size = len(source.encode("utf-8"))
        if size > MAX_FILE_BYTES:
            raise InspectionError(path + " exceeds the per-file inspection limit.")
        total += size
        if total > MAX_TOTAL_BYTES:
            raise InspectionError("The project snapshot exceeds the 1 MB inspection limit.")
        normalized.append((path, source))
    diagnostics, symbols, imports, rows = [], [], [], []
    languages = Counter()
    for path, source in normalized:
        suffix = PurePosixPath(path).suffix.lower()
        language = {".py":"Python",".js":"JavaScript",".jsx":"JavaScript/JSX",".mjs":"JavaScript",".cjs":"JavaScript",".ts":"TypeScript",".tsx":"TypeScript/JSX",".html":"HTML",".css":"CSS",".json":"JSON",".sql":"SQL"}.get(suffix, "Text")
        languages[language] += 1
        rows.append({"path":path,"language":language,"lines":source.count("\n")+(1 if source else 0),"bytes":len(source.encode("utf-8"))})
        if suffix == ".py":
            found, file_symbols, file_imports = inspect_python(path, source)
            diagnostics.extend(found)
            symbols.extend({"path":path, **symbol} for symbol in file_symbols)
            imports.extend({"path":path,"module":name,"language":"Python"} for name in file_imports)
        elif suffix == ".json":
            try:
                json.loads(source)
            except json.JSONDecodeError as error:
                diagnostics.append({"severity":"error","code":"invalid_json","path":path,"line":error.lineno,"column":error.colno,"message":error.msg})
        elif suffix in {".js",".jsx",".mjs",".cjs",".ts",".tsx"}:
            diagnostics.extend(bracket_diagnostics(source, path))
            security_patterns = [
                (r"\\beval\\s*\\(", "javascript_eval", "eval() executes dynamically constructed code."),
                (r"\\binnerHTML\\s*=", "unsafe_inner_html", "Untrusted innerHTML can create cross-site scripting."),
                (r"dangerouslySetInnerHTML", "dangerous_html_injection", "Review the trust boundary for dangerouslySetInnerHTML."),
            ]
            for pattern, code, message in security_patterns:
                for match in re.finditer(pattern, source):
                    diagnostics.append({"severity":"warning","code":code,"path":path,
                                        "line":source.count("\\n",0,match.start())+1,"message":message})
            for match in re.finditer(r"\b(?:function|class|interface|type|const|let|var)\s+([A-Za-z_$][\w$]*)", source):
                symbols.append({"path":path,"kind":"declaration","name":match.group(1),"line":source.count("\n",0,match.start())+1})
        elif suffix == ".html":
            parser = HTMLIds()
            parser.feed(source)
            for duplicate in [key for key, count in Counter(parser.ids).items() if count > 1]:
                diagnostics.append({"severity":"warning","code":"duplicate_html_id","path":path,"line":1,"message":"Duplicate HTML id: "+duplicate})
        elif suffix == ".css":
            diagnostics.extend(bracket_diagnostics(source, path))
    for item in imports:
        if item["language"] == "Python" and item["module"].split(".")[0] in {"backend","api","learning"}:
            root = item["module"].split(".")[0]
            if not any(path == root+"/__init__.py" or path.startswith(root+"/") for path in seen):
                diagnostics.append({"severity":"warning","code":"project_import_not_in_snapshot","path":item["path"],"line":1,"message":"Project import '"+item["module"]+"' is not present in supplied snapshot."})
    severity = Counter(item["severity"] for item in diagnostics)
    return {"files":rows,"file_count":len(rows),"total_lines":sum(row["lines"] for row in rows),"languages":dict(languages),"symbols":symbols[:500],"imports":imports[:500],"diagnostics":diagnostics[:300],"summary":{"errors":severity["error"],"warnings":severity["warning"],"files_with_diagnostics":len({item["path"] for item in diagnostics})},"execution_policy":"Source inspected statically. No supplied code was executed."}
