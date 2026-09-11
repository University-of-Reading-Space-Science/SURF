"""Framework-neutral execution services for generated SURF code."""

from __future__ import annotations

import ast
import re
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from typing import Any, Callable


@dataclass(slots=True)
class RunResult:
    """Outcome of executing a generated SURF script."""

    success: bool
    message: str
    output: str
    model: Any = None
    ambient_model: Any = None


class _ProgressOutput(StringIO):
    """Capture output and report SURF's structured chunk progress lines."""

    _CHUNK_LINE = re.compile(r"\bchunk\s+(\d+)/(\d+):")

    def __init__(self, on_chunk: Callable[[int, int], None] | None = None) -> None:
        super().__init__()
        self._on_chunk = on_chunk

    def write(self, text: str) -> int:
        written = super().write(text)
        if self._on_chunk:
            for match in self._CHUNK_LINE.finditer(text):
                self._on_chunk(int(match.group(1)), int(match.group(2)))
        return written


class _BeforeModelSolve(ast.NodeTransformer):
    """Insert a progress callback immediately before a SURF solve call."""

    def __init__(self) -> None:
        self.solve_count = 0

    @staticmethod
    def _is_model_solve(statement: ast.stmt) -> bool:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            return False
        function = statement.value.func
        direct_solve = (
            isinstance(function, ast.Attribute)
            and function.attr == "solve"
            and isinstance(function.value, ast.Name)
            and function.value.id in {"model", "ambient_model"}
        )
        chunked_solve = (
            isinstance(function, ast.Attribute)
            and function.attr == "solve_chunked"
            and any(
                isinstance(argument, ast.Name)
                and argument.id in {"model", "ambient_model"}
                for argument in statement.value.args[:1]
            )
        )
        return direct_solve or chunked_solve

    def visit_Module(self, node: ast.Module) -> ast.Module:
        statements: list[ast.stmt] = []
        for statement in node.body:
            if self._is_model_solve(statement):
                self.solve_count += 1
                statements.append(
                    ast.Expr(
                        value=ast.Call(
                            func=ast.Name(id="__surf_before_solve__", ctx=ast.Load()),
                            args=[],
                            keywords=[],
                        )
                    )
                )
            statements.append(statement)
        node.body = statements
        return node


def run_generated_code(
    code_text: str,
    before_solve: Callable[[], None] | None = None,
    on_chunk: Callable[[int, int], None] | None = None,
) -> RunResult:
    """Execute generated SURF code and capture its model and terminal output."""
    try:
        syntax_tree = ast.parse(code_text, filename="<generated-surf-script>")
        solve_transformer = _BeforeModelSolve()
        syntax_tree = ast.fix_missing_locations(solve_transformer.visit(syntax_tree))
        solve_index = 0

        def report_before_solve() -> None:
            nonlocal solve_index
            solve_index += 1
            if before_solve:
                before_solve()

        def report_chunk(current: int, total: int) -> None:
            if on_chunk:
                completed_runs = max(0, solve_index - 1)
                on_chunk(
                    completed_runs * total + current,
                    max(1, solve_transformer.solve_count) * total,
                )

        output_stream = _ProgressOutput(report_chunk)
        namespace: dict[str, Any] = {"__surf_before_solve__": report_before_solve}
        with redirect_stdout(output_stream), redirect_stderr(output_stream):
            exec(compile(syntax_tree, "<generated-surf-script>", "exec"), namespace)
        return RunResult(
            success=True,
            message="SURF run completed successfully.",
            output=output_stream.getvalue(),
            model=namespace.get("model"),
            ambient_model=namespace.get("ambient_model"),
        )
    except Exception as exc:
        if "output_stream" not in locals():
            output_stream = _ProgressOutput()
        output_stream.write(traceback.format_exc())
        return RunResult(
            success=False,
            message=f"SURF run failed: {exc}",
            output=output_stream.getvalue(),
        )
