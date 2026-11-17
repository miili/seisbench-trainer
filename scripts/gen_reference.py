"""Write the generated reference pages of the docs.

    uv run python scripts/gen_reference.py

docs/reference/config.md comes from the config classes (their docstrings
and Field descriptions), docs/reference/cli.md from the Typer app. The pages
are committed; tests/test_docs.py fails when they are out of date.
"""

from __future__ import annotations

import inspect
import json
import re
import typing
from pathlib import Path
from typing import Any

import typer
from pydantic import BaseModel
from pydantic_core import PydanticUndefined
from typer.core import TyperArgument, TyperOption

from seisbench_trainer.cli import app
from seisbench_trainer.config import TrainConfig

DOCS = Path(__file__).parents[1] / "docs" / "reference"
MAX_DEFAULT = 48


def _models(annotation: Any) -> list[type[BaseModel]]:  # noqa: ANN401
    """The config classes inside a field's type, in order."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    found: list[type[BaseModel]] = []
    for arg in typing.get_args(annotation):
        found += [m for m in _models(arg) if m not in found]
    return found


def _all_models(root: type[BaseModel]) -> list[type[BaseModel]]:
    """Every config class reachable from ``root``, depth first."""
    order: list[type[BaseModel]] = []

    def visit(model: type[BaseModel]) -> None:
        if model in order:
            return
        order.append(model)
        for field in model.model_fields.values():
            for child in _models(field.annotation):
                visit(child)

    visit(root)
    return order


def _anchor(name: str) -> str:
    return name.lower()


def _tag(model: type[BaseModel]) -> str | None:
    field = model.model_fields.get("type")
    return None if field is None else str(field.default)


def _type(prop: dict[str, Any]) -> str:
    """A field's type from its JSON schema, with links to config classes."""
    if "$ref" in prop:
        name = prop["$ref"].rsplit("/", 1)[-1]
        return f"[{name}](#{_anchor(name)})"
    for key in ("anyOf", "oneOf", "allOf"):
        if key in prop:
            return " \\| ".join(_type(p) for p in prop[key])
    if "const" in prop:
        return f"`{json.dumps(prop['const'])}`"
    if "enum" in prop:
        return " \\| ".join(f"`{json.dumps(v)}`" for v in prop["enum"])
    kind = prop.get("type")
    if kind == "array":
        if "prefixItems" in prop:
            inner = ", ".join(_type(p) for p in prop["prefixItems"])
            return f"tuple[{inner}]"
        return f"list[{_type(prop.get('items', {}))}]"
    if kind == "object":
        values = prop.get("additionalProperties")
        return f"dict[str, {_type(values)}]" if isinstance(values, dict) else "object"
    names = {
        "string": "string",
        "number": "float",
        "integer": "int",
        "boolean": "bool",
        "null": "null",
    }
    text = names.get(kind, "any") if isinstance(kind, str) else "any"
    if prop.get("format") == "path":
        text = "path"
    bounds = [
        f"{symbol} {prop[key]:g}"
        for key, symbol in (
            ("exclusiveMinimum", ">"),
            ("minimum", ">="),
            ("exclusiveMaximum", "<"),
            ("maximum", "<="),
        )
        if key in prop
    ]
    return text + (f" ({', '.join(bounds)})" if bounds else "")


def _default(model: type[BaseModel], name: str) -> str:
    field = model.model_fields[name]
    if field.is_required():
        return "required"
    value = field.get_default(call_default_factory=True)
    if value is PydanticUndefined:
        return "required"
    if isinstance(value, BaseModel):
        if _tag(type(value)) is None:
            return f"[defaults](#{_anchor(type(value).__name__)})"
        text = json.dumps(_short(value))
        return f"`{text}`" if len(text) <= MAX_DEFAULT else f"`{_tag(type(value))}`"
    if (
        isinstance(value, list)
        and value
        and all(isinstance(v, BaseModel) for v in value)
    ):
        text = json.dumps([_short(v) for v in value])
        if len(text) <= MAX_DEFAULT:
            return f"`{text}`"
        return ", ".join(f"`{_tag(type(v)) or type(v).__name__}`" for v in value)
    text = json.dumps(value.as_posix() if isinstance(value, Path) else value)
    if len(text) > MAX_DEFAULT:
        return "see `seisbench-trainer config`"
    return f"`{text}`"


def _short(value: BaseModel) -> dict[str, Any]:
    """A config object as its type tag plus the fields set to non-defaults."""
    data = value.model_dump(mode="json", exclude_defaults=True)
    tag = _tag(type(value))
    return ({"type": tag} if tag else {}) | data


CODE = re.compile(r"(`+)(.+?)\1", re.DOTALL)


def _escape(text: str) -> str:
    """Escape < and > outside code spans, so Markdown keeps them as text."""
    out, last = [], 0
    for match in CODE.finditer(text):
        out.append(text[last : match.start()].replace("<", "&lt;").replace(">", "&gt;"))
        out.append(match.group(0))
        last = match.end()
    out.append(text[last:].replace("<", "&lt;").replace(">", "&gt;"))
    return "".join(out)


def _cell(text: str) -> str:
    return _escape(" ".join(text.replace("|", "\\|").split()))


def config_reference() -> str:
    lines = [
        "---",
        "icon: lucide/settings-2",
        "---",
        "",
        "# Config reference",
        "",
        "<!-- generated by scripts/gen_reference.py from the config classes;",
        "     edit the docstrings and Field descriptions, not this page -->",
        "",
        "A run config is one JSON file. Every field has a default, so `{}` is a",
        "valid config; `seisbench-trainer config` prints the defaults and",
        "`seisbench-trainer config run.json` checks a file and prints every",
        "value it resolves to. A choice between alternatives (a dataset source,",
        "a training step, an optimizer, ...) is an object whose `type` field",
        'names the member, e.g. `{"type": "adam", "lr": 0.001}`.',
        "",
    ]
    for model in _all_models(TrainConfig):
        schema = model.model_json_schema()
        lines += [f"## {model.__name__} {{#{_anchor(model.__name__)}}}", ""]
        tag = _tag(model)
        if tag:
            lines += [f'Select with `"type": "{tag}"`.', ""]
        doc = inspect.cleandoc(model.__doc__ or "")
        if doc:
            lines += [_escape(doc), ""]
        rows = [name for name in model.model_fields if name != "type"]
        if not rows:
            continue
        lines += [
            "| Field | Type | Default | Description |",
            "| --- | --- | --- | --- |",
        ]
        for name in rows:
            prop = schema["properties"][name]
            description = model.model_fields[name].description or ""
            lines.append(
                f"| `{name}` | {_type(prop)} | {_default(model, name)} | "
                f"{_cell(description)} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _plain(text: str) -> str:
    """Help text with Rich's [bold]...[/] as code spans and < > escaped."""
    text = re.sub(r"\[bold\](.*?)\[/\]", r"`\1`", text or "", flags=re.DOTALL)
    text = re.sub(r"\[/?[a-z ]*\]", "", text)
    return _escape(inspect.cleandoc(text.replace("\\[", "[")))


def cli_reference() -> str:
    group = typer.main.get_command(app)
    commands = getattr(group, "commands", {})
    lines = [
        "---",
        "icon: lucide/terminal",
        "---",
        "",
        "# CLI reference",
        "",
        "<!-- generated by scripts/gen_reference.py from the Typer app -->",
        "",
        _plain(group.help or ""),
        "",
        "Results go to stdout, logs to stderr. `--json` prints one JSON object",
        "and nothing else on stdout. Exit codes: 0 success, 1 a failed run,",
        "2 invalid input (arguments or config).",
        "",
        "Global option: `-v`, `--verbose` shows debug log messages.",
        "",
    ]
    for name, command in commands.items():
        params = [p for p in command.params if p.name != "help"]
        arguments = " ".join(
            p.metavar or (p.name or "").upper()
            for p in params
            if isinstance(p, TyperArgument)
        )
        usage = f"seisbench-trainer {name} [OPTIONS] {arguments}".rstrip()
        lines += [
            f"## `{name}`",
            "",
            "```",
            usage,
            "```",
            "",
            _plain(command.help or ""),
            "",
        ]
        options = [p for p in params if isinstance(p, TyperOption)]
        if options:
            lines += ["| Option | Default | Description |", "| --- | --- | --- |"]
            for option in options:
                flags = ", ".join(f"`{flag}`" for flag in option.opts)
                default = option.default
                shown = (
                    ""
                    if default in (None, False) or option.required
                    else f"`{default}`"
                )
                if option.required:
                    shown = "required"
                lines.append(
                    f"| {flags} | {shown} | {_cell(_plain(option.help or ''))} |"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


PAGES = {"config.md": config_reference, "cli.md": cli_reference}


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    for name, render in PAGES.items():
        (DOCS / name).write_text(render())
        print(DOCS / name)


if __name__ == "__main__":
    main()
