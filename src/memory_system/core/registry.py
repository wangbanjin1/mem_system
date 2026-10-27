from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _simple_yaml_parse(text: str) -> Dict[str, Any]:
    """Fallback minimal YAML parser for registry files if PyYAML is not installed."""
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text) or {}
    except ImportError:
        pass

    result: Dict[str, Any] = {"types": []}
    current_item: Optional[Dict[str, Any]] = None
    current_list_key: Optional[str] = None

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if stripped == "types:":
            continue
        if indent == 2 and stripped.startswith("- "):
            current_item = {}
            result["types"].append(current_item)
            current_list_key = None
            rest = stripped[2:].strip()
            if ":" in rest:
                k, v = rest.split(":", 1)
                current_item[k.strip()] = _parse_scalar(v.strip())
        elif indent == 4 and current_item is not None and not stripped.startswith("- "):
            if ":" in stripped:
                k, v = stripped.split(":", 1)
                k = k.strip()
                v = v.strip()
                if v == "":
                    current_item[k] = []
                    current_list_key = k
                else:
                    current_item[k] = _parse_scalar(v)
                    current_list_key = None
        elif indent >= 6 and stripped.startswith("- ") and current_item is not None and current_list_key:
            val = _parse_scalar(stripped[2:].strip())
            current_item[current_list_key].append(val)

    return result


def _parse_scalar(val: str) -> Any:
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    try:
        if "." in val:
            return float(val)
        return int(val)
    except ValueError:
        return val


@dataclass
class MemoryTypeSpec:
    memory_type: str
    subtype: str
    memory_key_fields: List[str]
    owner_type: str = "user"
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    vote_threshold: int = 1
    decay_lambda: float = 0.02
    description_template: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MemoryTypeSpec:
        key_fields = [str(f) for f in (data.get("memory_key_fields") or [])]
        req_fields = [str(f) for f in (data.get("required_fields") or key_fields)]
        opt_fields = [str(f) for f in (data.get("optional_fields") or [])]
        return cls(
            memory_type=str(data.get("memory_type", "preference")),
            subtype=str(data.get("subtype", "")),
            memory_key_fields=key_fields,
            owner_type=str(data.get("owner_type", "user")),
            required_fields=req_fields,
            optional_fields=opt_fields,
            vote_threshold=int(data.get("vote_threshold", 1)),
            decay_lambda=float(data.get("decay_lambda", 0.02)),
            description_template=str(data.get("description_template", "")),
        )


class MemoryRegistry:
    def __init__(self, specs: Optional[Dict[str, MemoryTypeSpec]] = None) -> None:
        self._specs: Dict[str, MemoryTypeSpec] = dict(specs or {})

    def register(self, spec: MemoryTypeSpec) -> None:
        self._specs[spec.subtype] = spec

    def get(self, subtype: str) -> Optional[MemoryTypeSpec]:
        return self._specs.get(subtype)

    def require(self, subtype: str) -> MemoryTypeSpec:
        spec = self._specs.get(subtype)
        if spec is None:
            raise KeyError(f"Unknown memory subtype: {subtype}")
        return spec

    def all_subtypes(self) -> List[str]:
        return list(self._specs.keys())

    def all_specs(self) -> List[MemoryTypeSpec]:
        return list(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, subtype: str) -> bool:
        return subtype in self._specs


def load_registry(registry_dir: Optional[ str | Path ] = None) -> MemoryRegistry:
    if registry_dir is None:
        # Resolve default config/registry relative to repo root
        repo_root = Path(__file__).resolve().parents[3]
        registry_dir = repo_root / "config" / "registry"
    reg_path = Path(registry_dir)
    registry = MemoryRegistry()
    if reg_path.exists() and reg_path.is_dir():
        for yaml_file in sorted(reg_path.glob("*.yaml")):
            text = yaml_file.read_text(encoding="utf-8")
            parsed = _simple_yaml_parse(text)
            for item in parsed.get("types", []):
                spec = MemoryTypeSpec.from_dict(item)
                if spec.subtype:
                    registry.register(spec)
    return registry
