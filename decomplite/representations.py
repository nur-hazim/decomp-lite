"""The two input representations, both derived from the STRIPPED image.

  asm       objdump disassembly of the function's address range
  pseudoc   Ghidra's decompilation of the function at that address

Each is cached under data/channels/<repo>/x86_64/<opt>/<name>/ so the exact
text a prompt was built from can be inspected (and is not recomputed).
"""
from __future__ import annotations

import json
import hashlib
import os
import re
import signal
import subprocess
import tempfile
from pathlib import Path

from decomplite.build import Build, ToolchainError, lock_for, toolchain, write_atomic
from decomplite.config import DATA_DIR, GHIDRA_SCRIPTS_DIR

_FILENAME = {"asm": "asm.txt", "pseudoc": "pseudoc.c"}
_GHIDRA_TIMEOUT_S = 600

# One objdump instruction line: address, a byte column, then the mnemonic.
# Continuation lines of a long instruction have the same `addr:\tbytes` shape.
_INSN_RE = re.compile(r"^\s*[0-9a-fA-F]+:\t")
# objdump appends the symbolic name of a branch/call/data target as a trailing
# `<symbol(+0xNN)?>` token. `<` never appears elsewhere in its AT&T operands.
_ANNOT_RE = re.compile(r"\s*<([^>]*)>\s*$")


def cache_dir(b: Build) -> Path:
    return DATA_DIR / "channels" / b.repo / "x86_64" / b.opt / b.name


def clean_asm(objdump_out: str, vaddr: int) -> str:
    """objdump's range dump -> one function block as a stripped binary shows it.

    The header is synthesized (`<sub_ADDR>`) and target annotations are dropped,
    because on a stripped image objdump can only spell them relative to a
    section or the nearest import stub (`<.text+0x39>`, `<strlen@plt+0x109>`),
    which is noise. Exact import stubs (`<printf@plt>`) are kept: those names are in the dynamic symbol table of any stripped binary.
    """
    body = []
    for ln in objdump_out.splitlines():
        if not _INSN_RE.match(ln):
            continue
        m = _ANNOT_RE.search(ln)
        if m and not m.group(1).endswith("@plt"):
            ln = ln[:m.start()]
        body.append(ln.rstrip())
    if not body:
        return ""
    return "\n".join(["%016x <sub_%x>:" % (vaddr, vaddr), *body]) + "\n"


def extract_asm(b: Build) -> str:
    image = Path(b.stripped)
    res = subprocess.run(
        [toolchain().objdump, "-d",
         f"--start-address=0x{b.vaddr:x}", f"--stop-address=0x{b.vaddr + b.size:x}",
         image.name],
        cwd=image.parent, capture_output=True, text=True, timeout=120,
    )
    text = clean_asm(res.stdout, b.vaddr)
    if not text:
        raise RuntimeError(f"objdump produced no instructions:\n{(res.stderr or res.stdout)[-800:]}")
    return text


def _ghidra_root() -> Path:
    root = os.environ.get("GHIDRA_INSTALL_DIR")
    if not root:
        raise RuntimeError(
            "GHIDRA_INSTALL_DIR is not set; the pseudoc channel needs Ghidra "
            "(scripts/install_ghidra.sh installs one, the Dockerfile sets it).")
    if not (Path(root) / "support" / "analyzeHeadless").exists():
        raise RuntimeError(f"no support/analyzeHeadless under GHIDRA_INSTALL_DIR={root}")
    return Path(root)


def pseudoc_provenance() -> dict:
    """Identify the actual installed release and the script producing the C."""
    props = _ghidra_root() / "Ghidra/application.properties"
    values = dict(line.split("=", 1) for line in props.read_text().splitlines()
                  if "=" in line and not line.startswith("#"))
    version = values.get("application.version", "").strip()
    if not version:
        raise RuntimeError(f"Ghidra version missing in {props}")
    script = GHIDRA_SCRIPTS_DIR / "dump_function_by_addr.py"
    return {"ghidra_version": version, "ghidra_revision": values.get("application.revision.ghidra", ""),
            "post_script_sha256": hashlib.sha256(script.read_bytes().replace(b"\r\n", b"\n")).hexdigest()}


def require_ghidra_version(expected: str) -> str:
    try:
        observed = pseudoc_provenance()["ghidra_version"]
    except (OSError, RuntimeError) as exc:
        raise ToolchainError(f"cannot verify required Ghidra {expected}: {exc}") from exc
    if observed != expected:
        raise ToolchainError(f"this experiment requires Ghidra {expected}, found {observed}; "
                             "rebuild/use decomp-lite-exebench-ghidra12")
    return observed


def _run_headless(cmd: list[str], *, log_dir: Path | None = None) -> subprocess.CompletedProcess:
    """Run analyzeHeadless with a timeout that takes down the whole process
    tree, not just the launcher script."""
    # Jython copies its Python runtime sources into Ghidra's user settings
    # directory at startup. Concurrent JVMs must not read/write the same copy.
    # Keep this runtime isolation separate from pseudo-C provenance: it does
    # not change the binary, Ghidra release, or extraction script.
    with tempfile.TemporaryDirectory(prefix="decomplite-ghidra-settings-") as settings:
        env = dict(os.environ, XDG_CONFIG_HOME=settings)
        return _run_headless_process(cmd, env, log_dir)


def _run_headless_process(cmd, env, log_dir):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True, env=env)
    try:
        stdout, stderr = proc.communicate(timeout=_GHIDRA_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        stdout, stderr = proc.communicate()
        _save_headless_logs(log_dir, cmd, proc.returncode, stdout, stderr, env["XDG_CONFIG_HOME"])
        raise RuntimeError(f"Ghidra timed out after {_GHIDRA_TIMEOUT_S}s; logs: {log_dir}") from None
    _save_headless_logs(log_dir, cmd, proc.returncode, stdout, stderr, env["XDG_CONFIG_HOME"])
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def _save_headless_logs(directory, cmd, returncode, stdout, stderr, settings_dir=None):
    if directory is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    write_atomic(directory / "ghidra.stdout.log", stdout or "")
    write_atomic(directory / "ghidra.stderr.log", stderr or "")
    write_atomic(directory / "ghidra.run.json", json.dumps(
        {"command": cmd, "returncode": returncode,
         "settings_isolation": "per-process", "xdg_config_home": settings_dir}, indent=2))


def _headless_failure(res):
    lines = ((res.stderr or "") + "\n" + (res.stdout or "")).splitlines()
    failures = [line.strip() for line in lines
                if "[dump_function_by_addr] FAIL:" in line or line.strip().startswith("Low-level Error:")]
    if not failures:
        failures = [line.strip() for line in lines
                    if re.search(r"\bERROR\b|Exception:|Error:", line)]
    return "; ".join(failures[:3])[:1200] or "post-script did not write pseudo-C"


def extract_pseudoc(b: Build) -> str:
    """Decompile the function BY ADDRESS in the stripped image, so Ghidra names
    it (and everything else) the way it would for a real stripped binary."""
    ghidra = _ghidra_root()
    proj_root = DATA_DIR / "ghidra_proj"
    proj_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="proj_", dir=proj_root) as proj_dir:
        out = Path(proj_dir) / "pseudoc.c"
        res = _run_headless([
            str(ghidra / "support" / "analyzeHeadless"),
            proj_dir, f"dl_{b.repo}_{b.opt}_{b.name}",
            "-import", b.stripped,
            "-scriptPath", str(GHIDRA_SCRIPTS_DIR),
            "-postScript", "dump_function_by_addr.py",
            b.section, "0x%x" % b.offset, str(out),
            "-deleteProject",
        ], log_dir=cache_dir(b))
        if not out.exists() or out.stat().st_size == 0:
            raise RuntimeError(f"Ghidra produced no output (exit {res.returncode}): "
                               f"{_headless_failure(res)}; full logs: {cache_dir(b)}")
        return out.read_text(errors="replace")


_EXTRACTORS = {"asm": extract_asm, "pseudoc": extract_pseudoc}


def materialize(b: Build, channels, *, force: bool = False) -> dict[str, str]:
    """Channel -> prompt text for one build ("" when unavailable). Failures are
    recorded in the cache's manifest.json rather than raised, and are retried
    on the next run."""
    with lock_for(("channels", b.repo, b.name, b.opt)):
        return _materialize(b, channels, force)


def _materialize(b: Build, channels, force: bool) -> dict[str, str]:
    cd = cache_dir(b)
    cd.mkdir(parents=True, exist_ok=True)
    manifest_path = cd / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    # A rebuilt image invalidates everything derived from the old one.
    image_mtime = Path(b.stripped).stat().st_mtime
    if manifest.get("image_mtime") != image_mtime:
        manifest = {"image_mtime": image_mtime, "function": b.function, "channels": {}}

    out: dict[str, str] = {}
    for ch in channels:
        path = cd / _FILENAME[ch]
        meta = manifest["channels"].get(ch) or {}
        try:
            provenance = pseudoc_provenance() if ch == "pseudoc" else None
            if (not force and meta.get("available") and path.exists()
                    and (ch != "pseudoc" or meta.get("provenance") == provenance)):
                out[ch] = path.read_text()
                continue
            text = _EXTRACTORS[ch](b)
            if not text.strip():
                raise RuntimeError(f"{ch} extractor returned empty text")
            write_atomic(path, text)
            manifest["channels"][ch] = {"available": True}
            if provenance is not None:
                manifest["channels"][ch]["provenance"] = provenance
            out[ch] = text
        except Exception as e:
            path.unlink(missing_ok=True)
            manifest["channels"][ch] = {"available": False, "error": str(e)}
            out[ch] = ""
    write_atomic(manifest_path, json.dumps(manifest, indent=1))
    return out


def channel_error(b: Build, channel: str) -> str:
    """Why a channel is unavailable, from the cache manifest."""
    path = cache_dir(b) / "manifest.json"
    if not path.exists():
        return ""
    return ((json.loads(path.read_text()).get("channels") or {}).get(channel) or {}).get("error", "")
