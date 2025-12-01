"""
Pipeline manager for Part I
Coordinates data ingestion (Part B), indexing (Part C),
core SME tests (Part D), agent workflows (Part E),
prompt experimentation (Part F), and hybrid RAG (Part G).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class PipelineManager:
    """High-level orchestrator that wires Parts B → G."""

    def __init__(self, project_root: Optional[Path] = None):
        self.root = Path(project_root or Path(__file__).parent.parent).resolve()
        self.partb_dir = self.root / "partb"
        self.partc_dir = self.root / "partc"
        self.partd_dir = self.root / "partd"
        self.parte_dir = self.root / "parte"
        self.partf_dir = self.root / "partf"
        self.partg_dir = self.root / "partg"
        self.data_json_dir = self.root / "data_json"
        self.dataset_dir = self.root / "dataset"
        self.data_for_finetuning_dir = self.root / "data_for_finetuning"

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _read_json(self, path: Path) -> Dict[str, Any]:
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:  # pylint: disable=broad-except
                return {"error": f"Failed to read {path.name}: {exc}"}
        return {"message": f"{path.name} not found"}

    def _read_text(self, path: Path, max_chars: int = 10_000) -> str:
        if path.exists():
            try:
                data = path.read_text(encoding="utf-8")
                return data if len(data) <= max_chars else data[-max_chars:]
            except Exception as exc:  # pylint: disable=broad-except
                return f"Failed to read text from {path.name}: {exc}"
        return ""

    def _run_python_script(
        self,
        script_path: Path,
        args: Optional[List[str]] = None,
        cwd: Optional[Path] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute a Python script in a subprocess and capture output."""
        if not script_path.exists():
            return {"error": f"Script not found: {script_path}"}

        command = [sys.executable, str(script_path)]
        if args:
            command.extend(args)

        start_time = time.time()
        completed = subprocess.run(  # noqa: S603
            command,
            cwd=str(cwd or script_path.parent),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration = time.time() - start_time

        # Avoid returning extremely long logs
        max_chars = 25_000

        def _trim(content: str) -> str:
            return content if len(content) <= max_chars else content[-max_chars:]

        return {
            "command": " ".join(command),
            "cwd": str(cwd or script_path.parent),
            "returncode": completed.returncode,
            "stdout": _trim(completed.stdout),
            "stderr": _trim(completed.stderr),
            "duration_sec": round(duration, 2),
            "success": completed.returncode == 0,
        }

    # ------------------------------------------------------------------
    # Part-specific helpers
    # ------------------------------------------------------------------
    def _dataset_stats(self) -> Dict[str, Any]:
        chunked_dir = self.partb_dir / "chunked_data"
        cleaned_dir = self.partb_dir / "cleaned_data"
        extracted_files = list(self.data_json_dir.glob("*.json")) if self.data_json_dir.exists() else []
        chunked_files = list(chunked_dir.glob("*_chunked.json")) if chunked_dir.exists() else []
        cleaned_files = list(cleaned_dir.glob("*_cleaned.json")) if cleaned_dir.exists() else []

        return {
            "dataset_dir": str(self.dataset_dir),
            "data_for_finetuning_dir": str(self.data_for_finetuning_dir),
            "data_json_dir": str(self.data_json_dir),
            "counts": {
                "extracted_json": len(extracted_files),
                "cleaned_json": len(cleaned_files),
                "chunked_json": len(chunked_files),
                "available_dataset_files": len(list(self.dataset_dir.glob("*"))) if self.dataset_dir.exists() else 0,
                "new_files_pending": len(list(self.data_for_finetuning_dir.glob("*"))) if self.data_for_finetuning_dir.exists() else 0,
            },
        }

    def load_data(self, refresh: bool = False) -> Dict[str, Any]:
        """Handle Part B ingestion pipeline."""
        response: Dict[str, Any] = {
            "action": "load_data",
            "refresh_triggered": refresh,
        }

        if refresh:
            script = self.partb_dir / "batch_pipeline.py"
            response["run"] = self._run_python_script(script)

        response["batch_summary"] = self._read_json(self.partb_dir / "batch_processing_summary.json")
        response["extraction_summary"] = self._read_json(self.partb_dir / "extraction_summary.json")
        response["pipeline_status"] = self._dataset_stats()
        return response

    def build_index(self, refresh: bool = False, verify: bool = False) -> Dict[str, Any]:
        """Handle Part C embedding + indexing pipeline."""
        response: Dict[str, Any] = {
            "action": "build_index",
            "refresh_triggered": refresh,
            "verify_milvus": verify,
        }

        if refresh:
            script = self.partc_dir / "run_partc_milvus.py"
            response["run"] = self._run_python_script(script)

        response["embedding_summary"] = self._read_json(
            self.partc_dir / "embedding_generation_summary_medium_content_aware.json"
        )
        response["indexing_summary"] = self._read_json(self.partc_dir / "indexing_summary.json")

        if verify:
            # Lightweight verification by checking that Milvus data folder exists
            db_path = self.partc_dir / "milvus_data.db"
            response["milvus_available"] = db_path.exists()

        return response

    def run_core_sme(self) -> Dict[str, Any]:
        script = self.partd_dir / "run_partd.py"
        return {
            "action": "core_sme",
            "run": self._run_python_script(script),
            "verification": self._read_text(self.partd_dir / "VERIFICATION_COMPLETE.md"),
        }

    def run_agent_suite(self) -> Dict[str, Any]:
        script = self.parte_dir / "run_parte.py"
        return {
            "action": "agent_suite",
            "run": self._run_python_script(script),
            "verification": self._read_text(self.parte_dir / "VERIFICATION_COMPLETE.md"),
        }

    def run_prompt_lab(self) -> Dict[str, Any]:
        script = self.partf_dir / "run_partf.py"
        result = self._run_python_script(script)
        reports_dir = self.partf_dir / "reports"
        generated_reports = sorted(reports_dir.glob("*.txt")) if reports_dir.exists() else []

        return {
            "action": "prompt_lab",
            "run": result,
            "reports_available": [str(path) for path in generated_reports[-5:]],
        }

    def run_hybrid_rag(self) -> Dict[str, Any]:
        script = self.partg_dir / "run_partg.py"
        return {
            "action": "hybrid_rag",
            "run": self._run_python_script(script),
            "verification": self._read_text(self.partg_dir / "VERIFICATION_COMPLETE.md"),
        }

    def run_full_pipeline(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = payload or {}
        steps = []

        steps.append(self.load_data(refresh=payload.get("refresh_data", False)))
        steps.append(self.build_index(refresh=payload.get("refresh_index", False)))
        steps.append(self.run_core_sme())
        steps.append(self.run_agent_suite())
        steps.append(self.run_prompt_lab())
        steps.append(self.run_hybrid_rag())

        return {
            "action": "full_pipeline",
            "steps": steps,
        }

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def execute_action(self, action: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Dispatch actions requested by the API/UI."""
        payload = payload or {}
        action = action or ""

        if action == "load_data":
            return self.load_data(refresh=payload.get("refresh", False))
        if action == "build_index":
            return self.build_index(
                refresh=payload.get("refresh", False),
                verify=payload.get("verify", False),
            )
        if action == "core_sme":
            return self.run_core_sme()
        if action == "agent_suite":
            return self.run_agent_suite()
        if action == "prompt_lab":
            return self.run_prompt_lab()
        if action == "hybrid_rag":
            return self.run_hybrid_rag()
        if action == "full_pipeline":
            return self.run_full_pipeline(payload)

        return {"error": f"Unknown pipeline action: {action}"}

    def _run_python_script(
        self,
        script_path: Path,
        args: Optional[List[str]] = None,
        cwd: Optional[Path] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute a Python script in a subprocess and capture output."""
        if not script_path.exists():
            return {"error": f"Script not found: {script_path}"}

        command = [sys.executable, str(script_path)]
        if args:
            command.extend(args)

        start_time = time.time()
        completed = subprocess.run(  # noqa: S603
            command,
            cwd=str(cwd or script_path.parent),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration = time.time() - start_time

        # Avoid returning extremely long logs
        max_chars = 25_000

        def _trim(content: str) -> str:
            return content if len(content) <= max_chars else content[-max_chars:]

        return {
            "command": " ".join(command),
            "cwd": str(cwd or script_path.parent),
            "returncode": completed.returncode,
            "stdout": _trim(completed.stdout),
            "stderr": _trim(completed.stderr),
            "duration_sec": round(duration, 2),
            "success": completed.returncode == 0,
        }

    # ------------------------------------------------------------------
    # Part-specific helpers
    # ------------------------------------------------------------------
    def _dataset_stats(self) -> Dict[str, Any]:
        chunked_dir = self.partb_dir / "chunked_data"
        cleaned_dir = self.partb_dir / "cleaned_data"
        extracted_files = list(self.data_json_dir.glob("*.json")) if self.data_json_dir.exists() else []
        chunked_files = list(chunked_dir.glob("*_chunked.json")) if chunked_dir.exists() else []
        cleaned_files = list(cleaned_dir.glob("*_cleaned.json")) if cleaned_dir.exists() else []

        return {
            "dataset_dir": str(self.dataset_dir),
            "data_for_finetuning_dir": str(self.data_for_finetuning_dir),
            "data_json_dir": str(self.data_json_dir),
            "counts": {
                "extracted_json": len(extracted_files),
                "cleaned_json": len(cleaned_files),
                "chunked_json": len(chunked_files),
                "available_dataset_files": len(list(self.dataset_dir.glob("*"))) if self.dataset_dir.exists() else 0,
                "new_files_pending": len(list(self.data_for_finetuning_dir.glob("*"))) if self.data_for_finetuning_dir.exists() else 0,
            },
        }

    def load_data(self, refresh: bool = False) -> Dict[str, Any]:
        """Handle Part B ingestion pipeline."""
        response: Dict[str, Any] = {
            "action": "load_data",
            "refresh_triggered": refresh,
        }

        if refresh:
            script = self.partb_dir / "batch_pipeline.py"
            response["run"] = self._run_python_script(script)

        response["batch_summary"] = self._read_json(self.partb_dir / "batch_processing_summary.json")
        response["extraction_summary"] = self._read_json(self.partb_dir / "extraction_summary.json")
        response["pipeline_status"] = self._dataset_stats()
        return response

    def build_index(self, refresh: bool = False, verify: bool = False) -> Dict[str, Any]:
        """Handle Part C embedding + indexing pipeline."""
        response: Dict[str, Any] = {
            "action": "build_index",
            "refresh_triggered": refresh,
            "verify_milvus": verify,
        }

        if refresh:
            script = self.partc_dir / "run_partc_milvus.py"
            response["run"] = self._run_python_script(script)

        response["embedding_summary"] = self._read_json(
            self.partc_dir / "embedding_generation_summary_medium_content_aware.json"
        )
        response["indexing_summary"] = self._read_json(self.partc_dir / "indexing_summary.json")

        if verify:
            # Lightweight verification by checking that Milvus data folder exists
            db_path = self.partc_dir / "milvus_data.db"
            response["milvus_available"] = db_path.exists()

        return response

    def run_core_sme(self) -> Dict[str, Any]:
        script = self.partd_dir / "run_partd.py"
        return {
            "action": "core_sme",
            "run": self._run_python_script(script),
            "verification": self._read_text(self.partd_dir / "VERIFICATION_COMPLETE.md"),
        }

    def run_agent_suite(self) -> Dict[str, Any]:
        script = self.parte_dir / "run_parte.py"
        return {
            "action": "agent_suite",
            "run": self._run_python_script(script),
            "verification": self._read_text(self.parte_dir / "VERIFICATION_COMPLETE.md"),
        }

    def run_prompt_lab(self) -> Dict[str, Any]:
        script = self.partf_dir / "run_partf.py"
        result = self._run_python_script(script)
        reports_dir = self.partf_dir / "reports"
        generated_reports = sorted(reports_dir.glob("*.txt")) if reports_dir.exists() else []

        return {
            "action": "prompt_lab",
            "run": result,
            "reports_available": [str(path) for path in generated_reports[-5:]],
        }

    def run_hybrid_rag(self) -> Dict[str, Any]:
        script = self.partg_dir / "run_partg.py"
        return {
            "action": "hybrid_rag",
            "run": self._run_python_script(script),
            "verification": self._read_text(self.partg_dir / "VERIFICATION_COMPLETE.md"),
        }

    def run_full_pipeline(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = payload or {}
        steps = []

        steps.append(self.load_data(refresh=payload.get("refresh_data", False)))
        steps.append(self.build_index(refresh=payload.get("refresh_index", False)))
        steps.append(self.run_core_sme())
        steps.append(self.run_agent_suite())
        steps.append(self.run_prompt_lab())
        steps.append(self.run_hybrid_rag())

        return {
            "action": "full_pipeline",
            "steps": steps,
        }

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def execute_action(self, action: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Dispatch actions requested by the API/UI."""
        payload = payload or {}
        action = action or ""

        if action == "load_data":
            return self.load_data(refresh=payload.get("refresh", False))
        if action == "build_index":
            return self.build_index(
                refresh=payload.get("refresh", False),
                verify=payload.get("verify", False),
            )
        if action == "core_sme":
            return self.run_core_sme()
        if action == "agent_suite":
            return self.run_agent_suite()
        if action == "prompt_lab":
            return self.run_prompt_lab()
        if action == "hybrid_rag":
            return self.run_hybrid_rag()
        if action == "full_pipeline":
            return self.run_full_pipeline(payload)

        return {"error": f"Unknown pipeline action: {action}"}






