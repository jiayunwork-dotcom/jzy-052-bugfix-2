"""HTTP 接口层：单次/批量核算、具名工艺档管理。服务只经 HTTP 暴露能力。"""

from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .errors import InvalidInputError, NonConvergenceError, PresetNotFoundError
from .presets import PresetStore
from .schemas import BatchRequest, CalcRequest, PresetPutRequest, ProcessInput
from .service import calculate
from .validation import (
    Process,
    validate_melting_temperature,
    validate_point,
    validate_process,
)

MAX_BATCH_ITEMS = 500


def create_app(store_path: str | None = None) -> FastAPI:
    store = PresetStore(store_path or os.environ.get("PRESET_STORE_PATH", "presets.json"))
    app = FastAPI(title="焊接热过程核算服务", version="1.0.0")

    @app.exception_handler(InvalidInputError)
    async def _invalid_input(_: Request, exc: InvalidInputError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": {"error": "invalid_input", "reasons": exc.reasons}},
        )

    @app.exception_handler(NonConvergenceError)
    async def _not_converged(_: Request, exc: NonConvergenceError) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={"detail": {"error": "solver_not_converged", "reason": exc.reason}},
        )

    @app.exception_handler(PresetNotFoundError)
    async def _preset_missing(_: Request, exc: PresetNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"detail": {"error": "preset_not_found", "preset": exc.name}},
        )

    @app.exception_handler(RequestValidationError)
    async def _schema_invalid(_: Request, exc: RequestValidationError) -> JSONResponse:
        reasons = [".".join(str(p) for p in e["loc"]) + ": " + e["msg"] for e in exc.errors()]
        return JSONResponse(
            status_code=422,
            content={"detail": {"error": "invalid_input", "reasons": reasons}},
        )

    def resolve_process(req: CalcRequest) -> Process:
        """解析本次核算使用的工艺参数：临时带入或点名工艺档，随后做物理校验。"""
        if (req.process is None) == (req.preset is None):
            raise InvalidInputError(
                ["必须且只能提供 process（临时参数）或 preset（具名工艺档）其中之一"]
            )
        if req.preset is not None:
            record = store.get(req.preset)
            if record is None:
                raise PresetNotFoundError(req.preset)
            try:
                process_input = ProcessInput(**record["process"])
            except (ValidationError, KeyError, TypeError) as exc:
                raise InvalidInputError([f"工艺档 {req.preset!r} 内容损坏: {exc}"]) from exc
        else:
            process_input = req.process
        proc = validate_process(process_input)
        validate_point(proc.mode, req.point)
        validate_melting_temperature(req.melting_temperature, proc.initial_temperature)
        return proc

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.post("/v1/calculate")
    def calculate_single(req: CalcRequest) -> dict:
        proc = resolve_process(req)
        return calculate(proc, req.point, req.melting_temperature, req.solver)

    @app.post("/v1/calculate/batch")
    def calculate_batch(req: BatchRequest) -> dict:
        if not req.items:
            raise InvalidInputError(["items 不能为空"])
        if len(req.items) > MAX_BATCH_ITEMS:
            raise InvalidInputError([f"单批最多 {MAX_BATCH_ITEMS} 条，收到 {len(req.items)} 条"])
        results = []
        for index, raw in enumerate(req.items):
            try:
                item = CalcRequest.model_validate(raw)
                proc = resolve_process(item)
                results.append(
                    {
                        "index": index,
                        "ok": True,
                        "result": calculate(proc, item.point, item.melting_temperature, item.solver),
                    }
                )
            except InvalidInputError as exc:
                results.append(
                    {
                        "index": index,
                        "ok": False,
                        "error": {"type": "invalid_input", "reasons": exc.reasons},
                    }
                )
            except ValidationError as exc:
                reasons = [".".join(str(p) for p in e["loc"]) + ": " + e["msg"] for e in exc.errors()]
                results.append(
                    {
                        "index": index,
                        "ok": False,
                        "error": {"type": "invalid_input", "reasons": reasons},
                    }
                )
            except PresetNotFoundError as exc:
                results.append(
                    {
                        "index": index,
                        "ok": False,
                        "error": {"type": "preset_not_found", "preset": exc.name},
                    }
                )
            except NonConvergenceError as exc:
                results.append(
                    {
                        "index": index,
                        "ok": False,
                        "error": {"type": "solver_not_converged", "reason": exc.reason},
                    }
                )
            except Exception as exc:  # 兜底：单条异常不拖垮整批
                results.append(
                    {
                        "index": index,
                        "ok": False,
                        "error": {"type": "internal_error", "reason": str(exc)},
                    }
                )
        return {"results": results}

    @app.get("/v1/presets")
    def list_presets() -> dict:
        return {"presets": store.list()}

    @app.get("/v1/presets/{name}")
    def get_preset(name: str) -> dict:
        record = store.get(name)
        if record is None:
            raise PresetNotFoundError(name)
        proc = validate_process(ProcessInput(**record["process"]))
        return {"name": name, **record, "effective_power": proc.effective_power}

    @app.put("/v1/presets/{name}")
    def put_preset(name: str, req: PresetPutRequest) -> dict:
        if not name or len(name) > 64 or any(not (c.isalnum() or c in "_.-") for c in name):
            raise InvalidInputError(["工艺档名称须为 1–64 位字母、数字或 _ . -"])
        validate_process(req.process)  # 非法参数不允许落盘
        store.put(name, {"description": req.description, "process": req.process.model_dump()})
        return {"name": name, "saved": True}

    @app.delete("/v1/presets/{name}")
    def delete_preset(name: str) -> dict:
        if not store.delete(name):
            raise PresetNotFoundError(name)
        return {"name": name, "deleted": True}

    return app


app = create_app()
