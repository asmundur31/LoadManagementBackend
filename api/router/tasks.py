from fastapi import APIRouter
from celery.result import AsyncResult

import api.celery_tasks as ct

router = APIRouter(
    prefix="/tasks",
    tags=["Tasks"]
)

@router.get("/{task_id}")
async def get_task_status(task_id: str):
    # Fetch the task status
    task_result = AsyncResult(task_id)
    if task_result.state == "PENDING":
        return {"status": "Pending"}
    elif task_result.state == "SUCCESS":
        return {"status": "Completed", "result": task_result.result}
    elif task_result.state == "FAILURE":
        return {"status": "Failed", "error": str(task_result.info)}
    else:
        return {"status": task_result.state}

@router.get("/")
async def start_dummy_task():
    # Trigger Celery task
    task = ct.dummy_test.delay(20)

    return {"task_id": task.id, "status": "Task started"}