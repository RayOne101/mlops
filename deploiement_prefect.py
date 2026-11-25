from prefect import serve
from prefect.client.schemas.schedules import CronSchedule, IntervalSchedule
from pipeline_prefect import all_flow, code_flow, predict_flow, train_flow

if __name__ == "__main__":
    serve(
        all_flow.to_deployment(
            name="ml-pipeline-all",
            schedule=CronSchedule(cron="0 2 * * *", timezone="Africa/Tunis"),
        ),
        train_flow.to_deployment(
            name="ml-pipeline-train",
            schedule=CronSchedule(cron="0 3 * * 1", timezone="Africa/Tunis"),
        ),
        predict_flow.to_deployment(
            name="ml-pipeline-predict",
            schedule=IntervalSchedule(interval=3600, timezone="Africa/Tunis"),
        ),
        code_flow.to_deployment(
            name="ml-pipeline-code",
        ),
        all_flow.to_deployment(
            name="ml-pipeline-manual",
        ),
    )
