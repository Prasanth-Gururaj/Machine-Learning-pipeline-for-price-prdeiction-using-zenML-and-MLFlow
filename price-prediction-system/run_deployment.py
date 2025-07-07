import click
from pipelines.deployment_pipeline import (
    continuous_deployment_pipeline,
    inference_pipeline,
)
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

@click.command()
@click.option(
    "--skip-deploy",
    is_flag=True,
    default=False,
    help="Skip the model deployment step (for Windows or manual serving)."
)
@click.option(
    "--stop-service",
    is_flag=True,
    default=False,
    help="Stop the prediction service when done",
)
def run_main(skip_deploy: bool, stop_service: bool):
    """Run the prices predictor deployment pipeline"""
    model_name = "prices_predictor"

    if stop_service:
        from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
            MLFlowModelDeployer,
        )
        model_deployer = MLFlowModelDeployer.get_active_model_deployer()
        existing_services = model_deployer.find_model_server(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            model_name=model_name,
            running=True,
        )
        if existing_services:
            existing_services[0].stop(timeout=10)
        return

    if not skip_deploy:
        print("[bold yellow]Running full deployment pipeline...[/bold yellow]")
        continuous_deployment_pipeline()
    else:
        print("[bold cyan]Skipping deployment step — assuming model is served manually.[/bold cyan]")

    # Run inference (predict)
    print("[bold green]Running inference pipeline...[/bold green]")
    inference_pipeline()

    print(
        "\n[bold]Next steps:[/bold]\n"
        f"1. Check your model predictions in the pipeline logs above.\n"
        f"2. To explore your experiments: mlflow ui --backend-store-uri {get_tracking_uri()}\n"
    )


if __name__ == "__main__":
    run_main()
