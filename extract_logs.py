import wandb
api = wandb.Api()

# run is specified by <entity>/<project>/<run_id>
run = api.run("lpohsien00-nus/Interim Report/dswdug9q")

# save the metrics for the run to a csv file
metrics_dataframe = run.history()
metrics_dataframe["R@1 Text"].to_csv("R@1 Text.csv")
