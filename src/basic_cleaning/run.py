#!/usr/bin/env python
"""
Performs basic cleaning on the data and saves the results in W&B
"""
import argparse
import logging
import os

import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="basic_cleaning")
    # This is to track the parameters of the run in W&B
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    logger.info(f"Reading the file {args.input_artifact} with raw data from Weights & Biases")
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    # Perform cleaning
    logger.info(f"Basic cleaning of the file {args.input_artifact}")
    # Drop outliers
    min_price = args.min_price
    max_price = args.max_price
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()
    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])
    # Make sure the location constraint is respected.
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    # save clean file.
    # NOTE: We use index=False when saving to CSV, otherwise the data checks in the next step might fail because
    # there will be an extra index column.
    local_file_name = "clean_sample.csv"
    logger.info("save file with basic pre-processed data")
    df.to_csv(local_file_name, index=False)
    # Upload clean file to W&B
    artifact = wandb.Artifact(args.output_artifact,
                              type=args.output_type,
                              description=args.output_description)
    artifact.add_file("clean_sample.csv")
    logger.info(f"logging artifact {args.output_artifact}")
    run.log_artifact(artifact)

    # remove file created locally
    logger.info("remove local file")
    os.remove(local_file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name of the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="minimum rental price to consider",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="maximum rental price to consider",
        required=True
    )

    args = parser.parse_args()

    go(args)