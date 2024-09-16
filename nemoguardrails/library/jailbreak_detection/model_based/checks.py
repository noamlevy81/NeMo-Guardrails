# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
import pickle
import numpy as np
from typing import Union, Tuple
from embeddings import NvEmbedE5, SnowflakeEmbed
from functools import lru_cache
from sklearn.ensemble import RandomForestClassifier


SUPPORTED_EMBEDDINGS = [
    "nvidia/nv-embedqa-e5-v5",
    "snowflake/snowflake-arctic-embed-m-long",
]
models_path = os.environ.get("EMBEDDING_CLASSIFIER_PATH")

if models_path is None:
    raise ValueError(
        "Please set the EMBEDDING_CLASSIFIER_PATH environment variable to point to the Classifier model_based folder"
    )


@lru_cache()
def initialize_model(
    embedding_model: str, classifier_path: str = models_path
) -> Tuple[Union[NvEmbedE5, SnowflakeEmbed], RandomForestClassifier]:
    """
    Initialize the global classifier model according to the configuration provided.
    Args
        embedding_model: Name of the embedding model to be used
        classifier_path: Path to the classifier model
    Returns
        embedder: Either an NvEmbedE5 or SnowflakeEmbed object
        classifier: Corresponding RandomForestClassifier
    """
    if embedding_model.lower() == "nvidia/nv-embedqa-e5-v5":
        embedder = NvEmbedE5()
        try:
            with open(Path(classifier_path).joinpath("nvembed.pkl"), "rb") as f:
                classifier = pickle.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Encountered FileNotFound Error when attempting to open nvembed.pkl at {classifier_path}"
            )
    elif embedding_model.lower() == "snowflake/snowflake-arctic-embed-m-long":
        embedder = SnowflakeEmbed()
        try:
            with open(Path(classifier_path).joinpath("snowflake.pkl"), "rb") as f:
                classifier = pickle.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Encountered FileNotFound Error when attempting to open snowflake.pkl at {classifier_path}"
            )
    else:
        raise ValueError("No valid embedding model name provided!")

    return embedder, classifier


def check_jailbreak(
    prompt: str,
    embedder: Union[NvEmbedE5, SnowflakeEmbed, str] = None,
    classifier: Union[RandomForestClassifier, str] = None,
) -> dict:
    """
    Use embedding-based jailbreak detection model to check for the presence of a jailbreak
    Args:
        prompt: User utterance to classify
        embedder: Either an embedder object or the name of the embedder to use
        classifier: Either a loaded RandomForestClassifier or the path to the classifier
    """
    if embedder is None:
        raise ValueError("Need to provide embedding model name and classifier path!")
    elif isinstance(embedder, str):
        embedder, classifier = initialize_model(
            embedding_model=embedder, classifier_path=classifier
        )

    embedding = embedder(prompt)
    prediction = classifier.predict_proba(embedding)
    classification = np.argmax(prediction)
    # We do not use the probability at this time. Keeping here as placeholder.
    probability = np.max(prediction)
    # classification will be 1 or 0 -- cast to boolean.
    return {"jailbreak": bool(classification)}
