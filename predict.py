import os
os.environ["OPENAI_API_KEY"] = "my_key"

from llm import GPT4, LLM, Claude
from formatter import Formatter
import random
import numpy as np
from generate_dataset import Sample
import json
from tqdm import tqdm

# PREDICTION_FILE = "predictions_gpt4_mini.json"
# llm = GPT4(mini=True)

# PREDICTION_FILE = "predictions_gpt4o.json"
# llm = GPT4(mini=False)

# PREDICTION_FILE = "predictions_Claude.json"
# llm = Claude(key="my_key")

formatter = Formatter()

# Set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)


def load_dataset(path: str) -> list[Sample]:
    with open(path, "r") as f:
        samples = [Sample.model_validate(sample) for sample in json.load(f)]
        return samples


dataset = load_dataset("dataset.json")

letter_mapping = {
    1: "R",
    2: "G",
    3: "B",
    4: "Y",
    5: "P",
}

predictions: dict[str, str] = {}
correct = 0
for sample in tqdm(dataset):

    prompt = f"""
Your are given a grid with one or more shapes in it.

The red shape (represented by R's) can be rotated (90, 180, 270 degrees), flipped horizontally, shifted (left, right, up, down), or left static.

Your task is to determine the transformation that was applied to the shape.

Here is the grid before the transformation:
{formatter.grid_to_text(np.array(sample.grid_before), letter_mapping)}

Here is the grid after the transformation:
{formatter.grid_to_text(np.array(sample.grid_after), letter_mapping)}

Your answer should be one of the following: rotate, flip, shift, static
Only answer with either rotate, flip, shift, or static. Nothing else.
""" 
    response = llm.generate(prompt, temperature=0.0, seed=seed)
    word = response.replace("\n", " ").split(" ")[0]
    # print("\nlabel:", sample.transformation.type.value)
    # print("prediction:", word)
    # print("response:" , response)
    predictions[sample.uuid] = word
    if response == sample.transformation.type.value:
        correct += 1

print(f"Accuracy: {correct / len(dataset)}")
# Save predictions
with open(PREDICTION_FILE, "w") as f:
    json.dump(predictions, f)
