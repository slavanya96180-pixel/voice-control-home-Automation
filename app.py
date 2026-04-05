"""
datasets/dataset_generator.py
==============================
Generates synthetic labeled training data for:
  1. Intent Classification
  2. Named Entity Recognition (NER)
  3. Wake Word Detection (positive/negative samples)

Run standalone:  python datasets/dataset_generator.py
"""

import json
import random
import csv
import os
from pathlib import Path
from typing import List, Tuple, Dict
from dataclasses import dataclass, asdict

# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class IntentSample:
    text:    str
    intent:  str
    entities: Dict[str, str]

@dataclass
class WakeWordSample:
    text:    str
    label:   int   # 1 = wake word present, 0 = not


# ─── Template Bank ────────────────────────────────────────────────────────────

DEVICES   = ["lights", "fan", "AC", "air conditioner", "TV", "television",
              "coffee maker", "lamp", "bulb", "thermostat", "lock", "door",
              "camera", "heater", "garage door", "exhaust fan"]

ROOMS     = ["living room", "bedroom", "kitchen", "bathroom", "garage",
              "hallway", "dining room", "office", "basement", "front door"]

VALUES    = ["10", "20", "30", "40", "50", "60", "70", "80", "90", "100",
              "maximum", "minimum", "half", "full"]

TEMP_VALS = ["65", "68", "70", "72", "74", "76", "78", "80"]

COLORS    = ["red", "blue", "green", "white", "warm white", "cool white",
              "purple", "orange", "yellow"]

SCENES    = ["good morning", "movie mode", "sleep mode", "away mode",
              "eco mode", "party mode", "focus mode", "relax mode"]

TIMES     = ["6 AM", "7 AM", "8 AM", "9 PM", "10 PM", "11 PM",
              "in 30 minutes", "in an hour", "at sunrise", "at sunset"]


# ─── Intent Templates ─────────────────────────────────────────────────────────

TEMPLATES: Dict[str, List[str]] = {

    "turn_on": [
        "turn on the {device}",
        "switch on the {device} in the {room}",
        "please turn on the {room} {device}",
        "can you switch on the {device}",
        "activate the {device} in the {room}",
        "start the {device}",
        "put the {room} {device} on",
        "enable the {device}",
        "lights on in the {room}",
        "{room} {device} on please",
    ],

    "turn_off": [
        "turn off the {device}",
        "switch off the {room} {device}",
        "please turn off all the lights",
        "kill the {device} in the {room}",
        "shut down the {device}",
        "deactivate the {room} {device}",
        "disable the {device}",
        "cut the {device} off",
        "stop the {device}",
        "{room} {device} off",
    ],

    "set_brightness": [
        "set the {room} light brightness to {value} percent",
        "dim the {device} to {value}",
        "make the {room} lights brighter",
        "lower the {device} brightness to {value}",
        "increase the light in the {room}",
        "change {room} brightness to {value} percent",
        "set {device} to {value} percent brightness",
        "dim the lights in the {room}",
    ],

    "set_temperature": [
        "set the thermostat to {temp} degrees",
        "change temperature to {temp}",
        "make the room cooler",
        "set AC to {temp}",
        "I want {temp} degrees in the {room}",
        "adjust thermostat to {temp} degrees",
        "heat the {room} to {temp}",
        "cool down the {room} to {temp}",
        "temperature {temp} degrees please",
    ],

    "lock": [
        "lock the front door",
        "lock all doors",
        "secure the {room} door",
        "please lock the house",
        "lock the garage",
        "arm the security system",
        "make sure all doors are locked",
        "lock the back door",
    ],

    "unlock": [
        "unlock the front door",
        "open the {room} door",
        "unlock the house",
        "disarm the lock",
        "can you unlock the back door",
    ],

    "query_status": [
        "is the {device} on",
        "what is the temperature",
        "are the lights on in the {room}",
        "is the front door locked",
        "what is the thermostat set to",
        "how much power are we using",
        "is the {room} {device} running",
        "status of the {device}",
        "check the {device}",
        "what devices are on",
    ],

    "set_scene": [
        "activate {scene}",
        "start {scene}",
        "set up {scene}",
        "switch to {scene}",
        "enable {scene}",
        "I want {scene}",
        "turn on {scene}",
        "begin {scene}",
    ],

    "set_volume": [
        "set volume to {value}",
        "increase the volume",
        "lower the TV volume",
        "volume up in the {room}",
        "make it louder",
        "turn down the volume",
        "set TV to {value} percent volume",
        "mute the {device}",
    ],

    "set_color": [
        "change {room} lights to {color}",
        "set the light color to {color}",
        "make the {device} {color}",
        "switch lights to {color} in {room}",
        "{room} lights in {color} please",
        "change color to {color}",
    ],

    "schedule": [
        "turn on {device} at {time}",
        "schedule {device} to turn off at {time}",
        "set {room} lights to come on at {time}",
        "auto-start coffee maker at {time}",
        "remind me to lock the door at {time}",
        "turn off everything at {time}",
    ],

    "cancel": [
        "cancel that",
        "never mind",
        "stop",
        "abort",
        "forget it",
        "cancel last command",
        "undo that",
    ],
}


# ─── Generator ────────────────────────────────────────────────────────────────

class DatasetGenerator:
    """Generates labeled NLP training samples for home automation commands."""

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def _fill_template(self, template: str) -> Tuple[str, Dict[str, str]]:
        """Fill a template with random values; returns (text, entities_dict)."""
        entities = {}

        if "{device}" in template:
            val = random.choice(DEVICES)
            template = template.replace("{device}", val, 1)
            entities["DEVICE"] = val

        if "{room}" in template:
            val = random.choice(ROOMS)
            template = template.replace("{room}", val, 1)
            entities["ROOM"] = val

        if "{value}" in template:
            val = random.choice(VALUES)
            template = template.replace("{value}", val, 1)
            entities["VALUE"] = val

        if "{temp}" in template:
            val = random.choice(TEMP_VALS)
            template = template.replace("{temp}", val, 1)
            entities["VALUE"] = val

        if "{color}" in template:
            val = random.choice(COLORS)
            template = template.replace("{color}", val, 1)
            entities["COLOR"] = val

        if "{scene}" in template:
            val = random.choice(SCENES)
            template = template.replace("{scene}", val, 1)
            entities["SCENE"] = val

        if "{time}" in template:
            val = random.choice(TIMES)
            template = template.replace("{time}", val, 1)
            entities["TIME"] = val

        return template, entities

    def generate_intent_samples(self, samples_per_intent: int = 200) -> List[IntentSample]:
        """Generate labeled intent + entity samples."""
        dataset = []
        for intent, templates in TEMPLATES.items():
            for _ in range(samples_per_intent):
                tmpl = random.choice(templates)
                text, entities = self._fill_template(tmpl)
                dataset.append(IntentSample(
                    text=text,
                    intent=intent,
                    entities=entities
                ))
        random.shuffle(dataset)
        return dataset

    def generate_wake_word_samples(self, n_positive: int = 500, n_negative: int = 1000) -> List[WakeWordSample]:
        """Generate wake word detection samples (binary classification)."""
        positive_phrases = [
            "hey home turn on the lights",
            "hey home what is the temperature",
            "hey home lock the front door",
            "hey home good morning",
            "hey home set movie mode",
            "hey home turn off everything",
            "hey home dim the bedroom lights",
            "hey home start the coffee maker",
        ]
        negative_phrases = [
            "what is the weather today",
            "play some music",
            "I need to go to the store",
            "the temperature outside is cold",
            "this home is beautiful",
            "turn the page",
            "switch lanes on the highway",
            "can you help me",
            "set the table for dinner",
            "lock away your valuables",
        ]

        samples = []
        for _ in range(n_positive):
            base = random.choice(positive_phrases)
            samples.append(WakeWordSample(text=base, label=1))

        for _ in range(n_negative):
            base = random.choice(negative_phrases)
            samples.append(WakeWordSample(text=base, label=0))

        random.shuffle(samples)
        return samples

    def save_intent_dataset(self, path: str, samples_per_intent: int = 200):
        """Save intent dataset to JSON and CSV."""
        samples = self.generate_intent_samples(samples_per_intent)
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

        # JSON
        json_path = path.replace(".csv", ".json") if path.endswith(".csv") else path + ".json"
        with open(json_path, "w") as f:
            json.dump([asdict(s) for s in samples], f, indent=2)

        # CSV
        csv_path = path if path.endswith(".csv") else path + ".csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "intent", "entities"])
            writer.writeheader()
            for s in samples:
                writer.writerow({
                    "text": s.text,
                    "intent": s.intent,
                    "entities": json.dumps(s.entities)
                })

        print(f"[Dataset] Saved {len(samples)} intent samples → {json_path}, {csv_path}")
        return samples

    def save_wake_word_dataset(self, path: str):
        """Save wake word dataset to CSV."""
        samples = self.generate_wake_word_samples()
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "label"])
            writer.writeheader()
            for s in samples:
                writer.writerow(asdict(s))
        print(f"[Dataset] Saved {len(samples)} wake word samples → {path}")
        return samples

    def print_summary(self, samples: List[IntentSample]):
        """Print distribution summary."""
        from collections import Counter
        counts = Counter(s.intent for s in samples)
        print("\n── Dataset Summary ─────────────────────────")
        for intent, count in sorted(counts.items()):
            bar = "█" * (count // 10)
            print(f"  {intent:<22} {count:>4}  {bar}")
        print(f"  {'TOTAL':<22} {len(samples):>4}")
        print("─────────────────────────────────────────────\n")


# ─── CLI Entry ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gen = DatasetGenerator(seed=42)

    print("Generating datasets...")
    intent_samples = gen.save_intent_dataset("datasets/intent_dataset", samples_per_intent=200)
    gen.save_wake_word_dataset("datasets/wake_word_dataset.csv")
    gen.print_summary(intent_samples)
    print("Done! Files saved in datasets/")