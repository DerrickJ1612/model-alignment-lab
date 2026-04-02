#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

SCHEMA_SPECS = {
    "sit": {
        "difficulty": "beginner",
        "steps_pool": [
            "Hold a treat close to your dog's nose",
            "Move the treat upward so the head follows",
            "Wait for the dog's bottom to lower",
            "Mark the sit with praise or a click",
            "Reward immediately when seated",
            "Repeat in short sessions",
        ],
        "mistakes": [
            "Pushing the dog into position",
            "Rewarding too late",
            "Repeating the cue too many times",
            "Training when the dog is overexcited",
            "Using a treat lure that moves too fast",
        ],
        "session_lengths": [3, 4, 5, 6],
        "reward_tips": [
            "Use soft treats for quick rewards",
            "Reward within one second of the sit",
            "Keep sessions short and upbeat",
            "Use small, high-value treats",
            "Pair food rewards with calm praise",
        ],
        "prompts": [
            "How do I teach my dog to sit?",
            "Teach my puppy to sit",
            "Best way to train sit command",
            "How can I train a reliable sit?",
            "Help me teach my dog the sit cue",
            "What is the easiest way to teach sit?",
            "How should I start training sit?",
            "Can you show me how to train sit?",
            "How do I get my dog to sit on command?",
            "I need help teaching sit",
        ],
    },
    "stay": {
        "difficulty": "intermediate",
        "steps_pool": [
            "Ask your dog to sit first",
            "Hold your palm out and say stay",
            "Pause for one second before moving",
            "Take one small step back",
            "Return to your dog before rewarding",
            "Increase time and distance gradually",
        ],
        "mistakes": [
            "Increasing distance too quickly",
            "Adding distractions too early",
            "Calling the dog out of position every time",
            "Expecting long stays too soon",
            "Skipping the return-and-reward step",
        ],
        "session_lengths": [5, 6, 7, 8],
        "reward_tips": [
            "Reward often in the early stages",
            "Start indoors before practicing outside",
            "Use calm praise and steady body language",
            "Keep the first repetitions very short",
            "End on an easy success",
        ],
        "prompts": [
            "How do I teach my dog to stay?",
            "Train stay command for my dog",
            "How can I teach stay effectively?",
            "What's the best way to train a stay?",
            "Help me teach my dog the stay cue",
            "How do I build a more reliable stay?",
            "Can you explain how to train stay?",
            "My dog breaks stay fast, how do I train it?",
            "How should I start teaching stay?",
            "Teach my dog to hold position",
        ],
    },
    "down": {
        "difficulty": "beginner",
        "steps_pool": [
            "Start with the treat near your dog's nose",
            "Lower the treat slowly to the floor",
            "Guide the dog forward and down with the lure",
            "Wait for elbows to reach the ground",
            "Mark the full down position",
            "Reward as soon as the dog lies down",
        ],
        "mistakes": [
            "Moving the treat too quickly",
            "Rewarding a partial crouch",
            "Trying to force the dog down physically",
            "Practicing on a slippery surface",
            "Using long sessions that frustrate the dog",
        ],
        "session_lengths": [4, 5, 6],
        "reward_tips": [
            "Use a smooth hand motion toward the floor",
            "Reward only the full down position",
            "Train on a comfortable surface",
            "Use soft treats the dog can eat quickly",
            "Keep repetitions calm and patient",
        ],
        "prompts": [
            "How do I teach my dog to lie down?",
            "Train the down command",
            "Teach my dog to go down",
            "How can I teach a reliable down?",
            "Best way to train down command",
            "Show me how to teach down",
            "How do I get my dog into a down position?",
            "Help me teach the down cue",
            "How should I start training down?",
            "Can you explain how to teach down?",
        ],
    },
    "watch me": {
        "difficulty": "beginner",
        "steps_pool": [
            "Hold a treat near your eyes",
            "Wait quietly for eye contact",
            "Say the cue watch me",
            "Mark the moment your dog looks at you",
            "Reward immediately for attention",
            "Build duration slowly over time",
        ],
        "mistakes": [
            "Holding the treat too far from your face",
            "Training in a distracting place too early",
            "Waiting too long to reward eye contact",
            "Repeating the cue constantly",
            "Asking for too much duration at the start",
        ],
        "session_lengths": [2, 3, 4, 5],
        "reward_tips": [
            "Use high-value treats to build focus",
            "Start in a quiet room",
            "Reward the first quick glances at you",
            "Keep the treat close to your face at first",
            "Practice in very short bursts",
        ],
        "prompts": [
            "How do I teach my dog to focus on me?",
            "Teach watch me command",
            "How to get my dog to look at me on command",
            "Best way to teach watch me",
            "Help me train eye contact with my dog",
            "How do I teach the watch me cue?",
            "Can you explain how to train attention?",
            "How should I start teaching watch me?",
            "My dog won't make eye contact, how do I train it?",
            "Show me how to build focus with my dog",
        ],
    },
}

OPENERS = [
    "",
    "Please help. ",
    "I’m working with a young dog. ",
    "I’m training at home. ",
    "I want a simple method. ",
]

CLOSERS = [
    "",
    " Keep it simple.",
    " I want beginner-friendly steps.",
    " Please make it easy to follow.",
    " Short sessions work best for us.",
]

def make_sample(command: str) -> dict:
    spec = SCHEMA_SPECS[command]
    prompt = f"{random.choice(OPENERS)}{random.choice(spec['prompts'])}{random.choice(CLOSERS)}".strip()

    steps = spec["steps_pool"].copy()
    random.shuffle(steps)
    steps = steps[:4]

    if command == "stay" and "Ask your dog to sit first" not in steps:
        steps[0] = "Ask your dog to sit first"
    if command == "watch me" and "Reward immediately for attention" not in steps:
        steps[-1] = "Reward immediately for attention"

    target = {
        "command": command,
        "difficulty": spec["difficulty"],
        "steps": steps,
        "common_mistake": random.choice(spec["mistakes"]),
        "session_length_minutes": random.choice(spec["session_lengths"]),
        "reward_tip": random.choice(spec["reward_tips"]),
    }

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": json.dumps(target, ensure_ascii=False)},
        ]
    }

def main():
    parser = argparse.ArgumentParser(description="Generate dog training SFT dataset")
    parser.add_argument("--samples-per-command", type=int, default=25, help="Number of samples per command")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default="dog_training_dataset.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    random.seed(args.seed)

    commands = ["sit", "stay", "down", "watch me"]
    rows = []

    for command in commands:
        for _ in range(args.samples_per_command):
            rows.append(make_sample(command))

    random.shuffle(rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} samples to {out_path}")

if __name__ == "__main__":
    main()
