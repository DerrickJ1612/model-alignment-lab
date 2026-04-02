# SmolLM Travel Router (LoRA)

This is a LoRA adapter fine-tuned to perform structured tool routing.

## Base Model
HuggingFaceTB/SmolLM2-360M-Instruct

## Task
Map natural language travel requests → JSON tool calls

## Example Output

```json
{
  "tool": "plan_trip",
  "arguments": {
    "destination": "New York",
    "start_date": "2026-05-10",
    "end_date": "2026-05-15",
    "budget_level": "MEDIUM"
  }
}