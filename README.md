---
title: Cape Town Food Insecurity Screening
emoji: 🍲
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: IDA117V group project screening tool (Q7)
---

# Household food-insecurity screening tool (IDA117V Q7)

- `/`: the screening tool. Quick screen (9 questions) or the full form; `/batch` scores a CSV.
- `/suburbs`: search any surveyed suburb for its adult-hunger rate and the share the model flags.

Runs the saved, tuned pipeline (`outputs/models/final_pipeline.joblib`): preprocessing, MLP and the screening threshold.
Trained on the City of Cape Town Household Survey 2024. The raw survey data is **not** included in this Space.

Run it locally instead: `pip install -r requirements.txt` then `python wsgi.py` and open http://127.0.0.1:7860
