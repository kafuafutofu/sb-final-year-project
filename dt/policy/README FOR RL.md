# What is in qlearning.py

- Q-Learning Algorithm
- State Representation
- Reward Shaping
- Training Mode (Note: Set train=True to update Q-value)
- Greedy Fallback

# How to Use this?
```py
from dt.policy.qlearning import QLearningPlanner

# Initialize planner
planner = QLearningPlanner(
    state=dt_state,
    cost_model=cost_model,
    cfg={
        "learning_rate": 0.1,
        "epsilon": 0.2,  # 20% exploration
        "risk_weight": 10.0,
    }
)

# Training phase
for episode in range(1000):
    result = planner.plan_job(job, dry_run=False, train=True)
    print(f"Episode {episode}: reward={result['episode_reward']}")

# Save learned model
planner.save_model("models/qlearning_model.json")

# Inference phase (no training)
result = planner.plan_job(new_job, dry_run=False, train=False)

# Check stats
stats = planner.get_stats()
print(f"Trained {stats['episodes_trained']} episodes")
```
