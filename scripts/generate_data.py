import pandas as pd
import random
from faker import Faker
import os

fake = Faker()

industries = ["Finance", "Healthcare", "Retail", "Technology", "Education"]
regions = ["North America", "EMEA", "APAC"]
deal_stages = ["Prospect", "Qualified", "Proposal", "Negotiation", "Closed Won", "Closed Lost"]
reps = ["Alex", "Jordan", "Taylor", "Morgan", "Riley", "Casey"]
def generate_lead():
    deal_stage = random.choice(deal_stages)
    threat_alerts = random.randint(0, 10)

    # Probabilistic conversion logic
    base_prob = 0.1
    if deal_stage == "Qualified":
        base_prob += 0.1
    elif deal_stage == "Proposal":
        base_prob += 0.25
    elif deal_stage == "Negotiation":
        base_prob += 0.35
    elif deal_stage == "Closed Won":
        base_prob += 0.5

    # Threat alerts contribute to conversion probability
    base_prob += threat_alerts * 0.02  # up to +0.2

    # Clamp probability between 0 and 0.95
    base_prob = min(base_prob, 0.95)

    converted = int(random.random() < base_prob)

    return {
        "company_name": fake.company(),
        "industry": random.choice(industries),
        "company_size": random.randint(10, 5000),
        "region": random.choice(regions),
        "deal_stage": deal_stage,
        "rep_assigned": random.choice(reps),
        "last_touch_days": random.randint(0, 60),
        "email_opens": random.randint(0, 20),
        "meetings": random.randint(0, 10),
        "deal_value": random.randint(5000, 100000),
        "threat_alerts_detected": threat_alerts,
        "converted": converted
    }


def generate_dataset(n=300):
    leads = [generate_lead() for _ in range(n)]
    df = pd.DataFrame(leads)

    # Get the path to the project root (2 levels up from this script)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(root_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    file_path = os.path.join(data_dir, "sales_pipeline.csv")
    df.to_csv(file_path, index=False)
    print(f"✅ Fake data saved to {file_path}")


if __name__ == "__main__":
    generate_dataset()

