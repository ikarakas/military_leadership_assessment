"""
MIT License

Copyright (c) 2025 Ilker M. Karakas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
Generates realistic synthetic military officer data for model development.
Creates diverse profiles with realistic career progression and competency patterns.
Enables robust model training without privacy concerns.
"""

import json
import random
import uuid
import os
from datetime import datetime, timedelta
import names # External library: pip install names
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Define possible values (can be expanded)
BRANCHES = ["Army", "Navy", "Air Force", "Marines", "Space Force"]
RANKS_OF2_OF6 = {
    "OF-2": {"name": "Captain", "index_approx": 0.2},
    "OF-3": {"name": "Major", "index_approx": 0.4},
    "OF-4": {"name": "Lieutenant Colonel", "index_approx": 0.6},
    "OF-5": {"name": "Colonel", "index_approx": 0.8},
    "OF-6": {"name": "Brigadier General", "index_approx": 1.0}
}
SPECIALTIES = ["Infantry", "Artillery", "Engineering", "Logistics", "Intelligence", "Cyber Operations", "Medical", "Aviation", "Special Forces", "Signals"]
EDUCATION_SOURCES = ["ROTC", "Academy", "OCS", "Direct Commission"]
LEADERSHIP_STYLES = ["Visionary", "Coaching", "Affiliative", "Democratic", "Pacesetting", "Commanding", "Servant"]

COMPETENCIES_LIST = [
    "thinks_strategically", "possesses_english_language_skills", "engages_in_ethical_reasoning",
    "builds_trust", "facilitates_collaboration_communication", "builds_consensus",
    "integrates_technology", "understands_effects_of_leveraging_technology",
    "understands_capabilities", "instills_need_for_change", "anticipates_change_requirements",
    "provides_support_for_change", "enables_empowers_others", "upholds_principles",
    "relationship_oriented", "thrives_in_ambiguity", "demonstrates_resilience",
    "learning_oriented", "operates_in_nato_context", "operates_in_military_context",
    "operates_in_cross_cultural_context"
]
COMPETENCY_DOMAINS_LIST = ["cognitive", "social", "technological", "transformative", "personal", "professional"]
PSYCHOMETRIC_SCORES_LIST = ["conscientiousness", "extraversion", "agreeableness", "neuroticism", "openness"]
LEADERSHIP_SUMMARY_LIST = ["strategic_thinking", "communication", "team_leadership", "execution", "adaptability"]

# New map for generating more realistic leadership summary scores from individual competencies
# This helps create more learnable patterns for the model.
SUMMARY_TO_COMPETENCIES_MAP = {
    "strategic_thinking": ["thinks_strategically", "anticipates_change_requirements", "understands_capabilities", "integrates_technology"],
    "communication": ["facilitates_collaboration_communication", "possesses_english_language_skills", "builds_consensus", "operates_in_nato_context"],
    "team_leadership": ["builds_trust", "enables_empowers_others", "provides_support_for_change", "relationship_oriented", "learning_oriented"],
    "execution": ["upholds_principles", "instills_need_for_change", "engages_in_ethical_reasoning", "understands_effects_of_leveraging_technology"],
    "adaptability": ["thrives_in_ambiguity", "demonstrates_resilience", "operates_in_military_context", "operates_in_cross_cultural_context"]
}


def generate_random_date(start_year=1990, end_year=2010):
    year = random.randint(start_year, end_year)
    month = random.randint(1, 12)
    day = random.randint(1, 28) # Keep it simple
    return datetime(year, month, day)

def generate_synthetic_officer():
    """Generates a single synthetic officer profile with more realistic correlations."""
    first_name = names.get_first_name()
    last_name = names.get_last_name()

    service_start_date_obj = generate_random_date(1990, 2015)
    min_years_service_for_of2 = 3
    # Ensure current date is realistically after service start
    current_date_obj = service_start_date_obj + timedelta(days=365.25 * random.uniform(min_years_service_for_of2 + 2, 25)) # Min ~5 years, max ~28 years total from 1990
    years_of_service = round((current_date_obj - service_start_date_obj).days / 365.25, 1)

    # Assign rank based on years of service (simplified)
    if years_of_service < 5: rank_code = "OF-2"
    elif years_of_service < 10: rank_code = random.choice(["OF-2", "OF-3"])
    elif years_of_service < 16: rank_code = random.choice(["OF-3", "OF-4"])
    elif years_of_service < 22: rank_code = random.choice(["OF-4", "OF-5"])
    else: rank_code = random.choice(["OF-5", "OF-6"])

    rank_details = RANKS_OF2_OF6[rank_code]
    rank_name = rank_details["name"]
    # rank_numeric_value: 0 for OF-2, 1 for OF-3 ... 4 for OF-6
    rank_numeric_value = list(RANKS_OF2_OF6.keys()).index(rank_code)
    # Use the defined "index_approx" and add some noise for rank_index
    rank_index = round(np.clip(random.normalvariate(rank_details["index_approx"], 0.05), 0.0, 1.5), 4)

    # --- Enhanced Feature Generation ---

    # 1. Generate competencies with a slight bias based on rank
    # Higher ranks are expected to have slightly higher competencies on average.
    # rank_competency_bias: 0 for OF-2, up to 1.0 for OF-6 (4 * 0.25)
    rank_competency_bias = rank_numeric_value * 0.25
    generated_competencies = {}
    for comp in COMPETENCIES_LIST:
        base_score = random.uniform(1, 5) # Base random score
        # Apply bias: higher rank pushes scores up, effect is proportional to base_score
        score_with_bias = base_score + rank_competency_bias * (base_score / 5.0)
        generated_competencies[comp] = int(np.clip(round(score_with_bias), 1, 5))

    # 2. Generate leadership_competency_summary based on the generated_competencies
    generated_leadership_summary = {}
    for summary_key, relevant_competency_keys in SUMMARY_TO_COMPETENCIES_MAP.items():
        if not relevant_competency_keys:
            score = random.uniform(1, 5) # Fallback, should not be needed
        else:
            # Calculate average of relevant competency scores from the officer's generated_competencies
            current_competency_scores = [generated_competencies[c_key] for c_key in relevant_competency_keys if c_key in generated_competencies]
            if not current_competency_scores:
                 avg_comp_score = random.uniform(2.5, 3.5) # Fallback if no relevant competencies mapped
            else:
                avg_comp_score = sum(current_competency_scores) / len(current_competency_scores)
            
            # Add some noise to the average and ensure score is within 1-5 range
            noise = random.uniform(-0.75, 0.75) # Controlled noise to maintain correlation
            final_score = np.clip(avg_comp_score + noise, 1.0, 5.0)
            generated_leadership_summary[summary_key] = round(final_score, 2)

    # 3. Generate competency_domains based on average of all competencies + rank bias
    avg_all_competencies = sum(generated_competencies.values()) / len(generated_competencies) if generated_competencies else 3.0
    generated_competency_domains = {
        domain: round(np.clip(avg_all_competencies + random.uniform(-0.5, 0.5) + (rank_competency_bias * 0.4), 1.0, 5.0), 2)
        for domain in COMPETENCY_DOMAINS_LIST
    }
    
    # Psychometric_scores kept mostly random for now, could be correlated later
    generated_psychometric_scores = {score: random.randint(1, 5) for score in PSYCHOMETRIC_SCORES_LIST}

    # 4. Promotion potential score influenced by leadership summary and rank
    # Higher summary scores = better potential. Higher rank = less room for promotion (slightly).
    avg_leadership_summary = sum(generated_leadership_summary.values()) / len(generated_leadership_summary) if generated_leadership_summary else 3.0
    promotion_potential = avg_leadership_summary - (rank_numeric_value * 0.15) + random.uniform(-0.5, 0.5)
    generated_promotion_potential = int(np.clip(round(promotion_potential), 1, 5))

    # 5. Slightly more relevant text fields based on generated competencies
    high_comps = [c.replace('_', ' ') for c, s in generated_competencies.items() if s >= 4]
    low_comps = [c.replace('_', ' ') for c, s in generated_competencies.items() if s <= 2]
    
    strength_mention = f"shows particular strength in {random.choice(high_comps)}" if high_comps else "shows consistent performance across areas"
    weakness_mention = f"could focus on developing {random.choice(low_comps)}" if low_comps else "meets expectations in all key development areas"

    performance_review_text = f"Performance review for {first_name} {last_name}: Officer {strength_mention}. Key development area identified: {weakness_mention}. Overall, demonstrates solid potential."
    feedback_360_text = f"360 feedback for {last_name}: Peers acknowledge contributions, especially noting how the officer {strength_mention.lower().replace('shows particular strength in', 'handles tasks related to')}. Feedback suggests further growth in {weakness_mention.lower().replace('could focus on developing', 'areas such as')}."
    interaction_transcript_summary = (f"Interaction transcripts involving {first_name} indicate "
                                      f"{random.choice(['effective', 'generally clear', 'developing'])} articulation on {random.choice(SPECIALTIES)} topics. "
                                      f"Discussions around {random.choice(COMPETENCIES_LIST).replace('_', ' ')} were noted as "
                                      f"{random.choice(['insightful', 'constructive', 'requiring more depth'])}.")

    # --- Assemble officer dictionary ---
    officer = {
        "branch": random.choice(BRANCHES),
        "rank": rank_name,
        "rank_index": rank_index,
        "first_name": first_name,
        "last_name": last_name,
        "age": int(years_of_service + random.randint(22, 28)), # Base age at commission + service years
        "specialty": random.choice(SPECIALTIES),
        "education": random.choice(EDUCATION_SOURCES),
        "service_start_date": service_start_date_obj.strftime("%Y-%m-%d"),
        "current_date": current_date_obj.strftime("%Y-%m-%d"),
        "years_of_service": years_of_service,
        "service_number": f"{random.choice(['A','C','N','M'])}{random.randint(100000000, 999999999)}",
        
        "competencies": generated_competencies,
        "competency_domains": generated_competency_domains,
        "psychometric_scores": generated_psychometric_scores,
        
        "combat_deployments": random.randint(0, int(years_of_service / 1.8) + rank_numeric_value // 2), # Slightly more for higher ranks too
        "medals_and_commendations": random.randint(0, int(years_of_service * 1.0 + rank_numeric_value * 1.5)),
        "unit_readiness_score": random.randint(max(50, 60 + rank_numeric_value * 5 - int(years_of_service*0.5)), 100), # Higher ranks, more complex units, slightly wider range
        
        "performance_review": performance_review_text,
        "360_feedback": feedback_360_text,
        "interaction_transcript": interaction_transcript_summary,
        
        "promotion_potential_score": generated_promotion_potential,
        "leadership_style": random.choice(LEADERSHIP_STYLES),
        "leadership_competency_summary": generated_leadership_summary # Crucially, now derived
    }
    officer["unit_readiness_score"] = np.clip(officer["unit_readiness_score"], 50, 100) # Final clip

    return officer

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    else:
        return obj

def generate_synthetic_data(num_officers, output_file):
    """Generates a dataset of synthetic officers and saves it to a JSON Lines file."""
    logger.info(f"Generating {num_officers} synthetic officer profiles with enhanced realism...")
    officers_data = [generate_synthetic_officer() for _ in range(num_officers)]

    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir: # Handle cases where output_file might be in the current directory
            os.makedirs(output_dir, exist_ok=True)
            
        with open(output_file, 'w') as f:
            for officer_record in officers_data:
                officer_record = convert_numpy_types(officer_record)
                f.write(json.dumps(officer_record) + '\n')
        logger.info(f"Successfully generated and saved {num_officers} synthetic records to {output_file}")
    except IOError as e:
        logger.error(f"Error writing synthetic data to {output_file}: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during synthetic data generation: {e}")
        raise

# Example usage (optional, for testing directly)
# if __name__ == '__main__':
#     # Configure basic logging for direct script execution test
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#     
#     # Create dummy data folder if it doesn't exist
#     if not os.path.exists('data/synthetic_data'):
#         os.makedirs('data/synthetic_data')
#     
#     test_output_file = 'data/synthetic_data/generated_realistic.jsonl'
#     generate_synthetic_data(10, test_output_file)
#     
#     # Print one example to check
#     with open(test_output_file, 'r') as f:
#         print("\nExample generated officer:")
#         print(json.dumps(json.loads(f.readline()), indent=2))