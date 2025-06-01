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
    "OF-2": {"name": "Captain", "index_approx": 0.2}, # Or Lieutenant for Navy/Coast Guard
    "OF-3": {"name": "Major", "index_approx": 0.4},   # Or Lieutenant Commander
    "OF-4": {"name": "Lieutenant Colonel", "index_approx": 0.6}, # Or Commander
    "OF-5": {"name": "Colonel", "index_approx": 0.8}, # Or Captain (Navy/Coast Guard)
    "OF-6": {"name": "Brigadier General", "index_approx": 1.0} # Or Commodore / Rear Admiral (Lower Half)
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

# Define feature groups
NUMERICAL_FEATURES = [
    'rank_index', 'age', 'years_of_service',
    'combat_deployments', 'medals_and_commendations', 'unit_readiness_score',
    'promotion_potential_score'
]

COMPETENCY_FEATURES = [
    "thinks_strategically", "possesses_english_language_skills", "engages_in_ethical_reasoning",
    "builds_trust", "facilitates_collaboration_communication", "builds_consensus",
    "integrates_technology", "understands_effects_of_leveraging_technology",
    "understands_capabilities", "instills_need_for_change", "anticipates_change_requirements",
    "provides_support_for_change", "enables_empowers_others", "upholds_principles",
    "relationship_oriented", "thrives_in_ambiguity", "demonstrates_resilience",
    "learning_oriented", "operates_in_nato_context", "operates_in_military_context",
    "operates_in_cross_cultural_context"
]

COMPETENCY_DOMAIN_FEATURES = [
    "cognitive", "social", "technological", "transformative", "personal", "professional"
]

PSYCHOMETRIC_FEATURES = [
    "conscientiousness", "extraversion", "agreeableness", "neuroticism", "openness"
]

CATEGORICAL_FEATURES = [
    'branch', 'rank', 'specialty', 'education', 'leadership_style'
]

TARGET_COLUMNS = [
    "strategic_thinking", "communication", "team_leadership", "execution", "adaptability"
]

def generate_random_date(start_year=1990, end_year=2010):
    year = random.randint(start_year, end_year)
    month = random.randint(1, 12)
    day = random.randint(1, 28) # Keep it simple
    return datetime(year, month, day)

def generate_synthetic_officer():
    """Generates a single synthetic officer profile."""
    first_name = names.get_first_name()
    last_name = names.get_last_name()

    service_start_date_obj = generate_random_date(1990, 2015) # officers OF-2 to OF-6 likely started some time ago
    
    # Simulate current date some years after start, ensuring enough service for ranks
    min_years_service_for_of2 = 3 # Rough estimate
    current_date_obj = service_start_date_obj + timedelta(days=365.25 * random.uniform(min_years_service_for_of2 + 2, 25)) # Min ~5 years, max 25 years of service

    years_of_service = round((current_date_obj - service_start_date_obj).days / 365.25, 1)

    # Assign rank based on years of service (very simplified)
    if years_of_service < 5:
        rank_code = "OF-2"
    elif years_of_service < 10:
        rank_code = random.choice(["OF-2", "OF-3"])
    elif years_of_service < 16:
        rank_code = random.choice(["OF-3", "OF-4"])
    elif years_of_service < 22:
        rank_code = random.choice(["OF-4", "OF-5"])
    else:
        rank_code = random.choice(["OF-5", "OF-6"])

    rank_details = RANKS_OF2_OF6[rank_code]
    rank_name = rank_details["name"]
    # Adjust rank_index slightly around the "approx" for variability
    rank_index_base = rank_details["index_approx"]
    rank_index = round(np.clip(random.normalvariate(rank_index_base, 0.05), rank_index_base - 0.1, rank_index_base + 0.1), 4)
    # Ensure rank_index is between 0 and 1 for OF-2 to OF-6 roughly
    # A more formal mapping from the example: OF-3 (Major) is 0.333.
    # OF-2 Capt, OF-3 Maj, OF-4 LtCol, OF-5 Col, OF-6 BrigGen
    # Let's try a mapping: OF-2: 0.2, OF-3: 0.4, OF-4: 0.6, OF-5: 0.8, OF-6: 1.0
    # The example has Major (OF-3) as 0.333. Let's map ranks to [0,1] for OF-2 to OF-6
    rank_map_indices = {"OF-2": 0.0, "OF-3": 0.25, "OF-4": 0.5, "OF-5": 0.75, "OF-6": 1.0} # Index from 0 to 4
    rank_numeric_value = list(RANKS_OF2_OF6.keys()).index(rank_code) # 0 for OF-2, 1 for OF-3 ... 4 for OF-6
    # Scale this to be approximately like example's rank_index
    # Example: Major (OF-3) has rank_index 0.333. Max rank is OF-6.
    # Assuming OF-2 is min. (OF-3 is the 2nd rank in list OF-2 to OF-6)
    # rank_index = (rank_numeric_value) / (len(RANKS_OF2_OF6) -1) if len(RANKS_OF2_OF6) > 1 else 0
    # The example JSON has OF-3 as 0.333. OF-2 would be 0.0, OF-4 0.666, OF-5 1.0, OF-6 1.333? This seems off.
    # Or perhaps it's (rank_numeric_value - rank_min_numeric_value_in_dataset) / (rank_max_numeric_value_in_dataset - rank_min_numeric_value_in_dataset)
    # Let's use the provided "index_approx" values from RANKS_OF2_OF6 and add some noise.
    rank_index = round(np.clip(random.normalvariate(rank_details["index_approx"], 0.05), 0.0, 1.5), 4) # Allow some over/under
    # Re-evaluating rank_index from example: "Major" (OF-3) is 0.333. This is 1/3.
    # If OF-2 is 0/3=0, OF-3 is 1/3, OF-4 is 2/3, OF-5 is 3/3=1. Then OF-6?
    # This implies a scale over 4 ranks (e.g. OF-2 to OF-5).
    # For now, let's stick to the RANKS_OF2_OF6 values, as they are explicitly for OF-2 to OF-6.
    # The example rank_index 0.3333333333333333 for Major (OF-3) seems to imply it is (3-2)/(6-2-1) ? or (idx-1)/ (num_ranks-1) where idx is 1-based for OF2,OF3..
    # Let's re-evaluate the example for rank_index: OF-2, OF-3, OF-4, OF-5, OF-6. (5 ranks)
    # If OF-2 (index 0) -> 0.0
    #    OF-3 (index 1) -> 0.25
    #    OF-4 (index 2) -> 0.5
    #    OF-5 (index 3) -> 0.75
    #    OF-6 (index 4) -> 1.0
    # Major is OF-3. (rank_numeric_value / (len(RANKS_OF2_OF6)-1) ) -> 1 / 4 = 0.25. Example is 0.333 for Major.
    # Let's use the provided index_approx from RANKS_OF2_OF6 and let the model figure it out.

    officer = {
        "branch": random.choice(BRANCHES),
        "rank": rank_name,
        "rank_index": rank_index,
        "first_name": first_name,
        "last_name": last_name,
        "age": int(years_of_service + random.randint(22, 28)), # Base age + years of service
        "specialty": random.choice(SPECIALTIES),
        "education": random.choice(EDUCATION_SOURCES),
        "service_start_date": service_start_date_obj.strftime("%Y-%m-%d"),
        "current_date": current_date_obj.strftime("%Y-%m-%d"),
        "years_of_service": years_of_service,
        "service_number": f"{random.choice(['A','C','N','M'])}{random.randint(100000000, 999999999)}",
        "competencies": {comp: random.randint(1, 5) for comp in COMPETENCIES_LIST},
        "competency_domains": {domain: round(random.uniform(1, 5), 2) for domain in COMPETENCY_DOMAINS_LIST},
        "psychometric_scores": {score: random.randint(1, 5) for score in PSYCHOMETRIC_SCORES_LIST},
        "combat_deployments": random.randint(0, int(years_of_service / 2) + 1), # Max 1 deployment every 2 years avg
        "medals_and_commendations": random.randint(0, int(years_of_service * 1.5)), # More service, potentially more medals
        "unit_readiness_score": random.randint(50, 100),
        "performance_review": f"Generated performance review for {first_name} {last_name}. Focus: {random.choice(COMPETENCIES_LIST)}.",
        "360_feedback": f"Generated 360 feedback. Strengths in {random.choice(COMPETENCIES_LIST)}, needs development in {random.choice(COMPETENCIES_LIST)}.",
        "interaction_transcript": f"Generated interaction transcript. Shows {random.choice(['good', 'fair', 'poor'])} communication on {random.choice(SPECIALTIES)} topics.",
        "promotion_potential_score": random.randint(1, 5),
        "leadership_style": random.choice(LEADERSHIP_STYLES),
        "leadership_competency_summary": {
            summary_key: round(random.uniform(1, 5), 2) for summary_key in LEADERSHIP_SUMMARY_LIST
        }
    }
    return officer

def generate_synthetic_data(num_officers, output_file):
    """
    Generates synthetic officer data with a clear relationship between input competencies and target leadership competencies.
    Adds randomization to prevent memorization.
    """
    logger.info(f"Generating {num_officers} synthetic officer profiles...")
    officers = []
    for _ in range(num_officers):
        officer = {}
        # Generate basic info
        officer['first_name'] = f"Officer_{random.randint(1, 1000)}"
        officer['last_name'] = f"Last_{random.randint(1, 1000)}"
        officer['age'] = random.randint(25, 60)
        officer['years_of_service'] = random.randint(1, 35)
        officer['rank_index'] = random.randint(1, 10)
        officer['combat_deployments'] = random.randint(0, 10)
        officer['medals_and_commendations'] = random.randint(0, 20)
        officer['unit_readiness_score'] = random.uniform(0, 100)
        officer['promotion_potential_score'] = random.uniform(0, 100)
        officer['service_number'] = f"SN{random.randint(10000, 99999)}"
        officer['service_start_date'] = (datetime.now() - timedelta(days=random.randint(365, 365*35))).strftime('%Y-%m-%d')
        officer['current_date'] = datetime.now().strftime('%Y-%m-%d')
        officer['branch'] = random.choice(['Army', 'Navy', 'Air Force', 'Marines'])
        officer['rank'] = random.choice(['Lieutenant', 'Captain', 'Major', 'Colonel', 'General'])
        officer['specialty'] = random.choice(['Infantry', 'Aviation', 'Intelligence', 'Logistics', 'Medical'])
        officer['education'] = random.choice(['Bachelor', 'Master', 'PhD'])
        officer['leadership_style'] = random.choice(['Democratic', 'Autocratic', 'Transformational', 'Servant'])

        # Generate competencies with some randomization
        officer['competencies'] = {}
        for comp in COMPETENCY_FEATURES:
            # Add some randomization to prevent memorization
            officer['competencies'][comp] = round(random.uniform(1, 5), 2)

        # Generate competency domains
        officer['competency_domains'] = {}
        for domain in COMPETENCY_DOMAIN_FEATURES:
            officer['competency_domains'][domain] = round(random.uniform(1, 5), 2)

        # Generate psychometric scores
        officer['psychometric_scores'] = {}
        for trait in PSYCHOMETRIC_FEATURES:
            officer['psychometric_scores'][trait] = round(random.uniform(1, 5), 2)

        # Generate leadership competency summary based on competencies with added randomization
        officer['leadership_competency_summary'] = {}
        # Strategic Thinking: weighted sum of relevant competencies plus noise
        officer['leadership_competency_summary']['strategic_thinking'] = np.clip(
            0.5 * officer['competencies']['thinks_strategically'] +
            0.3 * officer['competencies']['engages_in_ethical_reasoning'] +
            0.2 * officer['competencies']['builds_trust'] +
            np.random.normal(0, 0.2),
            1, 5
        )
        # Communication: weighted sum of relevant competencies plus noise
        officer['leadership_competency_summary']['communication'] = np.clip(
            0.4 * officer['competencies']['possesses_english_language_skills'] +
            0.4 * officer['competencies']['facilitates_collaboration_communication'] +
            0.2 * officer['competencies']['builds_consensus'] +
            np.random.normal(0, 0.2),
            1, 5
        )
        # Team Leadership: weighted sum of relevant competencies plus noise
        officer['leadership_competency_summary']['team_leadership'] = np.clip(
            0.3 * officer['competencies']['builds_trust'] +
            0.3 * officer['competencies']['facilitates_collaboration_communication'] +
            0.2 * officer['competencies']['enables_empowers_others'] +
            0.2 * officer['competencies']['relationship_oriented'] +
            np.random.normal(0, 0.2),
            1, 5
        )
        # Execution: weighted sum of relevant competencies plus noise
        officer['leadership_competency_summary']['execution'] = np.clip(
            0.3 * officer['competencies']['understands_capabilities'] +
            0.3 * officer['competencies']['provides_support_for_change'] +
            0.2 * officer['competencies']['enables_empowers_others'] +
            0.2 * officer['competencies']['upholds_principles'] +
            np.random.normal(0, 0.2),
            1, 5
        )
        # Adaptability: weighted sum of relevant competencies plus noise
        officer['leadership_competency_summary']['adaptability'] = np.clip(
            0.3 * officer['competencies']['thrives_in_ambiguity'] +
            0.3 * officer['competencies']['demonstrates_resilience'] +
            0.2 * officer['competencies']['learning_oriented'] +
            0.2 * officer['competencies']['anticipates_change_requirements'] +
            np.random.normal(0, 0.2),
            1, 5
        )

        officers.append(officer)

    # Save to JSONL file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for officer in officers:
            f.write(json.dumps(officer) + '\n')
    logger.info(f"Successfully generated and saved {num_officers} synthetic records to {output_file}")

def convert_json_array_to_jsonl(file_path):
    """
    Converts a JSON array file to JSON Lines format.
    """
    logger.info(f"Converting {file_path} to JSON Lines format...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    logger.info(f"Successfully converted {file_path} to JSON Lines format.")

def apply_competency_derivation_logic(file_path):
    """
    Applies the competency derivation logic to an existing JSON Lines file.
    """
    logger.info(f"Applying competency derivation logic to {file_path}...")
    officers = []
    with open(file_path, 'r') as f:
        for line in f:
            officer = json.loads(line)
            # Apply competency derivation logic
            officer['leadership_competency_summary'] = {}
            # Strategic Thinking: weighted sum of relevant competencies plus noise
            officer['leadership_competency_summary']['strategic_thinking'] = float(np.clip(
                0.5 * officer['competencies']['thinks_strategically'] +
                0.3 * officer['competencies']['engages_in_ethical_reasoning'] +
                0.2 * officer['competencies']['builds_trust'] +
                np.random.normal(0, 0.2),
                1, 5
            ))
            # Communication: weighted sum of relevant competencies plus noise
            officer['leadership_competency_summary']['communication'] = float(np.clip(
                0.4 * officer['competencies']['possesses_english_language_skills'] +
                0.4 * officer['competencies']['facilitates_collaboration_communication'] +
                0.2 * officer['competencies']['builds_consensus'] +
                np.random.normal(0, 0.2),
                1, 5
            ))
            # Team Leadership: weighted sum of relevant competencies plus noise
            officer['leadership_competency_summary']['team_leadership'] = float(np.clip(
                0.3 * officer['competencies']['builds_trust'] +
                0.3 * officer['competencies']['facilitates_collaboration_communication'] +
                0.2 * officer['competencies']['enables_empowers_others'] +
                0.2 * officer['competencies']['relationship_oriented'] +
                np.random.normal(0, 0.2),
                1, 5
            ))
            # Execution: weighted sum of relevant competencies plus noise
            officer['leadership_competency_summary']['execution'] = float(np.clip(
                0.3 * officer['competencies']['understands_capabilities'] +
                0.3 * officer['competencies']['provides_support_for_change'] +
                0.2 * officer['competencies']['enables_empowers_others'] +
                0.2 * officer['competencies']['upholds_principles'] +
                np.random.normal(0, 0.2),
                1, 5
            ))
            # Adaptability: weighted sum of relevant competencies plus noise
            officer['leadership_competency_summary']['adaptability'] = float(np.clip(
                0.3 * officer['competencies']['thrives_in_ambiguity'] +
                0.3 * officer['competencies']['demonstrates_resilience'] +
                0.2 * officer['competencies']['learning_oriented'] +
                0.2 * officer['competencies']['anticipates_change_requirements'] +
                np.random.normal(0, 0.2),
                1, 5
            ))
            officers.append(officer)

    # Save the updated data back to the file
    with open(file_path, 'w') as f:
        for officer in officers:
            f.write(json.dumps(officer) + '\n')
    logger.info(f"Successfully applied competency derivation logic to {file_path}")

# Remove the conversion step and just apply the competency derivation logic
# apply_competency_derivation_logic('data/synthetic_data/generated_256.json')
# apply_competency_derivation_logic('data/synthetic_data/generated_1000.jsonl')
# apply_competency_derivation_logic('data/synthetic_data/generated_2500.jsonl')

if __name__ == "__main__":
    apply_competency_derivation_logic('data/synthetic_data/generated_256.json')
    apply_competency_derivation_logic('data/synthetic_data/generated_1000.jsonl')
    apply_competency_derivation_logic('data/synthetic_data/generated_2500.jsonl')