CHOICES = [
    "Train",
    "Swissmetro",
    "Car",
]

CHOICES_VAC = [
    "Unvaccinated",
    "Vaccinated_no_booster",
    "Booster",
]

INIT_CONCEPT_LIB = [
    "Travel mode choice is influenced heavily by trip purpose, cost, and travel time.",
    "Household decisions, income, work schedules, self-identity, and toll policies shape travel behavior and mode flexibility.",
    "Proximity to transit stops, quality public transport, bike lanes, and first/last-mile connectivity drive mode choices.",
    "Time constraints and complex planning require flexible departures; cars uniquely offer schedule freedom.",
    "Public transit can be slow/complex (especially with luggage/children) while cycling faces route and load difficulties.",
    "Personal safety concerns and inadequate infrastructure deter active travel modes.",
    "Public transport attractiveness (comfort, frequency, pricing) is crucial for reducing car dependency."
]

INIT_CONCEPT_LIB_VAC = [
    # Core first-dose levers
    "Belief that available COVID-19 vaccines are safe is the single biggest trigger for starting vaccination.",
    "Seeing COVID-19 as a serious, preventable threat lifts first-dose uptake.",
    "Trust in science and in government delivery boosts both first doses and boosters; distrust drags both.",

    # Social & demographic signals
    "Physician endorsement moves first doses, but nurses and other HCWs drive booster momentum (positive when pro-vax, negative when skeptical).",
    "Older age sharply raises booster odds, while younger adults need 'freedom & lifestyle' framing to start any dose.",

    # Booster-specific frictions
    "High ongoing attention can *backfire* at booster stage if dominated by side-effect chatter; low-attention followers boost on schedule.",
    "Refusing to state income flags institutional distrust and lowers both initial and booster uptake.",
    "Prior COVID-19 infection breeds 'natural-immunity' complacency, cutting booster intent.",
    "Even among the vaccinated, small logistical hassles (booking, transport, time off) shave booster coverage.",
    "Message fatigue and a sense of diminishing returns demand concise, variant-focused booster messaging."
]

SW_VAR_LIB = [
    "trip_purpose",                  # Purpose of the trip (commuting, business, shopping, etc.)
    "is_first_class_traveler",       # Whether the traveler used first class
    "ticket_payer_type",             # Who pays for the ticket (self, employer, etc.)
    "number_of_luggage_items",       # Number of luggage pieces
    "traveler_age_group",            # Age category of the traveler
    "is_male",                       # Gender (1: male, 0: female)
    "annual_income_level",           # Income bracket in thousand CHF
    "has_ga_travel_pass",            # Whether traveler has a Swiss GA travel card
    "origin_canton_code",            # Code representing the origin canton
    "destination_canton_code",       # Code representing the destination canton
    "is_car_available",              # Whether a car is available for use
    "train_total_travel_time_min",   # Total train travel time in minutes (door-to-door)
    "train_ticket_cost_chf",         # Total train travel cost in CHF
    "train_service_headway_min",     # Average waiting time between trains
    "sm_travel_time_min",            # Swissmetro travel time in minutes
    "sm_ticket_cost_chf",            # Swissmetro cost in CHF
    "sm_service_headway_min",        # Swissmetro headway in minutes
    "car_travel_time_min",           # Estimated car travel time in minutes
    "car_travel_cost_chf"            # Estimated car cost in CHF
]

SW_VAR_LIB_VAC = [
    'covid_threat',
    'covid_preventable_by_vax',
    'risk_of_covid_greater_than_vax',
    'vaccine_safe_to_me',
    'trust_government',
    'trust_science',
    'have_covid_sick_family_member',
    'age',
    'gender',
    'vax_protect_long_unsure',
    'vax_protect_long_yes',
    'less_attention_to_vax_info',
    'more_attention_to_vax_info',
    'nurse',
    'healthcare_worker',
    'physician',
    'have_university_degree',
    'income_below_median',
    'income_unknown']

VAR_MAPPING = {
    "trip_purpose": "PURPOSE",
    "is_first_class_traveler": "FIRST",
    "ticket_payer_type": "TICKET",
    "number_of_luggage_items": "LUGGAGE",
    "traveler_age_group": "AGE",
    "females_age_group": "AGE",
    "males_age_group": "AGE",
    "is_male": "MALE",
    "is_female": "FEMALE",
    "annual_income_level": "INCOME",
    "has_ga_travel_pass": "GA",
    "origin_canton_code": "ORIGIN",
    "destination_canton_code": "DEST",
    "destination_centon_code": "DEST",
    "is_car_available": "CAR_AV",
    "train_total_travel_time_min": "TRAIN_TT",
    "train_ticket_cost_chf": "TRAIN_CO",
    "train_service_headway_min": "TRAIN_HE",
    "sm_travel_time_min": "SM_TT",
    "sm_ticket_cost_chf": "SM_CO",
    "sm_service_headway_min": "SM_HE",
    "car_travel_time_min": "CAR_TT",
    "car_travel_cost_chf": "CAR_CO"
}

VARIABLE_MAP = {
    'trip_purpose': 'X_dict["trip_purpose"]',
    'is_first_class_traveler': 'X_dict["is_first_class_traveler"]',
    'ticket_payer_type': 'X_dict["ticket_payer_type"]',
    'number_of_luggage_items': 'X_dict["number_of_luggage_items"]',
    'traveler_age_group': 'X_dict["traveler_age_group"]',
    'females_age_group': 'X_dict["traveler_age_group"]',
    'males_age_group': 'X_dict["traveler_age_group"]',
    'is_male': 'X_dict["is_male"]',
    'is_female': 'X_dict["is_female"]',
    'annual_income_level': 'X_dict["annual_income_level"]',
    'has_ga_travel_pass': 'X_dict["has_ga_travel_pass"]',
    'origin_canton_code': 'X_dict["origin_canton_code"]',
    'destination_canton_code': 'X_dict["destination_canton_code"]',
    'is_car_available': 'X_dict["is_car_available"]',
    'train_total_travel_time_min': 'X_dict["train_total_travel_time_min"]',
    'train_ticket_cost_chf': 'X_dict["train_ticket_cost_chf"]',
    'train_service_headway_min': 'X_dict["train_service_headway_min"]',
    'sm_travel_time_min': 'X_dict["sm_travel_time_min"]',
    'sm_ticket_cost_chf': 'X_dict["sm_ticket_cost_chf"]',
    'sm_service_headway_min': 'X_dict["sm_service_headway_min"]',
    'car_travel_time_min': 'X_dict["car_travel_time_min"]',
    'car_travel_cost_chf': 'X_dict["car_travel_cost_chf"]'
}

VARIABLE_MAP_VAC = {
    'covid_threat': 'X_dict["covid_threat"]',
    'covid_preventable_by_vax': 'X_dict["covid_preventable_by_vax"]',
    'risk_of_covid_greater_than_vax': 'X_dict["risk_of_covid_greater_than_vax"]',
    'vaccine_safe_to_me': 'X_dict["vaccine_safe_to_me"]',
    'trust_government': 'X_dict["trust_government"]',
    'trust_science': 'X_dict["trust_science"]',
    'have_covid_sick_family_member': 'X_dict["have_covid_sick_family_member"]',
    'age': 'X_dict["age"]',
    'gender': 'X_dict["gender"]',
    'vax_protect_long_unsure': 'X_dict["vax_protect_long_unsure"]',
    'vax_protect_long_yes': 'X_dict["vax_protect_long_yes"]',
    'less_attention_to_vax_info': 'X_dict["less_attention_to_vax_info"]',
    'more_attention_to_vax_info': 'X_dict["more_attention_to_vax_info"]',
    'nurse': 'X_dict["nurse"]',
    'healthcare_worker': 'X_dict["healthcare_worker"]',
    'physician': 'X_dict["physician"]',
    'have_university_degree': 'X_dict["have_university_degree"]',
    'income_below_median': 'X_dict["income_below_median"]',
    'income_unknown': 'X_dict["income_unknown"]'
}

OP_LIB = [
    # "add",          # addition
    # "subtract",     # subtraction
    # "multiply",     # multiplication
    # "divide",       # division
    # "power",        # exponentiation
    "+",  # addition
    "-",  # subtraction
    "*",  # multiplication
    "/",  # division
    "**",  # exponentiation
    "sqrt",         # square root
    "log",          # natural logarithm
    "exp",          # exponential
    "abs",          # absolute value
    "sin",          # sine
    "cos",          # cosine
    "tan",          # tangent
    # "max",          # maximum
    "min",          # minimum
    "mod",          # modulo
]

IDENTIFY_FACTOR_RELATION_SYS = """You are a transportation planner specializing in analyzing the relationships among
various factors that influence travel behavior. You will be provided with two types of information: individual
features (delimited by <FEATURES> and </FEATURES>) and preliminary travel mode knowledge (delimited by <KNOWLEDGE>
and </KNOWLEDGE>). Your task is to carefully review these inputs and in detailed sentence describe how the provided
features interrelate. Ensure your response includes as many specific details as possible about the relationships,
but do not propose any new features or suggest modifications to the existing ones. Example: For an individual,
sensitivity to travel time increases in a squared manner, whereas sensitivity to travel cost follows a logarithmic
trend."""

IDENTIFY_FACTOR_RELATION_SYS2 = """You are a transportation planner specializing in analyzing the relationships among
various factors that influence travel behavior. You will be provided with two types of information: individual
features (delimited by <FEATURES> and </FEATURES>) and preliminary travel mode knowledge (delimited by <KNOWLEDGE>
and </KNOWLEDGE>). Your task is to carefully review these inputs and in detailed sentence describe how the provided
features interrelate. Ensure your response includes as many specific details as possible about the relationships,
but do not propose any new features or suggest modifications to the existing ones. Example: Time: quadratic, Cost: log, luggage: linear."""

IDENTIFY_FACTOR_RELATION_SYS2_VAC = """You are a public‑health analyst specializing in analyzing the relationships among
various factors that influence COVID‑19 vaccination behavior. You will be provided with two types of information: individual
features (delimited by <FEATURES> and </FEATURES>) and preliminary vaccination behavior knowledge (delimited by <KNOWLEDGE>
and </KNOWLEDGE>). Your task is to carefully review these inputs and in detailed sentence describe how the provided
features interrelate. Ensure your response includes as many specific details as possible about the relationships,
but do not propose any new features or suggest modifications to the existing ones. Example: Vaccine safety belief: linear positive, Government trust: logarithmic decay, Prior infection: inverse exponential, Age: piecewise increasing."""

IDENTIFY_FACTOR_RELATION = """
<FEATURES>{features}</FEATURES>
<KNOWLEDGE>{knowledge}</KNOWLEDGE>
"""

IDENTIFY_FACTOR_RELATION2 = """
<GROUP DESCRIPTION>{description}</GROUP DESCRIPTION>
<FEATURES>{features}</FEATURES>
<KNOWLEDGE>{knowledge}</KNOWLEDGE>
You should ONLY provide the relations between the features.
YOU MUST return your assumption in this exact format: ```["relation_0","relation_1", ...]```
"""

GEN_RANDOM_N_SYS = """You are a helpful assistant that proposes a mathematical expression based on some provided
suggestions. Your goal is to:

1. **Incorporate** each suggestion in a way that adds clarity or insight.
2. **Use only** the specified variables: {variables}
3. **Represent all constants** with the symbol "C", and all coefficients with the symbol "K".
4. **Restrict** yourself to the following operators: {operators}
5. **Explain briefly** how you integrated each suggestion.

Your response must:
- Propose exactly **{N}** new expressions.
- Provide **short commentary** for each expression, highlighting how each suggestion guided your design choices.
- MUST return in this exact format: ```[("expressions_0","Commentary_0"), ..., ("expressions_{N}","Commentary_{N}")]```
"""

GEN_RANDOM_N_SYS2 = """You are a helpful assistant that proposes mathematical expressions based on some provided
suggestions. Your goal is to:

0. **Task**: Generate utility functions for travel mode choice of group of {description}.
1. **Use only** the specified variables: {variables}
2. **Represent all constants** with the symbol "C", and all coefficients with the symbol "K".
3. **Restrict** yourself to the following operators: {operators}
4. **For each group**, suggest utility functions for train, car, and Swissmetro respectively.

Your response must:
- Propose exactly **{N}** groups of expressions.
- MUST return in this exact format: ```[("expressions_car","expressions_train","expressions_metro"), ...]```, replace expressions_mode with your proposed expressions.
"""

GEN_RANDOM_N_SYS2_VAC = """You are a helpful assistant that proposes mathematical expressions based on some provided
suggestions. Your goal is to:

0. **Task**: Generate utility functions for COVID‑19 vaccination decision outcomes of group {description}.
1. **Use only** the specified variables: {variables}
2. **Represent all constants** with the symbol "C", and all coefficients with the symbol "K".
3. **Restrict** yourself to the following operators: {operators}
4. **For each group**, suggest utility functions for three alternatives in this order:
   - remain unvaccinated,
   - vaccinated without booster,
   - vaccinated with at least one booster dose.

Your response must:
- Propose exactly **{N}** groups of expressions.
- MUST return in this exact format: ```[("expr_unvaccinated","expr_vaccinated_no_booster","expr_booster"), ...]```,
  replacing each expr_* with your proposed expression."""

GEN_RANDOM_N_SYS2_VAC_2 = """You are a helpful assistant that proposes mathematical expressions based on some provided
suggestions. Your goal is to:

0. **Task**: Generate utility functions for COVID‑19 vaccination decision outcomes of group {description}.
1. **Use only** the specified variables: {variables}
2. **Represent all constants** with the symbol "C", and all coefficients with the symbol "K".
3. **Restrict** yourself to the following operators: {operators}

Your response must:
- Propose exactly **{N}** groups of expressions.
- MUST return in this exact format: ```["expr1","expr2","expr3", ...]```, replacing each expr* with your proposed expression."""

GEN_RANDOM_N = """
Suggestions: {suggestions}
The expressions MUST be in python format, e.g. "K*x + K*y + C"
MUST return in this exact format: ```[("expressions_0","Commentary_0"), ..., ("expressions_{N}","Commentary_{N}")]```
"""

GEN_RANDOM_N2 = """
Suggestions: {suggestions}
DO NOT USE == or max/min() in your expressions.
"""

# generate concepts
ANALYZE_RESULTS_SYS = """
You are a creative and insightful mathematical research assistant. You have been provided with two sets of utility expressions: one function group labeled “Good Expressions” and one labeled “Bad Expressions.” Your objective is to hypothesize about the underlying assumptions or principles that might generate the good expressions yet exclude the bad ones.

Key Points:
1. Focus primarily on the good expressions' mathematical structures and any connections they might have to physical or applied contexts.
2. Capital “C” in any expression is just an arbitrary constant.
3. Do not discuss or compare the expressions in terms of their simplicity or complexity.
4. Provide your reasoning step by step, but keep it very concise and genuinely insightful. No more than 5 lines.
"""

ANALYZE_RESULTS = """
Good Expression 1: (train: {texpr1}, car: {cexpr1}, metro: {mexpr1}), accuracy: {acc1}
Good Expression 2: (train: {texpr2}, car: {cexpr2}, metro: {mexpr2}), accuracy: {acc2}
Bad Expression 1: (train: {bexpr1}, car: {bexpr2}, metro: {bexpr3}), accuracy: {acc3}

Above expressions are travel mode choice utility functions of group of {description}. Propose {N} hypotheses that would be appropriate given the expressions. Provide short commentary for each of your decisions. Do not talk about topics related to the simplicity or complexity of the expressions. I want ideas that are unique and interesting enough to amaze the world's best mathematicians. """

ANALYZE_RESULTS_VAC = """Good Expression 1: (unvaccinated: {expr_unvax1}, vaccinated_no_booster: {expr_vnb1}, booster: {expr_booster1}), accuracy: {acc1}
Good Expression 2: (unvaccinated: {expr_unvax2}, vaccinated_no_booster: {expr_vnb2}, booster: {expr_booster2}), accuracy: {acc2}
Bad Expression 1: (unvaccinated: {bexpr1}, vaccinated_no_booster: {bexpr2}, booster: {bexpr3}), accuracy: {acc3}

Above expressions are vaccination‑decision utility functions for group {description}. Propose {N} hypotheses that would be appropriate given the expressions. Provide short commentary for each of your decisions. Do not talk about topics related to the simplicity or complexity of the expressions. I want ideas that are unique and interesting enough to amaze the world's best epidemiologists and behavioral scientists."""

CROSS_OVER_SYS = """
You are a helpful assistant that recombines two mathematical expressions based on some provided suggestions. Your goal is to produce new expressions that:

1. **Blend or merge** elements from both reference expressions in a way that reflects the suggestions.
2. **Adhere to the following constraints**:
   - You may only use the variables in library: {variables}
   - All constants must be represented with the symbol C
   - Only the following operators are allowed: {operators}

**Guidelines**:
- Propose exactly **{N}** new expressions.
- Each new expression should integrate elements of both reference expressions. You can also propose new term with variables that are in library but not in old expressions.
- If any suggestions appear contradictory, reconcile them reasonably.

MUST return in this exact format: ```[("expressions_car","expressions_train","expressions_metro"), ...]```, replace expressions_ with your proposed expressions."""

CROSS_OVER_SYS_VAC = """
You are a helpful assistant that recombines vaccination‑decision utility expressions based on provided suggestions. Your goal is to produce new expressions that:

1. **Blend or merge** elements from both reference expressions in a way that reflects the suggestions.
2. **Adhere to the following constraints**:
   - You may only use the variables in library: {variables}
   - All constants must be represented with the symbol C
   - Only the following operators are allowed: {operators}

**Guidelines**:
- Propose exactly **{N}** new expression triplets.
- Each triplet should integrate elements of both reference triplets. You may also introduce new terms that employ allowed variables even if absent in the references.
- If any suggestions appear contradictory, reconcile them reasonably.

MUST return in this exact format: ```[("expr_unvaccinated","expr_vaccinated_no_booster","expr_booster"), ...]```, replacing each expr_* with your proposed expression."""

CROSS_OVER = """
Suggestion: {suggestions}
Reference Expression group 1: (train: {texpr1}, car: {cexpr1}, metro: {mexpr1})
Reference Expression group 2: (train: {texpr2}, car: {cexpr2}, metro: {mexpr2})

Propose {N} expressions that would be appropriate given the suggestions and references.
DO NOT USE == or max/min() in your expressions.
"""

CROSS_OVER_VAC = """
Suggestion: {suggestions}
Reference Expression group 1: (unvaccinated: {expr_unvax1}, vaccinated_no_booster: {expr_vnb1}, booster: {expr_booster1})
Reference Expression group 2: (unvaccinated: {expr_unvax2}, vaccinated_no_booster: {expr_vnb2}, booster: {expr_booster2})

Propose {N} expressions that would be appropriate given the suggestions and references.
DO NOT USE == or max/min() in your expressions.
"""


SELECTION_GUIDANCE_SYS = """You are a travel-behavior preference selector.
You will be given two blocks of information:

<DEMOGRAPHICS> ... </DEMOGRAPHICS>
<UTILITY_FUNCTION> ... </UTILITY_FUNCTION>

Your goal: choose the single best-matching high-level preference template for this group **exactly** from the catalogue below and output **only** the template name (uppercase).

CATALOGUE
- TIME_EFFICIENCY   : travellers primarily minimise total travel time.
- COST_SAVING       : travellers primarily minimise direct monetary cost.
- COMFORT_SEEKING   : travellers value comfort/service frequency and dislike crowding.
- BALANCED          : sensitivities are evenly distributed across factors.

Return nothing else — no commentary, no punctuation, just the template name."""

SELECTION_GUIDANCE = """
<DEMOGRAPHICS>{demographics}</DEMOGRAPHICS>
<UTILITY_FUNCTION>{utility}</UTILITY_FUNCTION>
"""

# --- Vaccination preference‑template selection prompts ---
SELECTION_GUIDANCE_SYS_VAC = """You are a vaccination‑behavior preference selector.
You will be given two blocks of information:

<DEMOGRAPHICS> ... </DEMOGRAPHICS>
<UTILITY_FUNCTION> ... </UTILITY_FUNCTION>

Your goal: choose the single best‑matching high‑level preference template for this group **exactly** from the catalogue below and output **only** the template name (uppercase).

CATALOGUE
- SAFETY_CONCERNED   : individuals' utility is dominated by perceived vaccine safety and fear of side‑effects.
- THREAT_AVOIDING    : individuals primarily respond to perceived severity of COVID‑19 and believe disease risk > vaccine risk.
- TRUSTING_AUTHORITY : individuals closely follow trusted scientific, governmental, or clinical recommendations.
- BALANCED           : sensitivities are relatively even across the listed factors.

Return nothing else — no commentary, no punctuation, just the template name."""

TEMPLATE_INDIVIDUAL = """Individual Profile
Demographics: {demographics}
{trip_info}
"""

PREDICTION_SYS = """You are a decision assistant that recommends the most suitable travel mode for an individual trip.

You will receive three blocks:
<TEMPLATE> … optimized preference template … </TEMPLATE>
<PROFILE> … individual profile … </PROFILE>
<ALTERNATIVES> … attributes of Swissmetro, Train, and Car … </ALTERNATIVES>

**Instructions:**
1. **Use the <TEMPLATE> as a guide** for understanding the individual's likely preference bias (e.g., time efficiency, cost saving, comfort seeking, balanced).
2. **Analyze the individual's profile** (age, gender, income, trip details) and **the attributes of the travel alternatives** (time and cost).
3. **Recommend the most suitable travel mode** from the following options: "Swissmetro", "Train", "Car".

**Output only the recommended travel mode.**
"""

PREDICTION_SYS_POSSIBILITY = """You are a decision assistant that recommends the most suitable travel mode for an individual trip by estimating a probability distribution over three options: Swissmetro, Train, and Car.

You will receive three blocks:
<TEMPLATE> … optimized preference template … </TEMPLATE>
<PROFILE> … individual profile … </PROFILE>
<ALTERNATIVES> … attributes of Swissmetro, Train, and Car … </ALTERNATIVES>

**Instructions:**
1. **Use the <TEMPLATE> as a guide** for understanding the individual's likely preference bias (e.g., time efficiency, cost saving, comfort seeking, balanced).
2. **Analyze the <PROFILE>** (age, gender, income, trip details) **and the <ALTERNATIVES>** (travel time, cost, headway).
3. **Estimate and output a probability** for each travel mode, such that all three probabilities sum to 1.

**Output format (JSON only):**
```json
{
  "Swissmetro": <float between 0 and 1>,
  "Train":      <float between 0 and 1>,
  "Car":        <float between 0 and 1>
}
```
No additional text; just the JSON object with normalized probabilities."""

PREDICTION_SYS_ZEROSHOT = """You are a decision assistant that predicts a probability distribution over three travel modes, Swissmetro, Train, and Car, for a single trip.

You will receive two blocks of text:
<TRIP_INFO>
… details like trip purpose, luggage, payment, origin, destination …
</TRIP_INFO>
<TRANSPORT_OPTIONS>
… list of modes with travel time, cost, headway …
</TRANSPORT_OPTIONS>

**Instructions:**
1. Use only the information in <TRIP_INFO> and <TRANSPORT_OPTIONS>.
2. Estimate a probability for each mode so they sum to 1.
3. **Output only** a JSON object, for example:
```json
{
  "Swissmetro": <float between 0 and 1>,
  "Train":      <float between 0 and 1>,
  "Car":        <float between 0 and 1>
}
```
No additional text; just the JSON object with normalized probabilities."""

PREDICTION_SYS_FEWSHOT = """You are a decision assistant that predicts a probability distribution over three travel modes—Swissmetro, Train, and Car—for a set of travel records.

You will receive multiple records. Each record consists of three blocks:
<TRIP_INFO>
… trip details: purpose, luggage, payment, origin, destination …
</TRIP_INFO>
<TRANSPORT_OPTIONS>
… each mode's travel time, cost, headway …
</TRANSPORT_OPTIONS>
<CHOICE>
… either a JSON object with probabilities (for examples), or left empty for the record to predict …
</CHOICE>

For records where <CHOICE> is filled, treat them as examples.
For the final record (with an empty <CHOICE>), output **only** the JSON object of normalized probabilities (summing to 1), with no extra text."""

PREDICTION_FEWSHOT = """
<TRIP_INFO>
{trip_info_1}
</TRIP_INFO>
<TRANSPORT_OPTIONS>
{transport_options_1}
</TRANSPORT_OPTIONS>
<CHOICE>
{choice_1}
</CHOICE>
<TRIP_INFO>
{trip_info_2}
</TRIP_INFO>
<TRANSPORT_OPTIONS>
{transport_options_2}
</TRANSPORT_OPTIONS>
<CHOICE>
{choice_2}
</CHOICE>
<TRIP_INFO>
{trip_info_3}
</TRIP_INFO>
<TRANSPORT_OPTIONS>
{transport_options_3}
</TRANSPORT_OPTIONS>
<CHOICE>
{choice_3}
</CHOICE>
<TRIP_INFO>
{trip_info_4}
</TRIP_INFO>
<TRANSPORT_OPTIONS>
{transport_options_4}
</TRANSPORT_OPTIONS>
<CHOICE>
{choice_4}
</CHOICE>
<TRIP_INFO>
{trip_info_5}
</TRIP_INFO>
<TRANSPORT_OPTIONS>
{transport_options_5}
</TRANSPORT_OPTIONS>
<CHOICE>
{choice_5}
</CHOICE>
<TRIP_INFO>
{trip_info_6}
</TRIP_INFO>
<TRANSPORT_OPTIONS>
{transport_options_6}
</TRANSPORT_OPTIONS>
<CHOICE>
Please predict the travel mode for this trip.
<CHOICE>"""

PREDICTION = """
<TEMPLATE>{template_name}</TEMPLATE>
<PROFILE>
{individual_block}
</PROFILE>
<ALTERNATIVES>
{options}
</ALTERNATIVES>
"""

PREDICTION_ZEROSHOT = """
<TRIP_INFO>
{trip_info}
</TRIP_INFO>
<TRANSPORT_OPTIONS>
{transport_options}
</TRANSPORT_OPTIONS>"""

PREDICTION_P = """
{persona}
<TEMPLATE>{template_name}</TEMPLATE>
<PROFILE>
{individual_block}
</PROFILE>
<ALTERNATIVES>
{options}
</ALTERNATIVES>
"""

PREDICTION_SYS_VAC = """You are a vaccination decision assistant.

You will receive three blocks:
<TEMPLATE> … optimized preference template … </TEMPLATE>
<PROFILE> … individual profile … </PROFILE>
<ALTERNATIVES> … attributes of the vaccination alternatives … </ALTERNATIVES>

**Instructions:**
1. **Use the <TEMPLATE> as a guide** for understanding the individual's likely preference bias (e.g., safety‑concerned, threat‑avoiding, trusting‑authority, balanced).
2. **Analyze the individual's profile** (age, gender, occupation, prior infection, demographic traits) **and the attributes of the vaccination alternatives** (efficacy, side‑effect risk, convenience, cost).
3. **Recommend the most suitable choice** from the following options (use these labels exactly):
   - "Unvaccinated"
   - "Vaccinated_no_booster"
   - "Booster"

[Debias] Assume all three classes are equally likely regardless of real-world prevalence. Focus only on textual evidence in the answer.

**Output only the recommended choice.**
"""

PREDICTION_SYS_VAC_POSSIBILITY = """You are a vaccination decision assistant.

You will receive three blocks:
<TEMPLATE> … optimized preference template … </TEMPLATE>
<PROFILE> … individual profile … </PROFILE>
<ALTERNATIVES> … attributes of the vaccination alternatives … </ALTERNATIVES>

**Instructions:**
1. **Use the <TEMPLATE> as a guide** for understanding the individual's likely preference bias (e.g., safety‑concerned, threat‑avoiding, trusting‑authority, balanced).
2. **Analyze the individual's profile** (age, gender, occupation, prior infection, demographic traits) **and the attributes of the vaccination alternatives** (efficacy, side‑effect risk, convenience, cost).
3. **Estimate and output a probability** for each vaccination alternative, such that all three probabilities sum to 1.

**Output format (JSON only):**
```json
{
  "Unvaccinated":           <float between 0 and 1>,
  "Vaccinated_no_booster": <float between 0 and 1>,
  "Booster":               <float between 0 and 1>
}
```
[Debias] Assume all three classes are equally likely regardless of real-world prevalence. Focus only on textual evidence in the answer.

No additional text; just the JSON object with normalized probabilities.
"""

PREDICTION_SYS_VAC_POSSIBILITY_FEWSHOT = """You are a vaccination decision assistant.

You will be given five examples in succession, each consisting of four blocks:
<TEMPLATE> … optimized preference template … </TEMPLATE>
<PROFILE> … individual profile … </PROFILE>
<ALTERNATIVES> … attributes of the vaccination alternatives … </ALTERNATIVES>
<CHOICE> … the person’s actual choice: "Unvaccinated", "Vaccinated_no_booster", or "Booster" … </CHOICE>

After those five examples, you will receive one more set of three blocks:
<TEMPLATE> … optimized preference template … </TEMPLATE>
<PROFILE> … individual profile … </PROFILE>
<ALTERNATIVES> … attributes of the vaccination alternatives … </ALTERNATIVES>

Your task is to:
1. **Learn from the five examples** how the <TEMPLATE>, <PROFILE>, and <ALTERNATIVES> map to the observed <CHOICE>.
2. **Analyze the final <TEMPLATE>, <PROFILE>, and <ALTERNATIVES>** in the same way.
3. **Estimate and output a probability** for each vaccination alternative, so that the three probabilities sum to 1.

**Output format (JSON only):**
```json
{
  "Unvaccinated":           <float between 0 and 1>,
  "Vaccinated_no_booster": <float between 0 and 1>,
  "Booster":               <float between 0 and 1>
}```
No additional text—just the JSON object with normalized probabilities."""

PREDICTION_VAC = """
<TEMPLATE>{template_name}</TEMPLATE>
<PROFILE>
{individual_block}
</PROFILE>
<ALTERNATIVES>
{options}
</ALTERNATIVES>
"""

PREDICTION_P_VAC = """
{persona}
<TEMPLATE>{template_name}</TEMPLATE>
<PROFILE>
{individual_block}
</PROFILE>
<ALTERNATIVES>
{options}
</ALTERNATIVES>
"""

# --- Vaccination persona generation prompts ---

PERSONA_SYS_VAC = """
You are a vaccination‑behavior analyst.

Goal
From demographic feature and survey answers of one individual, infer **one concise English sentence** that best describes the person's vaccination persona.

Instructions
1. Examine stated attitudes in the survey data to detect signals of safety concern, authority trust, threat perception, convenience sensitivity, etc.
2. Output **exactly one English sentence** summarizing the individual's vaccination persona.
3. Produce nothing else — no additional text, JSON, or extra line breaks.
4. [Debias] Treat every demographic or attitude group as equally likely to be pro- or anti-vaccination. Rely only on the provided data; avoid moral judgements, stereotypes, or normative language.
"""

PERSONA_VAC = """
Please generate a one‑sentence vaccination persona based on following demographic features and survey answers:

<FEATURES>
{features}
</FEATURES>
<SURVEY>
{survey}
</SURVEY>

[Debias] Assume no prior about vaccination behavior beyond what the text explicitly states.
"""

PERSONA_SYS = """
You are a transportation-behavior analyst.

Goal
From past trips of one individual, infer **one concise English sentence** that best describes the person's travel persona.

Instructions
1. Compare trip description and choices to detect indicators of time-sensitivity, cost-sensitivity, comfort preference, car availability, etc.
2. Output **exactly one English sentence** summarizing the traveler's persona.
3. Produce nothing else, no additional text, JSON, or extra line breaks."""

PERSONA = """
Please generate a one-sentence travel persona based on the following trip records:

{records}"""
