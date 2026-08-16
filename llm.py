

#API-key setup — DO NOT hard-code your key in this cell.

import os

# --- Local (with a .env file) ---
# from dotenv import load_dotenv
# load_dotenv()
# API_KEY = os.environ["GROQ_API_KEY"]

#--- Google Colab (Secrets panel) ---
from google.colab import userdata
API_KEY = userdata.get("GROQ_API_KEY")

# TODO: set API_KEY using ONE of the methods above.

# OpenAI-compatible client (works for Groq and OpenAI; Gemini users see their docs):
from openai import OpenAI

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.groq.com/openai/v1",   # remove this line if using OpenAI itself
)
MODEL = "llama-3.3-70b-versatile"                # or your provider's model name

print("Client ready.")

#part 1.1
def ask_llm(user_prompt, system_prompt="You are a helpful assistant.",
            temperature=0.7, max_tokens=500):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response, response.choices[0].message.content

# Test call
response, answer = ask_llm("What is the capital of Ghana?")
print(answer)
print(response.usage)

"""**What is the difference between the system and user roles? Give an example of something that belongs in each.**

system tells the assistant how to behave for the whole conversation for eaxample You are a terse code reviewer who only responds in bullet points.
user is the actual question or request for example "What is the capital of Ghana?"


 **2. What is a token, roughly? Why do API providers bill per token rather than per request?**

 Tokens: a token is a small piece of text, a word, part of a word, or punctuation (e.g. unbelievable could split into un, believ, able). Providers charge per token because cost depends on how much text is processed, not how many requests you send. A one-word request and a 10,000-word request cost very differently, even though both are one request.

"""

#part 1.2
question = "Suggest a name for a savings product for market traders in Accra."

print("=== Temperature = 0.0 ===")
for i in range(5):
    _, answer = ask_llm(question, temperature=0.0)
    print(f"{i+1}. {answer}\n")

print("=== Temperature = 1.2 ===")
for i in range(5):
    _, answer = ask_llm(question, temperature=1.2)
    print(f"{i+1}. {answer}\n")

"""**What did you observe at each temperature?**
At temperature=0.0, the model gives the same (or near-identical) answer every time it always picks the most likely next word, so there's little to no variation.

At temperature=1.2, the answers differ each time that is some more creative, some a bit odd or less coherent. The model is sampling more randomly instead of always picking the safest word.

**For the loan decision-support system you are about to build, which temperature regime is appropriate, and why?**

low temperature (close to 0.0) is the right choice. Loan decisions need to be consistent and predictable that is the same applicant with the same details should get the same recommendation every time. High temperature would introduce randomness into something that needs to be reliable, auditable, and fair, which is risky when real financial decisions are involved.

"""

#section 2

LETTERS = {
"L001": """Dear Sir/Madam,
My name is Akosua Mensah and I have been selling provisions at Makola Market for 12 years.
I am applying for a loan of GHS 8,000 to buy a deep freezer and expand into frozen foods.
My current stall makes about GHS 900 profit each month. I have saved GHS 2,500 with your
susu scheme over the past two years and I have never missed a contribution. I can repay
GHS 450 monthly over 20 months. My sister, a teacher, will stand as my guarantor.
Thank you for considering my application.""",

"L002": """Hello,
I am Kwame Boateng, a commercial driver in Kumasi. I need GHS 25,000 urgently to repair my
trotro engine and settle some personal debts. Business has been slow but it will surely
pick up after the festive season. I can pay back whenever the money comes. I do not have
collateral at the moment but God willing everything will be fine. Please help me quickly.""",

"L003": """Dear Loan Committee,
I am Efua Darko, owner of Darko Fashions, a registered dressmaking business in Takoradi
(registration no. BN-2019-4482). I employ three apprentices. I request GHS 15,000 to
purchase two industrial sewing machines and fabric stock ahead of the Christmas season.
Last year my December revenue alone was GHS 22,000; monthly profit averages GHS 2,800.
I hold a fixed deposit of GHS 5,000 with GCB which I can pledge. Proposed repayment:
GHS 1,100 monthly for 15 months. Attached are my sales records for the past 18 months.""",

"L004": """Good day,
My name is Yaw Owusu. I want a loan for my poultry farm at Nsawam. The amount is GHS 12,000
for feed and 500 new layers. I started the farm last year. Sometimes I make good money,
around GHS 1,500 in a good month, but bird flu affected us in March and I lost many birds.
I am rebuilding now. I can repay in 18 months. My uncle has agreed to guarantee the loan
with his taxi.""",

"L005": """Dear Manager,
I am writing on behalf of the Adenta Women's Weaving Cooperative (14 members). We seek
GHS 30,000 to buy a bulk order of yarn directly from the factory, cutting out middlemen and
raising our margins from 15% to about 35%. The cooperative has operated for 6 years and
holds GHS 9,000 in our group account. We propose repayment of GHS 2,000 monthly over
16 months, backed by our group savings and joint liability agreement.""",

"L006": """Hi,
This is Kofi. I saw your advert. I want GHS 50,000 to start a car washing business, a
provision shop, and also import phones from Dubai. I am 22 and full of energy. I have not
started any of these yet but my friends say I am very business minded. I will pay back in
one year when the businesses are booming. No collateral but I am trustworthy.""",
}

# Gold-standard labels for three letters (for Section 4 evaluation):
GOLD = {
  "L001": {"applicant_name": "Akosua Mensah", "amount_ghs": 8000,  "purpose": "buy deep freezer / expand into frozen foods",
           "monthly_profit_ghs": 900,  "has_collateral_or_guarantor": True,  "repayment_months": 20},
  "L003": {"applicant_name": "Efua Darko",    "amount_ghs": 15000, "purpose": "industrial sewing machines and fabric stock",
           "monthly_profit_ghs": 2800, "has_collateral_or_guarantor": True,  "repayment_months": 15},
  "L006": {"applicant_name": "Kofi",          "amount_ghs": 50000, "purpose": "car wash, provision shop, phone imports",
           "monthly_profit_ghs": None, "has_collateral_or_guarantor": False, "repayment_months": 12},
}

print(f"{len(LETTERS)} letters loaded.")

# section 3.1 V1: naive prompt
SUMMARY_PROMPT_V1 = "Summarize this:"

for letter_id in ["L002", "L006"]:
    _, answer = ask_llm(f"{SUMMARY_PROMPT_V1}\n\n{LETTERS[letter_id]}", temperature=0.0)
    print(f"=== V1 — {letter_id} ===")
    print(answer, "\n")

# section 3.2 V2: role + constraints
SUMMARY_SYSTEM_V2 = (
    "You are an assistant to a microfinance loan officer in Ghana. "
    "Summarize loan application letters factually and neutrally. "
    "Do not invent, assume, or infer any detail not explicitly stated in the letter. "
    "If a detail (e.g. collateral, repayment terms) is missing, do not mention it. "
    "Write exactly 3-4 sentences."
)

def SUMMARY_PROMPT_V2(letter_text):
    return f"Summarize this loan application:\n\n{letter_text}"

for letter_id in ["L002", "L006"]:
    _, answer = ask_llm(
        SUMMARY_PROMPT_V2(LETTERS[letter_id]),
        system_prompt=SUMMARY_SYSTEM_V2,
        temperature=0.0,
    )
    print(f"=== V2 — {letter_id} ===")
    print(answer, "\n")

"""**1. What concrete problems did V1's output have that V2 fixed?**
V1 often add extra comment like "This letter is about..." instead of just facts, and it can go longer than 3-4 sentences. It also sometimes fill in small detail that is not in the letter  like guessing at repayment plan or collateral when the letter never mention it. V2 fix this because it get a clear role (loan officer assistant) and strict rule: stay factual, no inventing detail, and keep to 3-4 sentences.


**Quote examples. 2. Why is "no invented details" an essential instruction in this application? What is this failure mode called in the LLM literature?**

"No invented details" is essential because a loan officer can read the summary and make a real decision based on it, if the LLM add a fake number or detail that is not true, the officer could approve or reject a loan based on wrong information. This failure mode is called hallucination in the LLM literature when a model state something confidently that is not grounded in the actual source text.

"""

#section 3.3
import json
import pandas as pd

EXTRACT_SYSTEM = (
    "You are a data extraction assistant for a microfinance loan officer in Ghana. "
    "Extract structured fields from loan application letters. "
    "Return ONLY a valid JSON object, with no extra text, no explanation, and no markdown fences. "
    "The JSON must have EXACTLY these keys:\n"
    "  applicant_name (string)\n"
    "  amount_ghs (number)\n"
    "  purpose (string)\n"
    "  monthly_profit_ghs (number or null)\n"
    "  has_collateral_or_guarantor (boolean)\n"
    "  repayment_months (number or null)\n"
    "If a field is not explicitly stated in the letter, use null. Do not guess or infer."
)

# One worked example — NOT from the LETTERS dataset
EXAMPLE_LETTER = (
    "Dear Sir, my name is Ama Serwaa, a hairdresser in Cape Coast. "
    "I request GHS 5,000 to buy new dryers. I did not mention my monthly profit. "
    "I have no guarantor or collateral. I can repay in 10 months."
)

EXAMPLE_OUTPUT = {
    "applicant_name": "Ama Serwaa",
    "amount_ghs": 5000,
    "purpose": "buy new dryers",
    "monthly_profit_ghs": None,
    "has_collateral_or_guarantor": False,
    "repayment_months": 10
}

def EXTRACT_PROMPT(letter_text):
    return (
        f"Example letter:\n{EXAMPLE_LETTER}\n\n"
        f"Example output:\n{json.dumps(EXAMPLE_OUTPUT)}\n\n"
        f"Now extract from this letter:\n{letter_text}"
    )

def extract_fields(letter_text, temperature=0):
    _, raw = ask_llm(
        EXTRACT_PROMPT(letter_text),
        system_prompt=EXTRACT_SYSTEM,
        temperature=temperature,
    )
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"Warning: failed to parse JSON:\n{raw}")
        return None

rows = []
for letter_id, letter_text in LETTERS.items():
    result = extract_fields(letter_text)
    if result is not None:
        result["letter_id"] = letter_id
        rows.append(result)
    else:
        rows.append({"letter_id": letter_id})

df = pd.DataFrame(rows).set_index("letter_id")
df

!git clone https://github.com/phyron-gif/lab-4-llm-decision-support.git

# Commented out IPython magic to ensure Python compatibility.
# %cd lab-4-llm-decision-support

# Commented out IPython magic to ensure Python compatibility.
# %%writefile prompts.py
# """
# Prompt templates for the loan decision-support system.
# 
# Evolution notes:
# - SUMMARY_PROMPT started as a naive one-liner (V1), which gave inconsistent
#   length and sometimes invented details. V2 added a system role and explicit
#   constraints (factual, neutral, no invented details, 3-4 sentences) to fix this.
# - EXTRACT_PROMPT uses an explicit JSON schema, one few-shot example from
#   outside the dataset, and a "use null, do not guess" rule to stop the model
#   from fabricating field values.
# - BRIEF_PROMPT forbids "approve"/"reject" outputs, restricting the model to
#   strengths, risks, missing info, and next-step suggestions only, so the
#   final decision always stays with a human loan officer.
# """
# 
# SUMMARY_SYSTEM_V2 = (
#     "You are an assistant to a microfinance loan officer in Ghana. "
#     "Summarize loan application letters factually and neutrally. "
#     "Do not invent, assume, or infer any detail not explicitly stated in the letter. "
#     "If a detail is missing, do not mention it. Write exactly 3-4 sentences."
# )
# 
# def SUMMARY_PROMPT_V2(letter_text):
#     return f"Summarize this loan application:\n\n{letter_text}"
# 
# 
# EXTRACT_SYSTEM = (
#     "You are a data extraction assistant for a microfinance loan officer in Ghana. "
#     "Extract structured fields from loan application letters. "
#     "Return ONLY a valid JSON object, with no extra text, no explanation, and no markdown fences. "
#     "The JSON must have EXACTLY these keys:\n"
#     "  applicant_name (string)\n"
#     "  amount_ghs (number)\n"
#     "  purpose (string)\n"
#     "  monthly_profit_ghs (number or null)\n"
#     "  has_collateral_or_guarantor (boolean)\n"
#     "  repayment_months (number or null)\n"
#     "If a field is not explicitly stated in the letter, use null. Do not guess or infer."
# )
# 
# EXAMPLE_LETTER = (
#     "Dear Sir, my name is Ama Serwaa, a hairdresser in Cape Coast. "
#     "I request GHS 5,000 to buy new dryers. I did not mention my monthly profit. "
#     "I have no guarantor or collateral. I can repay in 10 months."
# )
# 
# EXAMPLE_OUTPUT = {
#     "applicant_name": "Ama Serwaa",
#     "amount_ghs": 5000,
#     "purpose": "buy new dryers",
#     "monthly_profit_ghs": None,
#     "has_collateral_or_guarantor": False,
#     "repayment_months": 10
# }
# 
# def EXTRACT_PROMPT(letter_text):
#     import json
#     return (
#         f"Example letter:\n{EXAMPLE_LETTER}\n\n"
#         f"Example output:\n{json.dumps(EXAMPLE_OUTPUT)}\n\n"
#         f"Now extract from this letter:\n{letter_text}"
#     )
# 
# 
# BRIEF_SYSTEM = (
#     "You are an assistant to a microfinance loan officer in Ghana. "
#     "Your job is to help the officer review an application quickly — you do NOT make "
#     "the final loan decision. The human loan officer always makes the final decision. "
#     "Base everything strictly on the letter and the extracted data provided — do not "
#     "invent facts. Structure your output as:\n"
#     "1. Strengths (bullet points, grounded in the letter)\n"
#     "2. Risks / red flags (bullet points)\n"
#     "3. Missing information the officer should request\n"
#     "4. Suggested next step — choose ONE of: 'invite for interview', 'request documents', "
#     "'flag for senior review'. NEVER output 'approve' or 'reject' — that decision belongs "
#     "to the human officer only."
# )
# 
# def BRIEF_PROMPT(letter_text, extracted_json):
#     import json
#     return (
#         f"Loan application letter:\n{letter_text}\n\n"
#         f"Extracted data:\n{json.dumps(extracted_json)}\n\n"
#         "Generate the loan officer brief as instructed."
#     )

!git config --global user.email "jinorphiron@gmail.com.com"
!git config --global user.name "phyron-gif"

!git add prompts.py
!git commit -m "Add final prompt templates for summarize/extract/brief; document evolution from naive V1 summarizer to constrained V2, add few-shot and null-guard to extractor, forbid approve/reject in brief generator"

#3.4
!git log -1 --format="%H"

"""**1. Why must the few-shot example NOT come from the six letters you are processing?**
Few-shot example must not be from the six letters: if it is, the model can just copy that pattern instead of learning the format, and it make evaluation unfair since the model already "seen" that letter.

**2. Why "use null, do not guess" — what did the model do without that instruction**
"Use null, do not guess": without it, the model tend to fill in a plausible-looking value even when the letter doesn't state it. This is risky because a fabricated number inside neat JSON looks trustworthy, so a loan officer or system might act on it without question.


**3. Why is temperature=0 the right choice for extraction but arguably not for creative tasks?**
Temperature=0 for extraction: extraction is a lookup task  same letter should give same output every time. Creative tasks benefit from variety, but extraction just need consistency, so randomness only adds risk of wrong values.
"""

BRIEF_SYSTEM = (
    "You are an assistant to a microfinance loan officer in Ghana. "
    "Your job is to help the officer review an application quickly — you do NOT make "
    "the final loan decision. The human loan officer always makes the final decision. "
    "Base everything strictly on the letter and the extracted data provided — do not "
    "invent facts. Structure your output as:\n"
    "1. Strengths (bullet points, grounded in the letter)\n"
    "2. Risks / red flags (bullet points)\n"
    "3. Missing information the officer should request\n"
    "4. Suggested next step — choose ONE of: 'invite for interview', 'request documents', "
    "'flag for senior review'. NEVER output 'approve' or 'reject' — that decision belongs "
    "to the human officer only."
)

def BRIEF_PROMPT(letter_text, extracted_json):
    return (
        f"Loan application letter:\n{letter_text}\n\n"
        f"Extracted data:\n{json.dumps(extracted_json)}\n\n"
        "Generate the loan officer brief as instructed."
    )

briefs = {}
for letter_id, letter_text in LETTERS.items():
    extracted = df.loc[letter_id].to_dict()
    _, brief = ask_llm(
        BRIEF_PROMPT(letter_text, extracted),
        system_prompt=BRIEF_SYSTEM,
        temperature=0,
    )
    briefs[letter_id] = brief

for letter_id in ["L001", "L002", "L006"]:
    print(f"=== Brief — {letter_id} ===")
    print(briefs[letter_id], "\n")

"""**1. Compare the briefs for L003 (strong application) and L006 (weak application). Did the system identify the right strengths and red flags in each?**

L003's brief correctly picked up strengths like the registered business, 18 months of sales records, steady profit, and pledged collateral, marking it as low-risk. L006's brief correctly flagged the lack of any track record, three unrelated business ideas, and no collateral or guarantor, marking it as high-risk.

**2. Why did we forbid the model from outputting "approve"/"reject"? Give one practical and one ethical reason.**
Practically, the model only sees the letter and has no access to credit history or fraud checks, so it can't make a fully informed decision. Ethically, loan decisions directly affect people's livelihoods, and letting AI make or appear to make that call removes human accountability.
"""



gold_letters = ["L001", "L003", "L006"]
fields = ["applicant_name", "amount_ghs", "purpose", "monthly_profit_ghs",
          "has_collateral_or_guarantor", "repayment_months"]

def field_matches(field, extracted_val, gold_val):
    if field == "applicant_name":
        if extracted_val is None or gold_val is None:
            return extracted_val == gold_val
        return str(extracted_val).strip().lower() == str(gold_val).strip().lower()
    elif field == "purpose":

        if extracted_val is None or gold_val is None:
            return extracted_val == gold_val
        return str(extracted_val).strip().lower() == str(gold_val).strip().lower()
    else:
        return extracted_val == gold_val

comparison = {}
for field in fields:
    row = {}
    for letter_id in gold_letters:
        extracted_val = df.loc[letter_id, field]
        gold_val = GOLD[letter_id][field]
        row[letter_id] = field_matches(field, extracted_val, gold_val)
    row["accuracy"] = sum(row[l] for l in gold_letters) / len(gold_letters)
    comparison[field] = row

accuracy_df = pd.DataFrame(comparison).T
accuracy_df

import json

def run_reliability_test(letter_id, temperature, n=5):
    results = []
    for i in range(n):
        result = extract_fields(LETTERS[letter_id], temperature=temperature)
        results.append(result)
    return results

def summarize_reliability(results):
    valid_count = sum(1 for r in results if r is not None)
    # canonical string form for comparing identical outputs
    signatures = [json.dumps(r, sort_keys=True) for r in results if r is not None]
    unique_count = len(set(signatures))
    return valid_count, unique_count

temp0_results = run_reliability_test("L004", temperature=0, n=5)
temp1_results = run_reliability_test("L004", temperature=1.0, n=5)

temp0_valid, temp0_unique = summarize_reliability(temp0_results)
temp1_valid, temp1_unique = summarize_reliability(temp1_results)

print("=== Temperature = 0.0 ===")
print(f"Valid JSON: {temp0_valid}/5")
print(f"Unique outputs: {temp0_unique}/5")
for r in temp0_results:
    print(r)

print("\n=== Temperature = 1.0 ===")
print(f"Valid JSON: {temp1_valid}/5")
print(f"Unique outputs: {temp1_unique}/5")
for r in temp1_results:
    print(r)

# --- Test 1: Ask about a detail NOT in the letter ---
letter = LETTERS["L001"]
adversarial_question = f"{letter}\n\nQuestion: What is the applicant's credit score?"

_, test1_answer = ask_llm(
    adversarial_question,
    system_prompt=SUMMARY_SYSTEM_V2,
    temperature=0,
)
print("Test 1: Credit score question (not in letter) ")
print(test1_answer)

# --- Test 2: Feed extractor an irrelevant text ---
irrelevant_text = (
    "Weather report for Accra, Ghana — Tuesday: Partly cloudy with a high of "
    "31°C and a low of 24°C. Winds from the southwest at 12 km/h. Humidity "
    "around 78%. Chance of rain in the afternoon, 40%. UV index: high."
)

test2_result = extract_fields(irrelevant_text)
print(" Test 2: Weather report fed to extractor")
print(test2_result)

"""**1. Report your extraction accuracy. Which field was hardest for the model and why? 2. What did the reliability experiment show about temperature and production systems? 3. Did your system hallucinate under probing? If yes, how could the prompt (or the system design around it) reduce the risk?**

Accuracy: Most fields hit 100% applicant_name, amount_ghs, has_collateral_or_guarantor, repayment_months. monthly_profit_ghs got 66.7% (missed on L006, where profit isn't stated). Purpose was hardest at 0%, but that's because I use exact string match on free text, my wording just doesn't match gold word-for-word, even when correct in meaning.

Reliability: temperature 0.0 and 1.0 both gave 5/5 identical output on L004. So temperature barely affects extraction consistency here but temp=0 is still safer for production, to avoid variation on messier letters.

Hallucination: no, both tests Pass. Summarizer admitted the credit score wasn't mentioned, and extractor returned all nulls on the weather report. The "don't invent, use null" instructions worked.

**Part 4.4 — Appropriateness: should this system exist?
No code in this part — just judgment, which is the scarcest skill in AI for business.Student Reasoning — Appropriateness**

**1. Letters L002 and L006 would likely be declined. If the bank fully automated decisions with your system, who could be unfairly harmed, and how? Consider applicants who write poorly in English but run solid businesses.**
**2. Loan letters contain personal data. What are the implications of sending them to a third-party API in another country? What would you check before deploying this at a real Ghanaian microfinance institution?**

**3. Name TWO concrete safeguards you would build around this system in production (think: human review points, logging, appeal processes, monitoring)**

1. Full automation would harm applicants who write poorly but run real, solid businesses the system judges writing quality, not actual creditworthiness. This hits less-educated or rural applicants hardest, exactly the people microfinance is meant to serve.
2. Sending letters to a foreign API means personal financial data leaves Ghana's jurisdiction, with unclear training practices. Before deploying, I'd check: the provider's data policy, compliance with Ghana's Data Protection Act, a proper data processing agreement, and applicant consent.
3. Two safeguards: mandatory human review before any decision (system only suggests next steps, never approves/rejects), and logging + an appeal process, so decisions are auditable and applicants can request human reconsideration.

**Prompting as engineering: How is iterating on a prompt similar to and different from iterating on the model hyperparameters you tuned in Lab 3?**
both are iterative trial-and-adjust processes, but hyperparameters just tune how well a fixed task is learned, while prompting actually defines the task itself in natural language. A bad prompt changes the whole behavior, not just the performance.

**Trust: After your Section 4 evaluation, would you trust this system to run unattended?**
Trust: I would not trust it fully unattended. The hallucination probing (4.3) passing was reassuring, but it only cover two test cases combined with the appropriateness risk in 4.4 (penalizing poorly-written but legit applications), I'd keep a human in the loop.

 **What single evaluation result most influenced your answer?
Cost and scale: Estimate (from your response.usage numbers) the tokens needed to process 1,000 applications per month. What does that imply for provider choice?**

**Looking back at the course: You have now used classical ML (Lab 2), trained neural networks (Lab 3), and used a foundation model via API (Lab 4). For a task like this one, why does calling an API beat training your own model — and when would it not?**

an API wins here because I don't have the data or compute to train a model that reasons this well from scratch prompting alone gets strong results fast. Training your own would make sense with a large dataset, need for full data privacy, or massive scale where per-request cost adds up.
"""
