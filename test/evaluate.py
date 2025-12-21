import os
import sys
# Add parent folder to path so we can import 'core'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.retrieval import hybrid_search
from core.generation import generate_answer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 1. THE EXAM PAPER
# Test cases based on linea.pdf content (Simple Linear Regression material)
TEST_CASES = [
    {
        "type": "Text Retrieval - Concept",
        "question": "What is the simple linear regression model equation?",
        "truth": "The simple linear regression model is y = β0 + β1x, where β0 is the intercept and β1 is the slope."
    },
    {
        "type": "Text Retrieval - Method",
        "question": "What method is used to estimate the regression parameters β0 and β1?",
        "truth": "The least squares method is used to estimate the parameters by minimizing the sum of squared vertical deviations."
    },
    {
        "type": "Specific Data - Equation",
        "question": "What is the regression equation mentioned for the relationship between variables X and Y?",
        "truth": "Y = 27.1829 - 0.297561X"
    },
    {
        "type": "Specific Data - R-Squared",
        "question": "What is the R-Squared value mentioned in the regression analysis?",
        "truth": "The R-Squared value is 76.6% or 0.766"
    },
    {
        "type": "Visual/Chart - Correlation",
        "question": "What type of correlation is shown between air content and density in the scatterplot?",
        "truth": "The scatterplot shows a strong negative correlation between air content and density."
    },
    {
        "type": "Data Interpretation",
        "question": "What is the iodine value used for in the regression example?",
        "truth": "The iodine value is the amount of iodine necessary to saturate a sample of 100 g of oil."
    },
    {
        "type": "Statistical Concept",
        "question": "What do the 95% Confidence Interval and Prediction Interval provide in regression analysis?",
        "truth": "They provide a range of values within which the true relationship is likely to lie."
    },
    {
        "type": "Negative Test - Unrelated Question",
        "question": "Who is the CEO of SpaceX?",
        "truth": "I cannot find the answer in the provided documents."
    },
    {
        "type": "Negative Test - Out of Scope",
        "question": "What is the chemical formula for water?",
        "truth": "I cannot find the answer in the provided documents."
    }
]

def grade_answer(question, truth, prediction):
    """
    Asks Llama-4-Scout to be the strict judge.
    Includes error handling for "chatty" models.
    """
    prompt = f"""
    You are a strict Evaluator. Compare the Student Answer to the Ground Truth.
    
    Question: {question}
    Ground Truth: {truth}
    Student Answer: {prediction}
    
    SCORE RULES:
    - Return 1 if the meaning matches the truth.
    - Return 0 if the answer is wrong or unrelated.
    
    OUTPUT FORMAT:
    - You must output ONLY the single digit: 0 or 1.
    - NO explanation. NO markdown. NO extra text.
    """
    
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0, # Maximum strictness
            max_tokens=5   # Cut it off if it tries to yap
        )
        
        # CLEANUP LOGIC:
        # If the model says "The score is 1", we extract just "1"
        raw_content = response.choices[0].message.content.strip()
        
        if "1" in raw_content:
            return 1
        elif "0" in raw_content:
            return 0
        else:
            print(f"⚠️ Judge output unclear: '{raw_content}'. Defaulting to 0.")
            return 0
            
    except Exception as e:
        print(f"⚠️ Grading Error: {e}")
        return 0

def run_evaluation():
    print("👨‍⚖️ Starting Evaluation...")
    score = 0
    
    for test in TEST_CASES:
        print(f"\nTesting ({test['type']}): {test['question']}")
        
        # 1. Run your System
        print(f"   🔎 Retrieving...")
        context_chunks = hybrid_search(test['question'])
        prediction = generate_answer(test['question'], context_chunks)
        
        # 2. Judge it
        grade = grade_answer(test['question'], test['truth'], prediction)
        
        if grade == 1:
            print(f"   ✅ PASS")
            score += 1
        else:
            print(f"   ❌ FAIL")
            print(f"   👉 Expected: {test['truth']}")
            print(f"   👉 Got: {prediction}")
            
            # THE NEW DEBUG SECTION
            print(f"   ---------------------------------------------------")
            print(f"   🐛 DEBUG: What did the AI actually read?")
            for i, chunk in enumerate(context_chunks[:2]):  # Show top 2 sources
                print(f"   [Source {i+1}]: {chunk['text'][:200]}...")  # Print first 200 chars
                if 'image_paths' in chunk.get('metadata', {}):
                    image_paths_str = chunk['metadata']['image_paths']
                    print(f"      (Contains Images: {image_paths_str})")
            print(f"   ---------------------------------------------------")

    accuracy = (score / len(TEST_CASES)) * 100
    print(f"\n🏆 Final Accuracy: {accuracy}%")

if __name__ == "__main__":
    run_evaluation()