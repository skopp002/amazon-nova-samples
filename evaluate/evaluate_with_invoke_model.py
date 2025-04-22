import json
import pandas as pd
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nova_client import invoke_nova_with_pdf
import boto3
import time

'''
Evaluation of Nova Citations efficiency using invoke_model API 
'''

DATA_BUCKET = "datasets-veda-aiml"
MODEL_ID_LITE = "amazon.nova-lite-v1:0"
MODEL_ID_PRO = "amazon.nova-pro-v1:0"
MODEL_TO_TEST = MODEL_ID_PRO
CLAUDE_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
AWS_DEFAULT_REGION = "us-east-1"
EVAL_DATASET = "eval_single_prompt_dataset.jsonl"

def invoke_claude(prompt):
    """Invoke Claude model using Bedrock"""
    bedrock = boto3.client('bedrock-runtime',region_name=AWS_DEFAULT_REGION)
    
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.3
    }
    
    response = bedrock.invoke_model(
        modelId=CLAUDE_MODEL_ID,
        body=json.dumps(request_body)
    )
    
    response_body = json.loads(response['body'].read())
    return response_body['content'][0]['text']

def load_eval_dataset(file_path):
    """Load evaluation dataset from jsonl file"""
    eval_data = []
    with open(file_path, 'r') as file:
        for line in file:
            eval_data.append(json.loads(line))
    return eval_data

def create_evaluation_prompt(question, response, expected_answer=None):
    """Create a structured evaluation prompt focusing on accuracy and citations"""
    return f"""You are an expert evaluator. Evaluate the following response with special focus on accuracy and citations.

Question: {question}
Response to Evaluate: {response}
Expected Answer (if provided): {expected_answer if expected_answer else 'Not provided'}

Please evaluate on these specific criteria:

1. Accuracy (0-10):
   - How factually accurate is the information?
   - Does it align with the source documents?
   - Are there any factual errors or misrepresentations?

2. Citations (0-10):
   - Are claims properly supported with citations?
   - Are citations specific (e.g., mentioning years, page numbers)?
   - Are citations relevant to the claims made?

3. Overall Score (0-10):
   - Combined assessment considering both accuracy and citations

Provide your evaluation in the following JSON format:
{{
    "accuracy": {{
        "score": <0-10>,
        "explanation": "Detailed explanation of accuracy assessment",
        "errors_found": ["List any factual errors found", "if any"]
    }},
    "citations": {{
        "score": <0-10>,
        "explanation": "Detailed explanation of citation assessment",
        "missing_citations": ["List claims that need citations", "if any"]
    }},
    "overall_score": <0-10>,
    "summary": "Brief summary of the evaluation"
}}
"""

def run_batch_evaluation(eval_data, pdf_files):
    """Run batch evaluation with focus on accuracy and citations"""
    results = []
    batch_size = 5  # Process 5 items at a time
    
    for i in range(0, len(eval_data), batch_size):
        batch = eval_data[i:i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{len(eval_data)//batch_size + 1}")
        
        batch_results = []
        for item in batch:
            try:
                # Get Nova response
                nova_response = invoke_nova_with_pdf(
                    model_id=MODEL_ID_PRO,
                    question=item['prompt'],
                    pdf_files=pdf_files,
                    max_tokens=2000,
                    temperature=0.7
                )
                
                # Create evaluation prompt
                eval_prompt = create_evaluation_prompt(
                    question=item['prompt'],
                    response=nova_response,
                    expected_answer=item.get('expected_answer')
                )
                
                # Get evaluation from Claude
                evaluation = invoke_claude(eval_prompt)
                
                try:
                    evaluation_json = json.loads(evaluation)
                except json.JSONDecodeError:
                    print(f"Error parsing evaluation for prompt {item.get('id', '')}")
                    evaluation_json = {
                        "error": "Failed to parse evaluation",
                        "raw_response": evaluation
                    }
                
                # Create result entry
                result = {
                    'prompt_id': item.get('id', ''),
                    'prompt': item['prompt'],
                    'expected_answer': item.get('expected_answer', ''),
                    'nova_response': nova_response,
                    'evaluation': evaluation_json,
                    'timestamp': datetime.now().isoformat()
                }
                
                batch_results.append(result)
                
                # Add delay between items
                time.sleep(2)
                
            except Exception as e:
                print(f"Error processing prompt {item.get('id', '')}: {str(e)}")
                continue
        
        results.extend(batch_results)
        
        # Save intermediate results
        save_intermediate_results(results, i//batch_size + 1)
        
        # Add delay between batches
        if i + batch_size < len(eval_data):
            delay = 30
            print(f"\nWaiting {delay} seconds before next batch...")
            time.sleep(delay)
    
    return results

def save_intermediate_results(results, batch_num):
    """Save intermediate results after each batch"""
    filename = f'intermediate_results_batch_{batch_num}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved intermediate results to {filename}")

def analyze_evaluation_results(results):
    """Analyze evaluation results with focus on accuracy and citations"""
    total_items = len(results)
    accuracy_scores = []
    citation_scores = []
    overall_scores = []
    low_citation_items = []
    low_accuracy_items = []
    
    for item in results:
        eval_data = item.get('evaluation', {})
        
        accuracy_score = eval_data.get('accuracy', {}).get('score', 0)
        citation_score = eval_data.get('citations', {}).get('score', 0)
        overall_score = eval_data.get('overall_score', 0)
        
        accuracy_scores.append(accuracy_score)
        citation_scores.append(citation_score)
        overall_scores.append(overall_score)
        
        if accuracy_score < 7:
            low_accuracy_items.append({
                'prompt_id': item.get('prompt_id'),
                'score': accuracy_score,
                'errors': eval_data.get('accuracy', {}).get('errors_found', [])
            })
            
        if citation_score < 7:
            low_citation_items.append({
                'prompt_id': item.get('prompt_id'),
                'score': citation_score,
                'missing_citations': eval_data.get('citations', {}).get('missing_citations', [])
            })
    
    analysis = {
        'summary': {
            'total_evaluated': total_items,
            'average_accuracy': sum(accuracy_scores) / total_items if accuracy_scores else 0,
            'average_citation_score': sum(citation_scores) / total_items if citation_scores else 0,
            'average_overall_score': sum(overall_scores) / total_items if overall_scores else 0
        },
        'low_performing_items': {
            'accuracy_issues': low_accuracy_items,
            'citation_issues': low_citation_items
        }
    }
    
    return analysis

def main():
    # Configuration
    eval_dataset_path = EVAL_DATASET
    output_file = f'evaluation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    # PDF files configuration
    pdf_files = [
        {
            "bucket": DATA_BUCKET,
            "key": "amazon-shareholder-letters/All Amazon Shareholder Letters-1997.pdf"
        }
    ]
    
    try:
        # Load evaluation dataset
        print("Loading evaluation dataset...")
        eval_data = load_eval_dataset(eval_dataset_path)
        
        # Run batch evaluation
        print("Running batch evaluation...")
        results = run_batch_evaluation(eval_data, pdf_files)
        
        if results:
            # Analyze results
            print("\nAnalyzing results...")
            analysis = analyze_evaluation_results(results)
            
            # Save results and analysis
            output = {
                'results': results,
                'analysis': analysis
            }
            
            print(f"\nSaving results to {output_file}...")
            save_intermediate_results(output, output_file)
            
            # Print summary
            print("\nEvaluation Summary:")
            print(f"Total items evaluated: {analysis['summary']['total_evaluated']}")
            print(f"Average accuracy score: {analysis['summary']['average_accuracy']:.2f}")
            print(f"Average citation score: {analysis['summary']['average_citation_score']:.2f}")
            print(f"Average overall score: {analysis['summary']['average_overall_score']:.2f}")
            print(f"Items with accuracy issues: {len(analysis['low_performing_items']['accuracy_issues'])}")
            print(f"Items with citation issues: {len(analysis['low_performing_items']['citation_issues'])}")
            
            print("\nEvaluation completed successfully!")
        else:
            print("Evaluation failed to complete.")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")

if __name__ == "__main__":
    main()
