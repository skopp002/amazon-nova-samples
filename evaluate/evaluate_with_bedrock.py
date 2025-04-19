import boto3
from datetime import datetime
import os

# Generate unique name for the job
job_name = f"Model-evaluation-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

# Configure your knowledge base and model settings
evaluator_model = "mistral.mistral-large-2402-v1:0"
generator_model = "amazon.nova-pro-v1:0"
account_id = os.environ.get('AWS_ACCOUNT_ID')
role_name = os.environ.get('AWS_ROLE_NAME')
ROLE_ARN = f"arn:aws:iam::{account_id}:role/{role_name}"
DATA_BUCKET = "nova-citations"
OUTPUT_DIR = "evaluation_output"
INPUT_DATASET = "eval_dataset.jsonl"

# Specify S3 locations for evaluation data and output
input_data = f"s3://{DATA_BUCKET}/evaluation_data/input.jsonl"
output_path = f"s3://{DATA_BUCKET}/evaluation_output/"


# Generator Models
GENERATOR_MODELS = [
    "anthropic.claude-3-haiku-20240307-v1:0",
    "amazon.nova-micro-v1:0"
]


from typing import List, Dict, Any

def get_aws_session():
    """
    Create AWS session from environment variables.
    Required env vars: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
    Optional env vars: AWS_REGION, AWS_SESSION_TOKEN
    """
    required_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return boto3.Session(
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION', 'us-east-1'),  # default to us-east-1 if not specified
        aws_session_token=os.getenv('AWS_SESSION_TOKEN')  # optional session token
    )
bedrock_client = get_aws_session().client('bedrock')    
    
def upload_to_s3(local_file: str, bucket: str, s3_key: str) -> bool:
    """
    Upload a file to S3 with error handling.
    
    Returns:
        bool: Success status
    """
    try:
        s3_client = get_aws_session().client('s3')
        s3_client.upload_file(local_file, bucket, s3_key)
        print(f"✓ Successfully uploaded to s3://{bucket}/{s3_key}")
        return True
    except Exception as e:
        print(f"✗ Error uploading to S3: {str(e)}")
        return False


def create_llm_judge_evaluation(
    client,
    job_name: str,
    role_arn: str,
    input_s3_uri: str,
    output_s3_uri: str,
    evaluator_model_id: str,
    generator_model_id: str,
    dataset_name: str = None,
    task_type: str = "General" # must be General for LLMaaJ
):    
    # All available LLM-as-judge metrics
    llm_judge_metrics = [
        "Builtin.Correctness",
        "Builtin.Completeness", 
        "Builtin.Faithfulness",
        "Builtin.Helpfulness",
        "Builtin.Coherence",
        "Builtin.Relevance",
        "Builtin.FollowingInstructions",
        "Builtin.ProfessionalStyleAndTone",
        "Builtin.Harmfulness",
        "Builtin.Stereotyping",
        "Builtin.Refusal"
    ]

    # Configure dataset
    dataset_config = {
        "name": dataset_name or "CustomDataset",
        "datasetLocation": {
            "s3Uri": input_s3_uri
        }
    }

    try:
        response = client.create_evaluation_job(
            jobName=job_name,
            roleArn=role_arn,
            applicationType="ModelEvaluation",
            evaluationConfig={
                "automated": {
                    "datasetMetricConfigs": [
                        {
                            "taskType": task_type,
                            "dataset": dataset_config,
                            "metricNames": llm_judge_metrics
                        }
                    ],
                    "evaluatorModelConfig": {
                        "bedrockEvaluatorModels": [
                            {
                                "modelIdentifier": evaluator_model_id
                            }
                        ]
                    }
                }
            },
            inferenceConfig={
                "models": [
                    {
                        "bedrockModel": {
                            "modelIdentifier": generator_model_id
                        }
                    }
                ]
            },
            outputDataConfig={
                "s3Uri": output_s3_uri
            }
        )
        return response
        
    except Exception as e:
        print(f"Error creating evaluation job: {str(e)}")
        raise
        
 # Create evaluation job
try:
    # Create the bedrock client using the session
    
    llm_as_judge_response = create_llm_judge_evaluation(
        client=bedrock_client,
        job_name=job_name,
        role_arn=ROLE_ARN,
        input_s3_uri=input_data,
        output_s3_uri=output_path,
        evaluator_model_id=evaluator_model,
        generator_model_id=generator_model,
        task_type="General"
    )
    print(f"✓ Created evaluation job: {llm_as_judge_response['jobArn']}")
except Exception as e:
    print(f"✗ Failed to create evaluation job: {str(e)}")
    raise


# Get job ARN based on job type
evaluation_job_arn = llm_as_judge_response['jobArn']
# Check job status
check_status = bedrock_client.get_evaluation_job(jobIdentifier=evaluation_job_arn) 
print(f"Job Status: {check_status['status']}")

# Consistent Evaluator
EVALUATOR_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"

def run_model_comparison(
    generator_models: List[str],
    evaluator_model: str
) -> List[Dict[str, Any]]:
    evaluation_jobs = []
    
    for generator_model in generator_models:
        job_name = f"llmaaj-{generator_model.split('.')[0]}-{evaluator_model.split('.')[0]}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        
        try:
            response = create_llm_judge_evaluation(
                client=bedrock_client,
                job_name=job_name,
                role_arn=ROLE_ARN,
                input_s3_uri=input_data,
                output_s3_uri=f"{output_path}/{job_name}/",
                evaluator_model_id=evaluator_model,
                generator_model_id=generator_model,
                task_type="General"
            )
            
            job_info = {
                "job_name": job_name,
                "job_arn": response["jobArn"],
                "generator_model": generator_model,
                "evaluator_model": evaluator_model,
                "status": "CREATED"
            }
            evaluation_jobs.append(job_info)
            
            print(f"✓ Created job: {job_name}")
            print(f"  Generator: {generator_model}")
            print(f"  Evaluator: {evaluator_model}")
            print("-" * 80)
            
        except Exception as e:
            print(f"✗ Error with {generator_model}: {str(e)}")
            continue
            
    return evaluation_jobs




if __name__ == "__main__":
    
    # Upload dataset
    s3_key =f"{OUTPUT_DIR}/{INPUT_DATASET}"
    upload_success = upload_to_s3(INPUT_DATASET, DATA_BUCKET, s3_key)
    if not upload_success:
        raise Exception("Failed to upload dataset to S3")
    # Run model comparison
    evaluation_jobs = run_model_comparison(GENERATOR_MODELS, EVALUATOR_MODEL)

    # Print results
    print("\nEvaluation Jobs Summary:")
    for job in evaluation_jobs:
        print(f"Job Name: {job['job_name']}")
        print(f"Job ARN: {job['job_arn']}")
        print(f"Generator Model: {job['generator_model']}")
        print(f"Evaluator Model: {job['evaluator_model']}")
        print(f"Status: {job['status']}")
        print("-" * 80)