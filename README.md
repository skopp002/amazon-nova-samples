# nova-citations



## Setting up

```
git clone git@ssh.gitlab.aws.dev:skoppar/nova-citations.git
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to use this solution:
[nova_client.py](./nova_client.py) provides a programmatic way to test citations with Nova model

# Usage
The client supports evaluating prompts against PDF documents using Amazon Bedrock's Nova model.

## Features
- PDF document citation support
- Configurable model parameters
- Batch processing capabilities

## Example
```python
from nova_client import invoke_nova_with_pdf

response = invoke_nova_with_pdf(
    model_id="amazon.nova-pro-v1:0",
    question="What is Amazon's leadership principle about customer obsession?",
    pdf_files=[{
        "bucket": "XXXXXXXXXXX",
        "key": "path/to/document.pdf"
    }]
) 
```

## Evaluation
