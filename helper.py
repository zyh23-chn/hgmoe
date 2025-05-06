import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='mm-graph-org/mm-graph',
    repo_type='dataset',
    local_dir='./data',
)

# snapshot_download(
#     repo_id='mm-graph-org/mm-graph/ele-fashion',
#     repo_type='dataset',
#     local_dir='./data',
#     resume_download=True,
# )
