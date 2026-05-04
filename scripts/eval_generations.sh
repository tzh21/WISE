model=${1:?}; shift

generated_base=/share/project/tzh/WISE/local/generated_images
export IMAGE_DIR=$generated_base/$model
export VLLM_API_BASE="http://127.0.0.1:8000/v1"
export VLLM_API_KEY="EMPTY"
export JUDGE_MODEL="Qwen3.5"
export MAX_WORKERS=96

bash eval_qwen.sh