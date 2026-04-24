set -a
source .env
set +a

run_id=$(python script/setup.py --model_config config/kaec/edmdc_config.yaml | tail -n 1)
echo "Run ID: $run_id"
python script/process.py --run_id $run_id
python script/train.py --run_id $run_id
python script/validate.py --run_id $run_id --n_test 10
