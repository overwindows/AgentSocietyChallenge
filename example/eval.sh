#conda activate websocietysimulator

python3 RecAgent_baseline_v1.py  --data-dir /mnt/q/AgentRecBench/process_data/output_data_all --task-type classic --task-set amazon
python3 RecAgent_baseline_v1.py  --data-dir /mnt/q/AgentRecBench/process_data/output_data_all --task-type classic --task-set goodreads
python3 RecAgent_baseline_v1.py  --data-dir /mnt/q/AgentRecBench/process_data/output_data_all --task-type classic --task-set yelp

python3 RecAgent_baseline_v1.py  --data-dir /mnt/q/AgentRecBench/process_data/output_data_all --task-type item_cold_start --task-set amazon
python3 RecAgent_baseline_v1.py  --data-dir /mnt/q/AgentRecBench/process_data/output_data_all --task-type item_cold_start --task-set goodreads
python3 RecAgent_baseline_v1.py  --data-dir /mnt/q/AgentRecBench/process_data/output_data_all --task-type item_cold_start --task-set yelp

python3 RecAgent_baseline_v1.py  --data-dir /mnt/q/AgentRecBench/process_data/output_data_all --task-type user_cold_start --task-set amazon
python3 RecAgent_baseline_v1.py  --data-dir /mnt/q/AgentRecBench/process_data/output_data_all --task-type user_cold_start --task-set goodreads
python3 RecAgent_baseline_v1.py  --data-dir /mnt/q/AgentRecBench/process_data/output_data_all --task-type user_cold_start --task-set yelp

python3 RecAgent_baseline_v1.py  --data-dir /mnt/q/AgentRecBench/process_data/output_data_long --task-type long_term --task-set amazon
python3 RecAgent_baseline_v1.py  --data-dir /mnt/q/AgentRecBench/process_data/output_data_long --task-type long_term --task-set goodreads
python3 RecAgent_baseline_v1.py  --data-dir /mnt/q/AgentRecBench/process_data/output_data_long --task-type long_term --task-set yelp

python3 RecAgent_baseline_v1.py  --data-dir /mnt/q/AgentRecBench/process_data/output_data_short --task-type short_term --task-set amazon
python3 RecAgent_baseline_v1.py  --data-dir /mnt/q/AgentRecBench/process_data/output_data_short --task-type short_term --task-set goodreads
python3 RecAgent_baseline_v1.py  --data-dir /mnt/q/AgentRecBench/process_data/output_data_short --task-type short_term --task-set yelp