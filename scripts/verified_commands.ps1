# 已验证可跑命令。无需激活虚拟环境，直接复制执行。

cd "D:\项目\源文件\deploy\FreTS-main"
..\.venv-frets\Scripts\python.exe run_longExp.py --data covid --root_path ./dataset/ --data_path covid.csv --features M --enc_in 55 --dec_in 55 --c_out 55 --seq_len 36 --label_len 18 --pred_len 24 --train_epochs 3 --batch_size 4 --use_gpu False

cd "D:\项目\源文件\deploy\Time-GNN-main"
..\.venv-timegnn\Scripts\python.exe TimeGNN_train.py

cd "D:\项目\源文件\deploy\sep-main"
..\.venv-sep\Scripts\python.exe -c "from exp.exp_model import Exp_Model; print('SEP Exp_Model import ok')"
