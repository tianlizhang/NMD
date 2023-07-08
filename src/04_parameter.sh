# embedding dimension
python run_exp.py --data uci --model d2v_tim --alpha 0.5 --f 32 --log f_32
python run_exp.py --data uci --model d2v_tim --alpha 0.5 --f 64 --log f_64
python run_exp.py --data uci --model d2v_tim --alpha 0.5 --f 256 --log f_256

python run_exp.py --data uci --model ddne_tim --f 32 --log f_32
python run_exp.py --data uci --model ddne_tim --f 64 --log f_64
python run_exp.py --data uci --model ddne_tim --f 256 --log f_256

python run_exp.py --data uci --model gcn_lstm_tim --f 32 --log f_32
python run_exp.py --data uci --model gcn_lstm_tim --f 64 --log f_64
python run_exp.py --data uci --model gcn_lstm_tim --f 256 --log f_256

python run_exp.py --data uci --model egcn_tim --f 32 --log f_32
python run_exp.py --data uci --model egcn_tim --f 64 --log f_64
python run_exp.py --data uci --model egcn_tim --f 256 --log f_256

python run_exp.py --data uci --model stgsn_tim --f 32 --log f_32
python run_exp.py --data uci --model stgsn_tim --f 64 --log f_64
python run_exp.py --data uci --model stgsn_tim --f 256 --log f_256


# num neg
python run_exp.py --data uci --model d2v_tim --k 10 --log k_10
python run_exp.py --data uci --model d2v_tim --k 50 --log k_50
python run_exp.py --data uci --model d2v_tim --k 200 --log k_200

python run_exp.py --data uci --model ddne_tim --k 10 --log k_10
python run_exp.py --data uci --model ddne_tim --k 50 --log k_50
python run_exp.py --data uci --model ddne_tim --k 200 --log k_200

python run_exp.py --data uci --model gcn_lstm_tim --k 10 --log k_10
python run_exp.py --data uci --model gcn_lstm_tim --k 50 --log k_50
python run_exp.py --data uci --model gcn_lstm_tim --k 200 --log k_200

python run_exp.py --data uci --model egcn_tim --k 10 --log k_10
python run_exp.py --data uci --model egcn_tim --k 50 --log k_50
python run_exp.py --data uci --model egcn_tim --k 200 --log k_200

python run_exp.py --data uci --model stgsn_tim --k 10 --log k_10
python run_exp.py --data uci --model stgsn_tim --k 50 --log k_50
python run_exp.py --data uci --model stgsn_tim --k 200 --log k_200


# lr
python run_exp.py --data uci --model d2v_tim --lr 0.03 --log lr_0.03 # 209-0
python run_exp.py --data uci --model d2v_tim --lr 0.07 --log lr_0.07

python run_exp.py --data uci --model ddne_tim --lr 0.03 --log lr_0.03
python run_exp.py --data uci --model ddne_tim --lr 0.07 --log lr_0.07

python run_exp.py --data uci --model gcn_lstm_tim --lr 0.03 --log lr_0.03
python run_exp.py --data uci --model gcn_lstm_tim --lr 0.07 --log lr_0.07

python run_exp.py --data uci --model egcn_tim --lr 0.03 --log lr_0.03
python run_exp.py --data uci --model egcn_tim --lr 0.07 --log lr_0.07

python run_exp.py --data uci --model stgsn_tim --lr 0.03 --log lr_0.03
python run_exp.py --data uci --model stgsn_tim --lr 0.07 --log lr_0.07