# number of snapshots
python run_exp.py --data uci --model d2v_tim --alpha 0.5 --w 8 --log win_8
python run_exp.py --data uci --model d2v_tim --alpha 0.5 --w 6 --log win_6
python run_exp.py --data uci --model d2v_tim --alpha 0.5 --w 4 --log win_4
python run_exp.py --data uci --model d2v_tim --alpha 0.5 --w 2 --log win_2

python run_exp.py --data uci --model ddne_tim --w 8 --log win_8
python run_exp.py --data uci --model ddne_tim --w 6 --log win_6
python run_exp.py --data uci --model ddne_tim --w 4 --log win_4
python run_exp.py --data uci --model ddne_tim --w 2 --log win_2

python run_exp.py --data uci --model gcn_lstm_tim --w 8 --log win_8
python run_exp.py --data uci --model gcn_lstm_tim --w 6 --log win_6
python run_exp.py --data uci --model gcn_lstm_tim --w 4 --log win_4
python run_exp.py --data uci --model gcn_lstm_tim --w 2 --log win_2

python run_exp.py --data uci --model egcn_tim --w 8 --log win_8
python run_exp.py --data uci --model egcn_tim --w 6 --log win_6
python run_exp.py --data uci --model egcn_tim --w 4 --log win_4
python run_exp.py --data uci --model egcn_tim --w 2 --log win_2

python run_exp.py --data uci --model stgsn_tim --w 8 --log win_8
python run_exp.py --data uci --model stgsn_tim --w 6 --log win_6
python run_exp.py --data uci --model stgsn_tim --w 4 --log win_4
python run_exp.py --data uci --model stgsn_tim --w 2 --log win_2


# weight alpha
python run_exp.py --data uci --model d2v_tim --alpha 0.2 --log alpha_0.2 # 209-0
python run_exp.py --data uci --model d2v_tim --alpha 0.5 --log alpha_0.5
python run_exp.py --data uci --model d2v_tim --alpha 1.5 --log alpha_1.5
python run_exp.py --data uci --model d2v_tim --alpha 2.0 --log alpha_2.0

python run_exp.py --data uci --model ddne_tim --alpha 0.2 --log alpha_0.2
python run_exp.py --data uci --model ddne_tim --alpha 0.5 --log alpha_0.5
python run_exp.py --data uci --model ddne_tim --alpha 1.5 --log alpha_1.5
python run_exp.py --data uci --model ddne_tim --alpha 2.0 --log alpha_2.0

python run_exp.py --data uci --model gcn_lstm_tim --alpha 0.2 --log alpha_0.2
python run_exp.py --data uci --model gcn_lstm_tim --alpha 0.5 --log alpha_0.5
python run_exp.py --data uci --model gcn_lstm_tim --alpha 1.5 --log alpha_1.5
python run_exp.py --data uci --model gcn_lstm_tim --alpha 2.0 --log alpha_2.0

python run_exp.py --data uci --model egcn_tim --alpha 0.2 --log alpha_0.2
python run_exp.py --data uci --model egcn_tim --alpha 0.5 --log alpha_0.5
python run_exp.py --data uci --model egcn_tim --alpha 1.5 --log alpha_1.5
python run_exp.py --data uci --model egcn_tim --alpha 2.0 --log alpha_2.0

python run_exp.py --data uci --model stgsn_tim --alpha 0.2 --log alpha_0.2
python run_exp.py --data uci --model stgsn_tim --alpha 0.5 --log alpha_0.5
python run_exp.py --data uci --model stgsn_tim --alpha 1.5 --log alpha_1.5
python run_exp.py --data uci --model stgsn_tim --alpha 2.0 --log alpha_2.0