# uci
python TIP.py --data uci --model d2v --epoch 1
python TIP.py --data uci --model d2v_tim --epoch 8

python TIP.py --data uci --model ddne --epoch 1
python TIP.py --data uci --model ddne_tim --epoch 14

python TIP.py --data uci --model gcn_lstm --epoch 24
python TIP.py --data uci --model gcn_lstm_tim --epoch 0

python TIP.py --data uci --model egcn --epoch 0
python TIP.py --data uci --model egcn_tim --epoch 48

python TIP.py --data uci --model stgsn --epoch 11
python TIP.py --data uci --model stgsn_tim --epoch 2


# dblp
python TIP.py --data dblp --model d2v --epoch 13
python TIP.py --data dblp --model d2v_tim --epoch 40

python TIP.py --data dblp --model ddne --epoch 12
python TIP.py --data dblp --model ddne_tim --epoch 13

python TIP.py --data dblp --model gcn_lstm --epoch 41
python TIP.py --data dblp --model gcn_lstm_tim --epoch 15

python TIP.py --data dblp --model egcn --epoch 47
python TIP.py --data dblp --model egcn_tim --epoch 1 --gid 1

python TIP.py --data dblp --model stgsn --epoch 0
python TIP.py --data dblp --model stgsn_tim --epoch 41



# aps
python TIP.py --data aps --model d2v --epoch 43
python TIP.py --data aps --model d2v_tim --epoch 48

python TIP.py --data aps --model ddne --epoch 49
python TIP.py --data aps --model ddne_tim --epoch 42

python TIP.py --data aps --model gcn_lstm --epoch 48
python TIP.py --data aps --model gcn_lstm_tim --epoch 32

python TIP.py --data aps --model egcn --epoch 48
python TIP.py --data aps --model egcn_tim  --gid 0 --epoch 9

python TIP.py --data aps --model stgsn --epoch 46
python TIP.py --data aps --model stgsn_tim --epoch 18



# bco
python TIP.py --data bco --model d2v --epoch 1
python TIP.py --data bco --model d2v_tim --epoch 3

python TIP.py --data bco --model ddne --epoch 1
python TIP.py --data bco --model ddne_tim --epoch 10

python TIP.py --data bco --model gcn_lstm --epoch 1
python TIP.py --data bco --model gcn_lstm_tim --epoch 0

python TIP.py --data bco --model egcn --epoch 4
python TIP.py --data bco --model egcn_tim  --gid 1 --epoch 1

python TIP.py --data bco --model stgsn --epoch 1
python TIP.py --data bco --model stgsn_tim --epoch 7



# as
python TIP.py --data as --model d2v --epoch 43
python TIP.py --data as --model d2v_tim --epoch 48

python TIP.py --data as --model ddne --epoch 49
python TIP.py --data as --model ddne_tim --epoch 42

python TIP.py --data as --model gcn_lstm --epoch 48
python TIP.py --data as --model gcn_lstm_tim --epoch 32

python TIP.py --data as --model egcn --epoch 48
python TIP.py --data as --model egcn_tim  --gid 0 --epoch 9

python TIP.py --data as --model stgsn --epoch 46
python TIP.py --data as --model stgsn_tim --epoch 18