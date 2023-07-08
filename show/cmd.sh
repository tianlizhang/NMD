python cites_mrr.py --data as --model d2v --epoch 43
python cites_mrr.py --data as --model d2v_tim --epoch 48

python cites_mrr.py --data as --model ddne --epoch 49
python cites_mrr.py --data as --model ddne_tim --epoch 42 --gid 0

python cites_mrr.py --data as --model gcn_lstm --epoch 48
python cites_mrr.py --data as --model gcn_lstm_tim --epoch 32

python cites_mrr.py --data as --model egcn --epoch 48
python cites_mrr.py --data as --model egcn_tim --epoch 9

python cites_mrr.py --data as --model stgsn --epoch 46
python cites_mrr.py --data as --model stgsn_tim --epoch 18




python cites_mrr.py --data bco --model d2v_tim --epoch 3

python cites_mrr.py --data bco --model ddne --epoch 1
python cites_mrr.py --data bco --model ddne_tim --epoch 10

python cites_mrr.py --data bco --model gcn_lstm --epoch 1
python cites_mrr.py --data bco --model gcn_lstm_tim --epoch 0

python cites_mrr.py --data bco --model egcn --epoch 4
python cites_mrr.py --data bco --model egcn_tim --epoch 1

python cites_mrr.py --data bco --model stgsn --epoch 1 --gid 0
python cites_mrr.py --data bco --model stgsn_tim --epoch 7 --gid 0