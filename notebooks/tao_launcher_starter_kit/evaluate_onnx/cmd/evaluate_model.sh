cd ~/projects/tao_tutorials/notebooks/tao_launcher_starter_kit/evaluate_onnx
uv run eval_reid_onnx.py \
    --model 'models/resnet50_market1501_aicity156.onnx'

cd ~/projects/tao_tutorials/notebooks/tao_launcher_starter_kit/evaluate_onnx
uv run eval_reid_onnx.py \
    --model 'tao_pretrained_models/market1501/resnet50_market1501_model.onnx'


cd ~/projects/tao_tutorials/notebooks/tao_launcher_starter_kit/evaluate_onnx
uv run eval_reid_onnx.py \
    --model 'tao_pretrained_models/market1501/resnet50_market1501_model.onnx' \
    --query-dir '../data/reidentificationnet/market1501/sample_market1501/sample_query' \
    --gallery-dir '../data/reidentificationnet/market1501/sample_market1501/sample_test'


cd ~/projects/tao_tutorials/notebooks/tao_launcher_starter_kit/evaluate_onnx
uv run eval_reid_onnx.py \
    --model 'models/resnet50_market1501_aicity156.onnx' \
    --query-dir '../data/reidentificationnet/market1501/sample_market1501/sample_query' \
    --gallery-dir '../data/reidentificationnet/market1501/sample_market1501/sample_test'

