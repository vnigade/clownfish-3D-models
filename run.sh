### Cautiously use CUDA_LAUNCH_BLOCKING=1. It makes call to single GPU synchronously. Also, it may block the process if used on multiple GPUs due to NCCL.
: ${sample_duration:="16"}
: ${sample_size:="224"}
: ${batch_size:="16"}
: ${ckpt_num:="288"}
: ${window_stride:="4"}
: ${model:="$1"}
: ${siminet_path:=''}
: ${predict_type:='val'} # [train, val]
: ${run_type:='predict'} # [train, predict]
: ${split_type:='cross_subject'} # [cross_subject, cross_view]


# Train Resnext-101 
if [[ "${model}" == "resnext-101" && "${run_type}" == "train" ]]; then
echo "Training model resnext-101..."
python3 main.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd_${split_type}.json --result_path model_ckpt/resnext-101/sample_duration_${sample_duration}/image_size_${sample_size}/${split_type} --dataset pkummd --n_classes 400 --n_finetune_classes 51 --pretrain_path models/resnext-101-64f-kinetics.pth --ft_begin_index 4 --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32 --sample_duration ${sample_duration} --batch_size ${batch_size} --n_threads 8 --checkpoint 3 --no_val --sample_size ${sample_size} --n_epochs 300 # --resume_path $HOME/datasets/PKUMMD/model_ckpt/resnext-101/sample_duration_${sample_duration}/image_size_${sample_size}/${split_type}/save_${ckpt_num}.pth

# Predict Resnext-101
elif [[ "${model}" == "resnext-101" && "${run_type}" == "predict" ]]; then
echo "Predicting model resnext-101..."
python3 predict_window.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd_${split_type}.json --result_path model_ckpt --dataset pkummd --n_classes 51 --ft_begin_index 4 --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32 --sample_duration ${sample_duration} --batch_size 8 --n_threads 8 --checkpoint 3 --no_val --resume_path $HOME/datasets/PKUMMD/model_ckpt/resnext-101/sample_duration_${sample_duration}/image_size_${sample_size}/${split_type}/save_${ckpt_num}.pth --no_train --test_subset ${predict_type} --test --window_size=${sample_duration} --window_stride=${window_stride} --scores_dump_path scores_dump/resnext-101/sample_duration_${sample_duration}/image_size_${sample_size}/window_${window_stride}/${split_type}/${predict_type}/ --sample_size ${sample_size}

# Predict Resnet-18 
elif [[ "${model}" == "resnet-18" && "${run_type}" == "predict" ]]; then
echo "Predicting model resnet-18..."
python3 predict_window.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd_${split_type}.json --result_path model_ckpt --dataset pkummd --n_classes 51 --ft_begin_index 4 --model resnet --model_depth 18 --resnet_shortcut A --batch_size 1 --n_threads 8 --checkpoint 5 --sample_duration ${sample_duration} --resume_path $HOME/datasets/PKUMMD/model_ckpt/resnet-18/sample_duration_${sample_duration}/non_kd_train/image_size_${sample_size}/${split_type}/save_${ckpt_num}.pth --no_train --no_val --test_subset ${predict_type} --test --window_size=${sample_duration} --window_stride=${window_stride} --scores_dump_path scores_dump/resnet-18/sample_duration_${sample_duration}/image_size_${sample_size}/window_${window_stride}/${split_type}/${predict_type}/raw_features/ --sample_size ${sample_size} --no_softmax_in_test # --resume_path_sim ${siminet_path}

# Train: Resnet-18
elif [[ "${model}" == "resnet-18" && "${run_type}" == "train" ]]; then
echo "Training model resnet-18..."
python3 main.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd_${split_type}.json --result_path model_ckpt/resnet-18/sample_duration_${sample_duration}/non_kd_train/image_size_${sample_size}/${split_type}/ --dataset pkummd --n_classes 400 --n_finetune_classes 51 --pretrain_path models/resnet-18-kinetics.pth --ft_begin_index 4 --model resnet --model_depth 18 --resnet_shortcut A --batch_size ${batch_size} --n_threads 8 --checkpoint 3 --sample_duration ${sample_duration} --no_val --n_epochs 300 --sample_size ${sample_size} --resume_path $HOME/datasets/PKUMMD/model_ckpt/resnet-18/sample_duration_${sample_duration}/non_kd_train/image_size_${sample_size}/${split_type}/save_${ckpt_num}.pth

# Train SimiNet
elif [ "${model}" == "siminet" ]; then
echo "Training siminet for resnet-18.."
python3 main_siminet.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd_${split_type}.json --result_path model_ckpt/siminet/resnet-18/sample_duration_${sample_duration}/image_size_${sample_size}/bce_loss/${split_type} --dataset pkummd_sim --n_classes 400 --n_finetune_classes 51 --pretrain_path models/resnet-18-kinetics.pth --ft_begin_index 4 --model resnet --model_depth 18 --resnet_shortcut A --batch_size ${batch_size} --n_threads 8 --checkpoint 3 --sample_duration ${sample_duration} --no_val --sample_size ${sample_size} --resume_path $HOME/datasets/PKUMMD/model_ckpt/resnet-18/sample_duration_${sample_duration}/non_kd_train/image_size_${sample_size}/${split_type}/save_${ckpt_num}.pth --learning_rate 0.001 # --resume_path_sim ~/datasets/PKUMMD/model_ckpt/siminet/resnet-18/sample_duration_${sample_duration}/image_size_${sample_size}/bce_loss/${split_type}/save_84.pth

# Train SimiNet for mobilenet 
elif [ "${model}" == "siminet-mobilenet" ]; then
echo "Training siminet for mobilenet.."
python3 main_siminet.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd_${split_type}.json --result_path model_ckpt/siminet/mobilenet/sample_duration_${sample_duration}/image_size_${sample_size}/bce_loss/ --dataset pkummd_sim --n_classes 600 --n_finetune_classes 51 --pretrain_path models/kinetics_mobilenet_1.0x_RGB_${sample_duration}_best.pth --ft_begin_index 4 --model mobilenet --model_depth 1 --batch_size ${batch_size} --n_threads 8 --checkpoint 3 --sample_duration ${sample_duration} --no_val --sample_size ${sample_size} --resume_path $HOME/datasets/PKUMMD/model_ckpt/mobilenet/sample_duration_${sample_duration}/image_size_${sample_size}/save_${ckpt_num}.pth --learning_rate 0.001 # --resume_path_sim ${siminet_path}

# Train MobileNet for early discard
elif [[ "${model}" == "mobilenet-early-discard" && "${run_type}" == "train" ]]; then
echo "Training model mobilenet-early-discard..."
sample_size=112
python3 main_early_discard.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd_${split_type}.json --result_path model_ckpt/${model}/sample_duration_${sample_duration}/image_size_${sample_size}/ --dataset pkummd_ed --n_classes 600 --n_finetune_classes 1 --pretrain_path models/kinetics_mobilenet_1.0x_RGB_${sample_duration}_best.pth --ft_begin_index 4 --model mobilenet --model_depth 1 --batch_size ${batch_size} --n_threads 8 --checkpoint 3 --sample_duration ${sample_duration} --no_val --sample_size ${sample_size} --n_epochs 300 --learning_rate 0.01 # --resume_path $HOME/datasets/PKUMMD/model_ckpt/${model}/sample_duration_${sample_duration}/image_size_${sample_size}/save_${ckpt_num}.pth

# Predict MobileNet for early discard
elif [[ "${model}" == "mobilenet-early-discard" && "${run_type}" == "predict" ]]; then
echo "Predicting model mobilenet-early-discard..."
sample_size=112
python3 main_early_discard.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd_${split_type}.json --result_path model_ckpt/${model}/sample_duration_${sample_duration}/image_size_${sample_size}/ --dataset pkummd_ed --n_classes 600 --n_finetune_classes 1 --pretrain_path models/kinetics_mobilenet_1.0x_RGB_${sample_duration}_best.pth --ft_begin_index 4 --model mobilenet --model_depth 1 --batch_size ${batch_size} --n_threads 8 --checkpoint 3 --sample_duration ${sample_duration} --no_train --sample_size ${sample_size} --n_epochs 300 --learning_rate 0.01 --resume_path $HOME/datasets/PKUMMD/model_ckpt/${model}/sample_duration_${sample_duration}/image_size_${sample_size}/save_${ckpt_num}.pth

# Train MobileNet
elif [[ "${model}" == "mobilenet" && "${run_type}" == "train" ]]; then
echo "Training model mobilenet..."
sample_size=112
PROG="main.py"
python3 ${PROG} --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd_${split_type}.json --result_path model_ckpt/${model}/sample_duration_${sample_duration}/image_size_${sample_size}/ --dataset pkummd --n_classes 600 --n_finetune_classes 51 --pretrain_path models/kinetics_mobilenet_1.0x_RGB_${sample_duration}_best.pth --ft_begin_index 4 --model mobilenet --model_depth 1  --batch_size ${batch_size} --n_threads 8 --checkpoint 3 --sample_duration ${sample_duration} --no_val --sample_size ${sample_size} --train_crop random --n_epochs 300 --learning_rate 0.01 --resume_path $HOME/datasets/PKUMMD/model_ckpt/${model}/sample_duration_${sample_duration}/image_size_${sample_size}/save_${ckpt_num}.pth

# Predict MobileNet
elif [[ "${model}" == "mobilenet" && "${run_type}" == "predict" ]]; then
echo "Predicting model mobilenet..."
sample_size=112
PROG="predict_window.py"
python3 ${PROG} --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd_${split_type}.json --result_path model_ckpt/${model}/sample_duration_${sample_duration}/image_size_${sample_size}/ --dataset pkummd --n_classes 600 --n_finetune_classes 51 --pretrain_path models/kinetics_mobilenet_1.0x_RGB_${sample_duration}_best.pth --ft_begin_index 4 --model mobilenet --model_depth 1  --batch_size ${batch_size} --n_threads 8 --sample_duration ${sample_duration} --no_train --no_val --test_subset ${predict_type} --test --window_size=${sample_duration} --window_stride=${window_stride} --sample_size ${sample_size} --resume_path $HOME/datasets/PKUMMD/model_ckpt/${model}/sample_duration_${sample_duration}/image_size_${sample_size}/save_${ckpt_num}.pth --scores_dump_path scores_dump/${model}/sample_duration_${sample_duration}/image_size_${sample_size}/window_${window_stride}/${predict_type}/raw_features/ --no_softmax_in_test
 
fi
