### Cautiously use CUDA_LAUNCH_BLOCKING=1. It makes call to single GPU synchronously. Also, it may block the process if used on multiple GPUs due to NCCL
: ${sample_duration:="16"}
: ${sample_size:="224"}
: ${batch_size:="32"}
: ${ckpt_num:="288"}
: ${model:="resnet-18"}

if [ "${model}" == "resnet-18" ]; then
echo "Calibrating resnet-18.."
python3 main_calibration.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd.json --result_path model_ckpt/calibration --dataset pkummd --n_classes 400 --n_finetune_classes 51 --pretrain_path models/resnet-18-kinetics.pth --ft_begin_index 4 --model resnet --model_depth 18 --resnet_shortcut A --batch_size ${batch_size} --n_threads 8 --checkpoint 3 --sample_duration ${sample_duration} --no_train --sample_size ${sample_size} --resume_path $HOME/datasets/PKUMMD/model_ckpt/resnet-18/sample_duration_${sample_duration}/non_kd_train/image_size_${sample_size}/save_${ckpt_num}.pth --no_softmax_in_test
elif [ "${model}" == "resnext-101" ]; then
echo "Calibrating resnext-101.."
python3 main_calibration.py --root_path ~/datasets/PKUMMD/ --video_path rgb_frames/train --annotation_path splits/pkummd.json --result_path model_ckpt/calibration --dataset pkummd --n_classes 400 --n_finetune_classes 51 --pretrain_path models/resnext-101-64f-kinetics.pth --ft_begin_index 4 --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32 --sample_duration ${sample_duration} --batch_size ${batch_size} --n_threads 8 --checkpoint 3 --no_train --sample_size ${sample_size} --resume_path $HOME/datasets/PKUMMD/model_ckpt/resnext-101/sample_duration_${sample_duration}/image_size_${sample_size}/save_${ckpt_num}.pth --no_softmax_in_test
fi
